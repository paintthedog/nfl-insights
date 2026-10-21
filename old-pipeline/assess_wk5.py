import re, numpy as np, pandas as pd, lines_model as L, nflreadpy as nfl
W=5
# ---------- GAME LINES (walk-forward: fit on games before week 5)
pbp, sched = L.load(); g = L.build(pbp, sched)
tr = g[g.result.notna() & g.spread_line.notna() & (g.min_games>=3) & ((g.season<2026)|((g.season==2026)&(g.week<W)))]
mm, mt = L.fit(tr)
te = g[(g.season==2026)&(g.week==W)&g.result.notna()].copy()
o = L.predict(te, mm, mt)
o["mdl_side_cover"] = np.sign(o.m_margin - o.spread_line) == np.sign(o.result - o.spread_line)
o["mdl_ou_hit"] = np.sign(o.m_total - o.total_line) == np.sign(o.total - o.total_line)
print("=== GAMES (n=%d)" % len(o))
print(f"margin MAE: model {np.mean(np.abs(o.result-o.m_margin)):.1f} | closing spread {np.mean(np.abs(o.result-o.spread_line)):.1f}")
print(f"total  MAE: model {np.mean(np.abs(o.total-o.m_total)):.1f} | closing total {np.mean(np.abs(o.total-o.total_line)):.1f} | model total bias (actual-model) {np.mean(o.total-o.m_total):+.1f}, market bias {np.mean(o.total-o.total_line):+.1f}")
print(f"model side ATS {o.mdl_side_cover.sum()}-{(~o.mdl_side_cover).sum()} | model O/U side {o.mdl_ou_hit.sum()}-{(~o.mdl_ou_hit).sum()} | straight-up winner {(np.sign(o.m_margin)==np.sign(o.result)).sum()}/{len(o)} vs favorites {(np.sign(o.spread_line)==np.sign(o.result)).sum()}/{len(o)}")
print(o[["game_id","spread_line","m_margin","result","total_line","m_total","total"]].round(1).to_string(index=False))
# ---------- ATD
p = nfl.load_pbp([2026]).to_pandas(); p=p[p.week==W]; np2=p[p.two_point_attempt.ne(1)]
done=set(p.groupby("game_id").desc.last().loc[lambda s:s.str.contains("END GAME",na=False)].index)
pl=pd.read_parquet("data/players.parquet",columns=["gsis_id","display_name"])
norm=lambda s: re.sub(r"\s+"," ",re.sub(r"[^a-z ]"," ",re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?","",str(s).lower().replace("'","")))).strip()
NM={}; [NM.setdefault(norm(n),[]).append(gg) for gg,n in zip(pl.gsis_id,pl.display_name)]
touched=set(np2.passer_player_id.dropna())|set(np2.rusher_player_id.dropna())|set(np2.receiver_player_id.dropna())|set(np2.td_player_id.dropna())
def pid(n,gid):
    c=NM.get(norm(n),[]); 
    if len(c)==1: return c[0]
    gm=np2[np2.game_id==gid]; ids=set(gm.passer_player_id.dropna())|set(gm.rusher_player_id.dropna())|set(gm.receiver_player_id.dropna())
    h=[x for x in c if x in ids]; return h[0] if h else (c[0] if c else None)
tds=np2[np2.touchdown.eq(1)&np2.td_player_id.notna()].groupby("td_player_id").size()
a=pd.read_csv("props_log.csv"); a=a[(a.week==W)&(a.pulled_at=="2026-10-11T11:22Z")&a.model_p.notna()&a.game_id.isin(done)].copy()
a["pid"]=[pid(n,gid) for n,gid in zip(a.player,a.game_id)]; a=a[a.pid.isin(touched)]
a["hit"]=a.pid.map(tds).fillna(0).gt(0).astype(int)
a["mkt"]=1/a.price; a["mkt"]=a.mkt*0.85   # rough vig removal for ATD (DK ATD boards carry ~15%+ overround)
bm=np.mean((a.model_p-a.hit)**2); bk=np.mean((a.mkt-a.hit)**2)
print(f"\n=== ANYTIME TD (n={len(a)} priced players): hits {a.hit.sum()} | model expected {a.model_p.sum():.1f} | market (de-vigged) expected {a.mkt.sum():.1f}")
print(f"Brier: model {bm:.4f}  market {bk:.4f}  (lower is better)")
a["band"]=pd.cut(a.model_p,[0,.1,.2,.3,.45,1],labels=["<10%","10-20%","20-30%","30-45%","45%+"])
print(a.groupby("band",observed=True).agg(n=("hit","size"),model=("model_p","mean"),market=("mkt","mean"),actual=("hit","mean")).round(2).to_string())
a.to_csv("/tmp/claude-0/-home-claude/61a3e72b-dbd0-5c34-8049-91e3f7bfb7b5/scratchpad/atd_wk5_full.csv",index=False)
# ---------- PLAYER STAT LINES
ST={"player_pass_yds":np2.groupby("passer_player_id").passing_yards.sum(),"player_rush_yds":np2.groupby("rusher_player_id").rushing_yards.sum(),
    "player_reception_yds":np2.groupby("receiver_player_id").receiving_yards.sum(),"player_receptions":np2.groupby("receiver_player_id").complete_pass.sum()}
u=pd.read_csv("props_log_ou.csv"); u=u[(u.week==W)&(u.pulled_at=="2026-10-11T11:51Z")&u.model_mean.notna()&u.game_id.isin(done)].copy()
u["pid"]=[pid(n,gid) for n,gid in zip(u.player,u.game_id)]; u=u[u.pid.isin(touched)]
u["act"]=[float(ST[m].get(x,0)) for m,x in zip(u.market,u.pid)]
mk={"player_pass_yds":"Pass yds","player_rush_yds":"Rush yds","player_reception_yds":"Rec yds","player_receptions":"Receptions"}
print(f"\n=== PLAYER STATS vs DK lines (n={len(u)})")
for m_,x in u.groupby("market"):
    x=x.copy(); x["tier"]=pd.qcut(x.point,2,labels=["lower lines","higher lines"],duplicates="drop")
    print(f"{mk[m_]:<11} MAE model {np.mean(np.abs(x.act-x.model_mean)):6.1f} | DK line {np.mean(np.abs(x.act-x.point)):6.1f} | model bias {np.mean(x.act-x.model_mean):+6.1f} | line bias {np.mean(x.act-x.point):+6.1f}")
    for t,y in x.groupby("tier",observed=True):
        print(f"     {t:<13} n={len(y):>3}  actual {y.act.mean():6.1f}  model {y.model_mean.mean():6.1f}  line {y.point.mean():6.1f}")
