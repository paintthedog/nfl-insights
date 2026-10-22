import re, pandas as pd, numpy as np, nflreadpy as nfl
W=5
p = nfl.load_pbp([2026]).to_pandas(); p = p[p.week==W]
done = set(p.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
np2 = p[p.two_point_attempt.ne(1)]
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id","display_name"])
norm = lambda s: re.sub(r"\s+"," ",re.sub(r"[^a-z ]"," ",re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?","",str(s).lower().replace("'","")))).strip()
NAME={}
for g,n in zip(pl.gsis_id, pl.display_name): NAME.setdefault(norm(n),[]).append(g)
touched = set(np2.passer_player_id.dropna())|set(np2.rusher_player_id.dropna())|set(np2.receiver_player_id.dropna())|set(np2.td_player_id.dropna())
def pid(n, gid):
    c = NAME.get(norm(n),[])
    if len(c)==1: return c[0]
    gm = np2[np2.game_id==gid]; ids = set(gm.passer_player_id.dropna())|set(gm.rusher_player_id.dropna())|set(gm.receiver_player_id.dropna())
    h=[x for x in c if x in ids]; return h[0] if h else (c[0] if c else None)
tds = np2[np2.touchdown.eq(1)&np2.td_player_id.notna()].groupby("td_player_id").size()
ST = {"player_pass_yds": np2.groupby("passer_player_id").passing_yards.sum(), "player_rush_yds": np2.groupby("rusher_player_id").rushing_yards.sum(),
      "player_reception_yds": np2.groupby("receiver_player_id").receiving_yards.sum(), "player_receptions": np2.groupby("receiver_player_id").complete_pass.sum()}
# ATD (Sunday pull)
a = pd.read_csv("props_log.csv"); a = a[(a.week==W)&(a.pulled_at=="2026-10-11T11:22Z")&a.model_p.notna()&a.game_id.isin(done)].copy()
a["pid"]=[pid(n,g) for n,g in zip(a.player,a.game_id)]; a=a[a.pid.isin(touched)]
a["hit"]=a.pid.map(tds).fillna(0).gt(0).astype(int); a["ev"]=a.model_p*a.price-1; a["pnl"]=np.where(a.hit==1,a.price-1,-1.)
a.to_csv("/tmp/claude-0/-home-claude/61a3e72b-dbd0-5c34-8049-91e3f7bfb7b5/scratchpad/atd_graded.csv", index=False)
# O/U
o = pd.read_csv("props_log_ou.csv"); o=o[(o.week==W)&(o.pulled_at=="2026-10-11T11:51Z")&o.model_p_over.notna()&o.game_id.isin(done)].copy()
o["pid"]=[pid(n,g) for n,g in zip(o.player,o.game_id)]; o=o[o.pid.isin(touched)]
o["act"]=[float(ST[m].get(x,0)) for m,x in zip(o.market,o.pid)]
o=o[o.act!=o.point]
o["over_hit"]=(o.act>o.point).astype(int)
o["side"]=np.where(o.model_p_over*o.over-1 > (1-o.model_p_over)*o.under-1,"Over","Under")
o["ev"]=np.maximum(o.model_p_over*o.over-1,(1-o.model_p_over)*o.under-1)
o["win"]=np.where(o.side=="Over",o.over_hit,1-o.over_hit)
o["price"]=np.where(o.side=="Over",o.over,o.under); o["pnl"]=np.where(o.win==1,o.price-1,-1.)
o.to_csv("/tmp/claude-0/-home-claude/61a3e72b-dbd0-5c34-8049-91e3f7bfb7b5/scratchpad/ou_graded.csv", index=False)
print("ATD graded:", len(a), "| O/U graded:", len(o))
