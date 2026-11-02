"""Where does the model miss most? (Nov 2, Clinton). Every listed player in the last pre-kickoff archive of each graded game,
graded like the scorecard (on the field = offensive snap; a 0-touch day counts). Slices PPR misses by position, role,
news status and game, and lists the biggest single misses."""
import sys; sys.path.insert(0, ".")
import glob, json, numpy as np, pandas as pd
import proj_model as PM
pg = PM.player_games(PM.load())[0]
pg["ppr"] = pg.pass_yds * .04 + pg.pass_td * 4 - pg.ints * 2 + pg.rush_yds * .1 + pg.rush_td * 6 + pg.rec + pg.rec_yds * .1 + pg.rec_td * 6
act = pg.set_index(["game_id", "pid"])
sn = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_snaps", "offense_pct"])
snp = sn.set_index(["game_id", "gsis_id"]).offense_pct
sched = pd.read_parquet("data/sched.parquet"); sched = sched[sched.season == 2026]
QI = {q: i for i, q in enumerate(PM.QUANTS)}
rows = []
for wk in sorted({int(f.split("_wk")[1].split("_")[0]) for f in glob.glob("archive/data_wk*_*.json")}):
    snaps = sorted(glob.glob(f"archive/data_wk{wk}_*.json")); st = lambda f: f.split("_")[-2] + f.split("_")[-1][:4]
    for r in sched[(sched.week == wk) & sched.home_score.notna()].itertuples():
        pre = [f for f in snaps if st(f) < r.gameday.replace("-", "") + r.gametime.replace(":", "")]
        if not pre: continue
        D = json.load(open(pre[-1]))
        for p in D["proj"]["players"]:
            if p["team"] not in (r.home_team, r.away_team): continue
            m = p.get("mean") or {}; k = (r.game_id, p["pid"])
            sp = snp.get(k); on = sp is not None and sp > 0
            if k in act.index: a = act.loc[k]; a = a.iloc[0] if isinstance(a, pd.DataFrame) else a; v = float(a.ppr); on = True
            elif on: v = 0.0
            else: v = None
            q = (p.get("q") or {}).get("fp_ppr") or (p.get("sim") or {}).get("fp_ppr")
            rows.append(dict(wk=wk, game=f"{r.away_team}@{r.home_team}", team=p["team"], name=p["name"], pos=p["pos"], proj=float(m.get("fp_ppr") or 0),
                             p_play=p.get("p_play") or 1.0, flag=p.get("flag") or "", steady=(p.get("steady") or {}).get("score") if isinstance(p.get("steady"), dict) else p.get("steady"),
                             snap_hist=p.get("snap"), snap=sp, on=on, act=v, fav=(r.spread_line > 0) == (p["team"] == r.home_team),
                             margin=(r.result if p["team"] == r.home_team else -r.result), spread=(r.spread_line if p["team"] == r.home_team else -r.spread_line)))
o = pd.DataFrame(rows); o = o[o.proj >= 3]
g = o[o.on].copy(); g["err"] = g.act - g.proj; g["ae"] = g.err.abs()
pd.set_option("display.width", 200)
print(f"{len(o)} listed players projected 3+ PPR; {len(g)} on the field; {int((~o.on).sum())} never took a snap ({o[~o.on].proj.sum():.0f} projected points lost)")
def sl(by, df=g):
    return df.groupby(by, observed=True).agg(n=("err", "size"), proj=("proj", "mean"), act=("act", "mean"), bias=("err", "mean"), miss=("ae", "mean")).round(2)
print("\nby position:"); print(sl("pos"))
g["role"] = pd.cut(g.proj, [3, 6, 10, 15, 40], labels=["3-6 pts", "6-10", "10-15", "15+"])
print("\nby projected points:"); print(sl("role"))
g["snapchg"] = pd.cut((g.snap - g.snap_hist), [-2, -0.15, 0.15, 2], labels=["snaps fell 15+ pts", "about the same", "snaps rose 15+ pts"])
print("\nby snap share vs his recent share (only known after the game):"); print(sl("snapchg"))
g["flagged"] = np.where(g.flag.str.len() > 0, "flag / news note", "no flag")
print("\nby news flag:"); print(sl("flagged"))
g["qstat"] = np.where(g.p_play < 0.9, "questionable (p_play < 0.9)", "expected to play")
print(sl("qstat"))
g["script"] = pd.cut(g.margin - g.spread, [-60, -10, 10, 60], labels=["team did 10+ worse than spread", "within 10", "team did 10+ better"])
print("\nby game script vs the spread:"); print(sl(["script", "pos"]).unstack("pos")["bias"])
print("\nbiggest misses (on the field):")
print(g.reindex(g.ae.sort_values(ascending=False).index).head(15)[["wk", "game", "name", "pos", "proj", "act", "snap_hist", "snap", "flag"]].round(2).to_string(index=False))
print("\nnever took a snap, projected 5+:")
print(o[~o.on & (o.proj >= 5)][["wk", "game", "name", "pos", "proj", "p_play", "flag"]].to_string(index=False))
