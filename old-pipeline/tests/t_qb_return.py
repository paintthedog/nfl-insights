"""Does the model's QB-change penalty misfire when the change is a STARTER RETURNING (he started earlier that season)? 2019-2026 walk-forward."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, lines_model as L
pbp, sched = L.load(); g = L.build(pbp, sched); g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)].copy()
outs = []
for yr in range(2019, 2027):
    mm, mt = L.fit(g[g.season < yr]); outs.append(L.predict(g[g.season == yr].copy(), mm, mt))
o = pd.concat(outs)
# per team-game starter history from play-by-play
tg = L.team_games(pbp)[["game_id", "posteam", "qb"]]
s = sched[["game_id", "season", "gameday"]]; tg = tg.merge(s, on="game_id").sort_values("gameday")
tg["prior_starters"] = tg.groupby(["posteam", "season"]).qb.transform(lambda x: [set(x.iloc[:i]) for i in range(len(x))])
tg["prev_qb"] = tg.groupby(["posteam", "season"]).qb.shift()
tg["returning"] = [ (q != pq) and isinstance(pq, str) and (q in ps) for q, pq, ps in zip(tg.qb, tg.prev_qb, tg.prior_starters)]
tg["newqb"] = [ (q != pq) and isinstance(pq, str) and (q not in ps) for q, pq, ps in zip(tg.qb, tg.prev_qb, tg.prior_starters)]
H = tg.rename(columns={"posteam": "home_team", "returning": "h_ret", "newqb": "h_new"})[["game_id", "home_team", "h_ret", "h_new"]]
A = tg.rename(columns={"posteam": "away_team", "returning": "a_ret", "newqb": "a_new"})[["game_id", "away_team", "a_ret", "a_new"]]
o = o.merge(H, on=["game_id", "home_team"], how="left").merge(A, on=["game_id", "away_team"], how="left")
rows = []
for side, sg in (("h", 1), ("a", -1)):
    for kind in ("ret", "new"):
        m = o[f"{side}_{kind}"].fillna(False).astype(bool)
        x = o[m]
        model_err = sg * (x.result - x.m_margin)       # + = that team did better than the model said
        line_err = sg * (x.result - x.spread_line)
        rows.append(dict(change=("starter RETURNING" if kind == "ret" else "new/backup QB"), side=side, n=len(x), vs_model=model_err.mean(), vs_line=line_err.mean()))
R = pd.DataFrame(rows); W = R.assign(wm=R.vs_model * R.n, wl=R.vs_line * R.n).groupby("change")[["n", "wm", "wl"]].sum()
W["team beat MODEL by (pts)"] = W.wm / W.n; W["team beat LINE by (pts)"] = W.wl / W.n
print(W[["n", "team beat MODEL by (pts)", "team beat LINE by (pts)"]].round(2))
print("model's QB-change coefficient (all seasons):", round(L.fit(g)[0].params["qb_gap"], 2))
