"""How hard to pull each team's rushing-TD share toward the league average (td_model.RS_K). Walk-forward 2021-2026."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, td_model as T
pbp, sched, players = T.load(); pg, team, ctx = T.build_tables(pbp, sched, players)
BASE_R, BASE_P = list(T.RUSH_X), list(T.REC_X)
for K, X in [(30, 0), (30, 1)]:
    T.RS_K = K
    T.RUSH_X = BASE_R + (["l_xpg"] if X else []); T.REC_X = BASE_P + (["l_xpg"] if X else [])
    d = T.features(pg, team, ctx); d = d[(d.pr_n >= 1) & d.implied.notna() & d.rush_share.notna()]; d["hit"] = ((d.rtd + d.ptd) >= 1).astype(int)
    P = []
    for s in range(2021, 2027):
        tmm = T.fit_team(team[team.season < s]); mr, mp = T.fit_players(d[d.season < s], tmm); P.append(T.predict(d[d.season == s], tmm, mr, mp))
    p = pd.concat(P); y = p.hit; br = np.mean((p.p_td - y) ** 2); ll = -np.mean(y * np.log(p.p_td.clip(1e-4, 1 - 1e-4)) + (1 - y) * np.log((1 - p.p_td).clip(1e-4)))
    rb = p[(p.pos == "RB") & (p.pr_car_sh >= .45)]; lo = rb[rb.rush_share <= .3]
    top = p[p.groupby(["game_id", "posteam"]).p_td.rank(ascending=False, method="first") == 1]
    print(f"K={K:3d} xpg={X} brier {br:.5f} logloss {ll:.5f} | lead RBs pred {rb.p_td.mean():.3f} act {rb.hit.mean():.3f} | low-share teams n={len(lo)} pred {lo.p_td.mean():.3f} act {lo.hit.mean():.3f} | team's top option pred {top.p_td.mean():.3f} act {top.hit.mean():.3f}", flush=True)
