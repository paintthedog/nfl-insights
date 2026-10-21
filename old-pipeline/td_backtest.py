"""Walk-forward backtest of the anytime TD model, 2021-2026 Wk3."""
import numpy as np, pandas as pd, td_model as T
pbp, sched, players = T.load()
pg, team, ctx = T.build_tables(pbp, sched, players)
d = T.features(pg, team, ctx)
d = d[(d.pr_n >= 1) & d.implied.notna() & d.rush_share.notna()]
d["hit"] = ((d.rtd + d.ptd) >= 1).astype(int)
# naive baseline: player's own decayed TDs per game (Poisson), shrunk to league per-appearance rate
tdpg = d.assign(td=d.rtd + d.ptd)
tdpg = T.decayed_mean(tdpg, "pid", ["td"], T.PL_DECAY)
d = d.merge(tdpg[["game_id", "pid", "pr_td"]], on=["game_id", "pid"])

preds = []
for s in range(2021, 2027):
    tr, te = d[d.season < s], d[d.season == s]
    tmm = T.fit_team(team[team.season < s])
    mr, mp = T.fit_players(tr, tmm)
    preds.append(T.predict(te, tmm, mr, mp))
    if s == 2026: print("qb_scale", mr.qb_scale); print(tmm.params.round(3).to_dict()); print(mr.params.round(3).to_dict()); print(mp.params.round(3).to_dict())
p = pd.concat(preds)
lg = d[d.season < 2021].assign(td=lambda x: x.rtd + x.ptd).td.mean()
k = 3
p["base"] = 1 - np.exp(-((p.pr_td.fillna(lg) * p.pr_n + k * lg) / (p.pr_n + k)))
y = p.hit
def br(q): return np.mean((q - y) ** 2)
def ll(q): q = np.clip(q, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
c = np.full(len(y), y.mean())
print(f"\nn={len(p)} player-games, actual anytime TD rate {y.mean():.3f}")
for nm, q in [("constant", c), ("player's own recent TD rate", p.base), ("model", p.p_td)]:
    print(f"{nm:30s} brier {br(q):.4f}  logloss {ll(q):.4f}  skill {1 - br(q)/br(c):.1%}")
p["band"] = pd.cut(p.p_td, [0, .05, .1, .2, .3, .4, .5, .7, 1])
print("\nCalibration:\n", p.groupby("band", observed=True).agg(n=("hit", "size"), pred=("p_td", "mean"), actual=("hit", "mean")).round(3))
print("\nBy position:\n", p.groupby("pos").agg(n=("hit", "size"), pred=("p_td", "mean"), actual=("hit", "mean")).round(3).query("n > 200"))
# team-level sanity: sum of player lambdas vs actual skill-player TDs per team-game
tg = p.groupby(["game_id", "posteam"]).agg(lam=("lam", "sum"), act=("hit", "sum"))
tg2 = p.assign(td=p.rtd + p.ptd).groupby(["game_id", "posteam"]).agg(lam=("lam", "sum"), td=("td", "sum"))
print(f"\nper team-game: sum of player lambdas {tg2.lam.mean():.2f} vs actual TDs {tg2.td.mean():.2f}")
