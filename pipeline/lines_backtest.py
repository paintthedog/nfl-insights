"""Walk-forward backtest of the spread/total model vs the closing line, 2019-2026."""
import numpy as np, pandas as pd
import lines_model as L

pbp, sched = L.load()
g = L.build(pbp, sched)
g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)].copy()

rows = []
for s in range(2019, 2027):
    tr, te = g[g.season < s], g[g.season == s].copy()
    mm, mt = L.fit(tr)
    rows.append(L.predict(te, mm, mt))
    if s == 2026:
        print("margin model:", {k: round(v, 2) for k, v in mm.params.items()})
        print("total model: ", {k: round(v, 2) for k, v in mt.params.items()})
o = pd.concat(rows)

# ---- raw accuracy vs the market
print(f"\nn = {len(o)} games, 2019-2026")
for nm, pred in [("closing spread", o.spread_line), ("model margin", o.m_margin), ("blend 50/50", 0.5 * o.spread_line + 0.5 * o.m_margin)]:
    print(f"  margin MAE  {nm:16s} {np.mean(np.abs(o.result - pred)):.2f}   bias {np.mean(o.result - pred):+.2f}")
for nm, pred in [("closing total ", o.total_line), ("model total   ", o.m_total), ("blend 50/50   ", 0.5 * o.total_line + 0.5 * o.m_total)]:
    print(f"  total  MAE  {nm:16s} {np.mean(np.abs(o.total - pred)):.2f}   bias {np.mean(o.total - pred):+.2f}")

# residual SDs for cover probabilities
print(f"\nresidual sd: margin vs model {np.std(o.result - o.m_margin):.2f}, vs line {np.std(o.result - o.spread_line):.2f}; "
      f"total vs model {np.std(o.total - o.m_total):.2f}, vs line {np.std(o.total - o.total_line):.2f}")

# ---- ATS / O-U record when the model disagrees with the line by at least X
o["home_cover"] = np.sign(o.result - o.spread_line)      # +1 home covers, -1 away, 0 push
o["over_hit"] = np.sign(o.total - o.total_line)
print("\nAgainst the spread (bet the side the model likes):")
print(f"  {'edge >=':8s} {'bets':>6s} {'win%':>7s} {'ROI@-110':>9s}")
for x in [0, 1, 2, 3, 4, 5]:
    pick = np.sign(o.m_margin - o.spread_line); m = (np.abs(o.m_margin - o.spread_line) >= x) & (pick != 0) & (o.home_cover != 0)
    w = (pick[m] == o.home_cover[m]).mean(); n = m.sum()
    print(f"  {x:>5}    {n:6d} {w*100:6.1f}%  {(w*1.909 - 1)*100:+7.1f}%")
print("\nTotals (bet over/under the model likes):")
print(f"  {'edge >=':8s} {'bets':>6s} {'win%':>7s} {'ROI@-110':>9s}")
for x in [0, 1, 2, 3, 4, 5]:
    pick = np.sign(o.m_total - o.total_line); m = (np.abs(o.m_total - o.total_line) >= x) & (pick != 0) & (o.over_hit != 0)
    w = (pick[m] == o.over_hit[m]).mean(); n = m.sum()
    print(f"  {x:>5}    {n:6d} {w*100:6.1f}%  {(w*1.909 - 1)*100:+7.1f}%")

# by season, 3+ pt edges
print("\nATS win% at 3+ pt edge, by season:")
pick = np.sign(o.m_margin - o.spread_line); m = (np.abs(o.m_margin - o.spread_line) >= 3) & (o.home_cover != 0)
print(o[m].assign(w=(pick[m] == o.home_cover[m])).groupby("season").w.agg(["size", "mean"]).round(3).T.to_string())

# ---- single-factor checks the market may misprice
print("\nSituational, 2019-2026 (home-cover rate; 50% = market is fair):")
for label, mask in [("QB change, home", o.h_qb_change == 1), ("QB change, away", o.a_qb_change == 1),
                    ("home off bye", o.h_bye == 1), ("away off bye", o.a_bye == 1), ("home short week", o.h_short == 1),
                    ("wind 15+", o.wind_eff >= 15), ("dome", o.dome == 1), ("division game", o.div_game == 1)]:
    r = o[mask & (o.home_cover != 0)]
    print(f"  {label:18s} n={len(r):4d}  home covers {(r.home_cover > 0).mean()*100:5.1f}%   overs {(o[mask & (o.over_hit != 0)].over_hit > 0).mean()*100:5.1f}%")
o.to_pickle("lines_backtest.pkl")
