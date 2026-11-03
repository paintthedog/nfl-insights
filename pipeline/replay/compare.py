"""Compare graded replay variants on matched player-games. Usage: python replay/compare.py base k_ ca_ cb_   ("base" = untagged)"""
import sys, numpy as np, pandas as pd
tags = sys.argv[1:]
G = {t: pd.read_pickle(f"replay/out/{'' if t == 'base' else t}graded.pkl") for t in tags}
key = ["season", "week", "pid"]
common = set.intersection(*[set(map(tuple, g[key].values)) for g in G.values()])
for st, pos in [("car", ["RB"]), ("rush_yds", ["RB"]), ("fp_ppr", ["RB"]), ("rec_yds", ["WR", "TE", "RB"]), ("fp_ppr", ["WR", "TE"])]:
    print(f"== {st} ({'/'.join(pos)}), on the field, p_play 0.9+, matched")
    for t, g in G.items():
        x = g[g.on & (g.p_play >= 0.9) & g.pos.isin(pos)]
        x = x[np.array([k in common for k in map(tuple, x[key].values)]) & x[f"a_{st}"].notna() & x[f"m_{st}"].notna()]
        for lab, y in [("all", x), ("2025", x[x.season == 2025])]:
            e = y[f"a_{st}"] - y[f"m_{st}"]
            print(f"  {t:5s} {lab:4s} n={len(y):5d} MAE {e.abs().mean():7.3f} RMSE {np.sqrt((e ** 2).mean()):7.3f} bias {e.mean():+6.2f} below {y[f'b_{st}'].mean():.3f} above {y[f'u_{st}'].mean():.3f} mid {y[f'i_{st}'].mean():.3f}")
