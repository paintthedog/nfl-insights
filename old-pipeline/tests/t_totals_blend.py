import sys, numpy as np, pandas as pd; sys.path.insert(0, ".")
import lines_model as L
pbp, sched = L.load(); g = L.build(pbp, sched)
g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)].copy()
rows = []
for s in range(2019, 2027):
    mm, mt = L.fit(g[g.season < s]); rows.append(L.predict(g[g.season == s].copy(), mm, mt))
o = pd.concat(rows); print(f"games {len(o)} (2019-2026, walk-forward)")
for nm, a, m, ln in [("TOTALS", o.total, o.m_total, o.total_line), ("MARGIN", o.result, o.m_margin, o.spread_line)]:
    res = [(w, np.mean(np.abs(a - (w * m + (1 - w) * ln)))) for w in np.arange(0, 1.01, .1)]
    b = min(res, key=lambda t: t[1])
    print(f"\n{nm}: line-only MAE {res[0][1]:.2f} | model-only {res[-1][1]:.2f} | best blend w_model={b[0]:.1f} -> {b[1]:.2f}")
    gap = m - ln; side = np.sign(gap) == np.sign(a - ln); push = a == ln
    for t in [0, 2, 3, 4, 5]:
        sel = (np.abs(gap) >= t) & ~push
        n = sel.sum(); w = (side & sel).sum(); print(f"   model side when gap>={t}: {w}-{n-w} ({w/n:.1%})" + ("" if n == 0 else ""))
    for s_, x in o.assign(gap=gap, side=side, push=push).groupby("season"):
        x = x[(np.abs(x.gap) >= 3) & ~x.push]; 
        if len(x): print(f"      {s_} gap>=3: {x.side.mean():.1%} (n={len(x)})")
import json
def summ(a, m, ln):
    gap = m - ln; side = np.sign(gap) == np.sign(a - ln); sel = (np.abs(gap) >= 3) & (a != ln)
    return dict(n=int(sel.sum()), w=int((side & sel).sum()), mae_model=round(float(np.mean(np.abs(a - m))), 2), mae_line=round(float(np.mean(np.abs(a - ln))), 2))
json.dump(dict(games=len(o), seasons="2019-2026", totals=summ(o.total, o.m_total, o.total_line), spreads=summ(o.result, o.m_margin, o.spread_line)), open("data/lines_bt_summary.json", "w"))
