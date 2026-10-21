"""Calibrate the site's game-line probabilities (Oct 16). The old way treated the model as standalone: P(over) =
Normal((model total - line) / 13.1). The backtest says the model's gap to the line is worth far less than that.
Fit: P(home covers) and P(over) = logistic(a + b * gap), gap = model - line, trained on the model's own walk-forward
predictions from earlier seasons; pushes dropped. Scores 2021-2026 vs the old method and vs a coin flip.
Writes data/line_calib.json (fit on all seasons) for lines_project.py. Re-run in the offseason."""
import json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
import lines_model as L
pbp, sched = L.load(); g = L.build(pbp, sched)
g = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & (g.min_games >= 3)].copy()
outs = []
for yr in range(2019, 2027):
    mm, mt = L.fit(g[g.season < yr]); outs.append(L.predict(g[g.season == yr].copy(), mm, mt))
o = pd.concat(outs)
o["gap_m"] = o.m_margin - o.spread_line; o["gap_t"] = o.m_total - o.total_line
o["cover"] = np.where(o.result > o.spread_line, 1, np.where(o.result < o.spread_line, 0, np.nan))
o["over"] = np.where(o.total > o.total_line, 1, np.where(o.total < o.total_line, 0, np.nan))
o["old_c"] = 1 - norm.cdf((o.spread_line - o.m_margin) / L.MARGIN_SD); o["old_o"] = 1 - norm.cdf((o.total_line - o.m_total) / L.TOTAL_SD)
def fit(df, y, x):
    d = df.dropna(subset=[y]); m = sm.GLM(d[y], sm.add_constant(d[[x]]), family=sm.families.Binomial()).fit(); return m.params.to_dict()
pred = lambda p, x: 1 / (1 + np.exp(-(p["const"] + p[x.name] * x)))
rows, rel = [], []
for mk, y, x, old in (("spread", "cover", "gap_m", "old_c"), ("total", "over", "gap_t", "old_o")):
    te_all = []
    for yr in range(2021, 2027):
        p = fit(o[o.season < yr], y, x); te = o[(o.season == yr)].dropna(subset=[y]).copy(); te["new"] = pred(p, te[x]); te_all.append(te)
    t = pd.concat(te_all)
    for nm, col in (("coin flip", None), ("old (model as standalone)", old), ("calibrated", "new")):
        pp = np.full(len(t), 0.5) if col is None else t[col].clip(0.01, 0.99)
        rows.append(dict(market=mk, method=nm, n=len(t), brier=np.mean((t[y] - pp) ** 2), logloss=-np.mean(t[y] * np.log(pp) + (1 - t[y]) * np.log(1 - pp)),
                         max_shown=np.max(np.maximum(pp, 1 - pp))))
    t["side_p"] = np.maximum(t.new, 1 - t.new); t["side_hit"] = np.where(t.new >= 0.5, t[y], 1 - t[y])
    t["old_side_p"] = np.maximum(t[old], 1 - t[old]); t["old_hit"] = np.where(t[old] >= 0.5, t[y], 1 - t[y])
    for lo, hi in ((0.5, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 1)):
        s = t[(t.side_p >= lo) & (t.side_p < hi)]; rel.append(dict(market=mk, method="calibrated", band=f"{lo:.0%}-{hi:.0%}", n=len(s), said=s.side_p.mean(), hit=s.side_hit.mean()))
    for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 1)):
        s = t[(t.old_side_p >= lo) & (t.old_side_p < hi)]; rel.append(dict(market=mk, method="old", band=f"{lo:.0%}-{hi:.0%}", n=len(s), said=s.old_side_p.mean(), hit=s.old_hit.mean()))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(4).to_string(index=False)); print(); print(pd.DataFrame(rel).round(3).to_string(index=False))
final = {"spread": fit(o, "cover", "gap_m"), "total": fit(o, "over", "gap_t"), "fit_on": "2019-2026 walk-forward predictions", "fitted": str(pd.Timestamp.now())[:16]}
final = {k: ({"a": v["const"], "b": v[list(v)[1]]} if isinstance(v, dict) else v) for k, v in final.items()}
json.dump(final, open("data/line_calib.json", "w"), indent=1); print("\n", final)
