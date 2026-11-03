"""Anytime-TD recalibration test (Nov 3): logit(p') = a + b * logit(p_td), fit on 2024 replay weeks, judged on 2025."""
import numpy as np, pandas as pd
exec(open("replay/td_check.py").read().split("o = pd.DataFrame(rows)")[0].replace('print(', 'None and print('))
o = pd.DataFrame(rows); o = o[o.pos.isin(["RB", "WR", "TE"]) & o.p_td.notna()]; o["y"] = (o.td >= 1).astype(float)
lg = lambda p: np.log(p / (1 - p)); il = lambda x: 1 / (1 + np.exp(-x))
from scipy.optimize import minimize
tr, te = o[o.season == 2024], o[o.season == 2025]
nll = lambda w, d: -np.mean(d.y * np.log(il(w[0] + w[1] * lg(d.p_td.clip(.005, .995)))) + (1 - d.y) * np.log(1 - il(w[0] + w[1] * lg(d.p_td.clip(.005, .995)))))
w = minimize(nll, [0, 1], args=(tr,)).x
print(f"fit 2024: a={w[0]:+.3f} b={w[1]:.3f}")
for nm, d in (("2024", tr), ("2025", te)):
    p1 = il(w[0] + w[1] * lg(d.p_td.clip(.005, .995)))
    print(f"{nm}: Brier {((d.p_td - d.y) ** 2).mean():.5f} -> {((p1 - d.y) ** 2).mean():.5f} | sim {((d.sim_p - d.y) ** 2).mean():.5f}")
    b = pd.cut(d.p_td, [0, .1, .2, .3, .4, .5, 1])
    print(pd.DataFrame({"said": d.p_td, "new": p1, "y": d.y}).groupby(b, observed=True).agg(n=("y", "size"), said=("said", "mean"), new=("new", "mean"), scored=("y", "mean")).round(3).to_string())
print("\nblend of the site's (TD model) chance and the simulator's, in log-odds, weight on the simulator:")
for wt in [0, .25, .5, .75, 1]:
    out = []
    for nm, d in (("2024", tr), ("2025", te)):
        p = il((1 - wt) * lg(d.p_td.clip(.005, .995)) + wt * lg(d.sim_p.clip(.005, .995))); out.append(f"{nm} {((p - d.y) ** 2).mean():.5f}")
    print(f"  w={wt:.2f}: " + " | ".join(out))
x = o.copy(); b = pd.cut(x.p_td, [0, .1, .2, .3, .4, .5, 1])
print(x.groupby(b, observed=True).agg(n=("y", "size"), site=("p_td", "mean"), sim=("sim_p", "mean"), scored=("y", "mean")).round(3).to_string())
