"""Validate the copula: fit correlations on 2019-2023, test 2024-2026. Legs at ~70% and ~50% marginals; compare predicted joint
(copula) vs independence vs actual, for every same-game pair and a set of 3-leg combos."""
import sys; sys.path.insert(0, ".")
import itertools, numpy as np, pandas as pd
from scipy.stats import norm
import parlay_joint as J
o = pd.read_pickle("tests/_wf_proj.pkl"); sched = pd.read_parquet("data/sched.parquet")
T, cols = J.scores(o, sched); tr, te = T[T.season <= 2023], T[T.season >= 2024]
C = J.fit(tr, cols); allc = C["labels"]
rows = []
for p in (0.70, 0.50):
    th = norm.ppf(1 - p)
    for combo in list(itertools.combinations(allc, 2)) + [("QB pass_yds", "WR1 rec_yds", "WR2 rec_yds"), ("QB pass_yds", "WR1 rec_yds", "TE1 rec_yds"),
                                                         ("RB1 rush_yds", "WIN", "QB pass_yds"), ("WIN", "OPP QB pass_yds", "OPP WR1 rec_yds"), ("RB1 rush_yds", "RB1 rec", "WIN")]:
        if all(c.startswith("OPP") for c in combo): continue
        hits = (te[list(combo)] > th).all(axis=1); act = hits.mean()
        rows.append(dict(p=p, legs=" + ".join(combo), k=len(combo), actual=act, copula=J.joint([(c, p) for c in combo], C), indep=p ** len(combo), n=len(te)))
R = pd.DataFrame(rows)
for k in (2, 3):
    x = R[R.k == k]
    print(f"{k}-leg combos ({len(x)}), held-out 2024-26: mean |actual - copula| {np.mean(np.abs(x.actual-x.copula)):.4f}  vs  |actual - independent| {np.mean(np.abs(x.actual-x.indep)):.4f}")
x = R[R.k == 2].assign(err_c=lambda d: d.actual - d.copula, err_i=lambda d: d.actual - d.indep)
print("\nlargest correlations' held-out check (70% legs):")
print(x[x.p == 0.70].reindex(x[x.p == 0.70].err_i.abs().sort_values(ascending=False).index).head(10)[["legs", "actual", "copula", "indep"]].round(3).to_string(index=False))
print("\n3-leg combos:"); print(R[R.k == 3][["p", "legs", "actual", "copula", "indep"]].round(3).to_string(index=False))
