"""Nov 14 (Clinton: replace the interception model with a base rate?). Walk-forward by season: fit on every earlier season,
score the next. Is the QB-interception chance better than simple base rates? Usage: python tests/t_int_base.py (from pipeline/)"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from int_model import *
from spread_history import bucket
pbp, sched = load(); qb, tg, ctx = build_tables(pbp, sched)
df = prep(build_features(qb, tg, ctx)); df = df[(df.prior_games >= 1) & df.att.notna() & (df.att >= 10)]
R = []
for S in range(2019, 2027):
    tr = df[df.season < S]; te = df[df.season == S].dropna(subset=ATT_X + INT_X[1:])
    if not len(te): continue
    am = fit_attempts(tr); im = fit_ints(tr, am); t = predict(te, am, im)
    y = (t.ints >= 1).astype(int); hb = tr.assign(b=bucket(tr.exp_margin), h=(tr.ints >= 1)).groupby("b", observed=True).h.mean()
    lg = tr.ints.sum() / tr.att.sum()
    P = {"constant": np.full(len(t), (tr.ints >= 1).mean()), "by spread": bucket(t.exp_margin).map(hb).astype(float).fillna((tr.ints >= 1).mean()).values,
         "att x league rate": 1 - np.exp(-t.proj_att * lg), "att x QB rate": 1 - np.exp(-t.proj_att * t.qb_int_rate), "model": t.p_int.values}
    for k, p in P.items(): R.append(dict(season=S, m=k, n=len(t), se=((np.asarray(p) - y) ** 2).sum(), said=np.mean(p), hap=y.mean(), sd=np.std(p)))
    t = t.assign(y=y, c=P["constant"]); t["d"] = (t.c - t.y) ** 2 - (t.p_int - t.y) ** 2
    R.append(dict(season=S, m="_gain", n=len(t), se=t.d.sum(), said=t.d.std(), hap=0, sd=0))
R = pd.DataFrame(R); g = R[R.m != "_gain"]
print(g.assign(brier=g.se / g.n).pivot(index="season", columns="m", values="brier").round(4).to_string())
for lab, x in [("2019-26", g), ("2024-26", g[g.season >= 2024])]:
    a = x.groupby("m").agg(n=("n", "sum"), se=("se", "sum")); print(lab, (a.se / a.n).round(4).to_dict(), int(a.n.iloc[0]))
gg = R[R.m == "_gain"]; n = gg.n.sum(); mean = gg.se.sum() / n; sd = np.sqrt((gg.said ** 2 * gg.n).sum() / n)
print(f"model vs constant: Brier gain {mean:.4f} per game, z = {mean / (sd / np.sqrt(n)):.2f} over {n} games")
print(g.groupby("m").apply(lambda x: pd.Series(dict(said=np.average(x.said, weights=x.n), hap=np.average(x.hap, weights=x.n), spread=np.average(x.sd, weights=x.n)))).round(3).to_string())
