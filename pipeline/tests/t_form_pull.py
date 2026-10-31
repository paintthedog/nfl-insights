"""Season-form pullback (Oct 31, Clinton: "why are these numbers so low when season averages are much higher?").
When the model projects a player well below his own average THIS season (4+ games), pull the projection part of the way
back: p' = p + w * (avg - p), only when p < 0.9 * avg. One w per stat, fit on 2019-2023 walk-forward rows, judged on
2024-2026 (never seen in the fit). Walk-forward rows = tests/_wf_proj.pkl (raw projection + calibration)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
o = pd.read_pickle("tests/_wf_proj.pkl")
o = o[o.pos.isin(["RB", "WR", "TE"])].sort_values(["pid", "season", "week"]).copy()
STATS = ["tgt", "rec", "rec_yds", "car", "rush_yds"]
for st in STATS:
    g = o.groupby(["pid", "season"])[st]
    o[f"avg_{st}"] = g.transform(lambda s: s.shift().expanding().mean())
o["n_prior"] = o.groupby(["pid", "season"]).cumcount()
def apply(df, st, w, cut=0.9, nmin=4):
    p, a = df[f"p_{st}"], df[f"avg_{st}"]
    m = (df.n_prior >= nmin) & a.notna() & (p < cut * a)
    return np.where(m, p + w * (a - p), p), m
tr, te = o[o.season <= 2023], o[o.season >= 2024]
FIT = {}
print("stat      w    | 2024-26 affected rows: n  bias before -> after  MAE before -> after | all rows MAE before -> after")
for st in STATS:
    ws = np.arange(0, 0.61, 0.025)
    mse = [np.mean((apply(tr, st, w)[0] - tr[st]) ** 2) for w in ws]
    w = float(ws[int(np.argmin(mse))]); FIT[st] = w
    p2, m = apply(te, st, w); p1 = te[f"p_{st}"].to_numpy(); y = te[st].to_numpy(); mm = m.to_numpy()
    print(f"{st:9} {w:.3f} | {mm.sum():5d}  {np.mean(y[mm]-p1[mm]):+6.2f} -> {np.mean(y[mm]-p2[mm]):+6.2f}   {np.mean(abs(y[mm]-p1[mm])):6.2f} -> {np.mean(abs(y[mm]-p2[mm])):6.2f} |"
          f" {np.mean(abs(y-p1)):6.3f} -> {np.mean(abs(y-p2)):6.3f}")
print("FIT", FIT)
# deep-cut table on the test seasons, after the pullback
for st in ["rec", "rec_yds", "rush_yds"]:
    x = te[(te.n_prior >= 4) & (te[f"avg_{st}"] >= {"rec": 4, "rec_yds": 40, "rush_yds": 40}[st])].copy()
    x["adj"] = apply(x, st, FIT[st])[0]; x["b"] = pd.cut(x[f"p_{st}"] / x[f"avg_{st}"], [0, .6, .75, .9, 1.1, 9])
    print(st); print(x.groupby("b", observed=True).agg(n=(st, "size"), avg=(f"avg_{st}", "mean"), before=(f"p_{st}", "mean"), after=("adj", "mean"), actual=(st, "mean")).round(1).to_string())
