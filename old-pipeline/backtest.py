"""Backtest: train on 2016-2023, test on 2024 through 2026 Week 3."""
import numpy as np, pandas as pd, statsmodels.api as sm
from int_model import *

pbp, sched = load()
qb, tg, ctx = build_tables(pbp, sched)
df = prep(build_features(qb, tg, ctx))
df = df[(df.prior_games >= 1) & df.att.notna() & (df.att >= 10)]  # need some history

train = df[df.season <= 2023]
test = df[df.season >= 2024].dropna(subset=ATT_X + INT_X[1:])

am = fit_attempts(train)
im = fit_ints(train, am)
print(am.summary().tables[1])
print(im.summary().tables[1])

t = predict(test, am, im)
y = (t.ints >= 1).astype(int)

# baselines
lg = train.ints.sum() / train.att.sum()
b0 = 1 - np.exp(-t.proj_att * lg)                 # volume x league rate
b1 = 1 - np.exp(-t.proj_att * t.qb_int_rate)      # volume x shrunk QB rate

def brier(p): return np.mean((p - y) ** 2)
def ll(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

base = y.mean()
print(f"\nTest games: {len(t)}   actual P(>=1 INT): {base:.3f}")
print(f"{'model':38s} brier   logloss")
for nm, p in [("constant (hit rate)", np.full(len(y), base)),
              ("B0: proj att x league rate", b0),
              ("B1: proj att x shrunk QB rate", b1),
              ("Full model", t.p_int)]:
    print(f"{nm:38s} {brier(p):.4f}  {ll(p):.4f}")

bs_c = brier(np.full(len(y), base))
print(f"\nBrier skill vs constant: {1 - brier(t.p_int)/bs_c:.2%}  (B0: {1-brier(b0)/bs_c:.2%})")
print(f"Spread of predictions: 10th pct {t.p_int.quantile(.1):.2f}, 90th pct {t.p_int.quantile(.9):.2f}")

t["bin"] = pd.qcut(t.p_int, 5)
cal = t.assign(y=y).groupby("bin", observed=True).agg(n=("y", "size"), pred=("p_int", "mean"), actual=("y", "mean"))
print("\nCalibration (quintiles):\n", cal.round(3))

# attempts model error
print(f"\nAttempts MAE: {np.mean(np.abs(t.proj_att - t.att)):.1f}")
