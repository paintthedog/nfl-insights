"""Lead-back carry concentration (Nov 2, Clinton). Week 7-8 live: backs projected 8-14 carries got 14.1 vs 10.9 (38 games).
Question: does the model split backfield carries too evenly? Test: within each team-game, sharpen RB carry projections
p_i' = T * p_i^k / sum_j p_j^k (team RB total T unchanged), k fit on 2019-2023 walk-forward rows, judged on 2024-2026.
Rows = tests/_wf_proj.pkl (walk-forward raw projections; no news overrides)."""
import numpy as np, pandas as pd
o = pd.read_pickle("tests/_wf_proj.pkl")
o = o[o.pos == "RB"].copy()
o["T"] = o.groupby("game_id posteam".split()).p_car.transform("sum")
def sharpen(df, k):
    w = df.p_car.clip(lower=0) ** k
    return df["T"] * w / w.groupby([df.game_id, df.posteam]).transform("sum")
def band(df, col):
    b = pd.cut(df.p_car, [-.01, 3, 8, 14, 100])
    return df.assign(e=df.car - df[col]).groupby(b, observed=True).agg(n=("e", "size"), proj=(col, "mean"), act=("car", "mean"), bias=("e", "mean"), mae=("e", lambda e: e.abs().mean())).round(2)
tr, te = o[o.season <= 2023].copy(), o[o.season >= 2024].copy()
print("baseline by band, 2019-2023:"); print(band(tr, "p_car"))
print("baseline by band, 2024-2026:"); print(band(te, "p_car"))
res = []
for k in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]:
    tr["q"] = sharpen(tr, k); res.append((k, (tr.car - tr.q).abs().mean(), (tr.car - tr.q).mean()))
print(pd.DataFrame(res, columns=["k", "mae_fit", "bias_fit"]).round(4).to_string(index=False))
kb = min(res, key=lambda r: r[1])[0]
te["q"] = sharpen(te, kb)
print(f"best k={kb}; 2024-2026 MAE {(te.car - te.p_car).abs().mean():.4f} -> {(te.car - te.q).abs().mean():.4f}")
print(band(te, "q"))
s26 = te[te.season == 2026]; print(f"2026 only: MAE {(s26.car - s26.p_car).abs().mean():.4f} -> {(s26.car - s26.q).abs().mean():.4f}")
# MAE rewards pulling backups toward their median (often 0), so also check squared error, which rewards the right mean
print("RMSE by k (2019-2023 fit / 2024-2026):")
for k in [1.0, 1.1, 1.2, 1.3]:
    a = np.sqrt(((tr.car - sharpen(tr, k)) ** 2).mean()); b = np.sqrt(((te.car - sharpen(te, k)) ** 2).mean()); print(f"  k={k}: {a:.4f} / {b:.4f}")
