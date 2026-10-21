"""Team-total scaling (Oct 17): production forces receivers' catches/targets/yards to sum to the QB's projection x 0.97.
Walk-forward 2019-2026: is that better than no scaling, or than meeting halfway (geometric mean of the two team estimates)?"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
o = pd.read_pickle("tests/_wf_proj.pkl")
qb = o[(o.pos == "QB") & o.qb_start.astype(bool)].drop_duplicates(["game_id", "posteam"]).set_index(["game_id", "posteam"])
rc = o[o.pos.isin(["WR", "TE", "RB"])].copy()
for st, qc in (("rec", "cmp"), ("tgt", "att"), ("rec_yds", "pass_yds")):
    tot = rc.groupby(["game_id", "posteam"])[f"p_{st}"].sum()
    q = qb[f"p_{qc}"] * (0.985 if st == "tgt" else 1.0) * 0.97
    k = pd.concat([tot.rename("rcv"), q.rename("qb")], axis=1).dropna()
    k["s_qb"] = k.qb / k.rcv; k["s_half"] = np.sqrt(k.qb / k.rcv)
    x = rc.join(k[["s_qb", "s_half"]], on=["game_id", "posteam"]).dropna(subset=["s_qb"])
    x = x[x.pos.isin(["WR", "TE"]) & (x[f"p_{st}"] > 0.5)]
    for lab, mult in (("no scaling", 1.0), ("scale to QB (production)", x.s_qb), ("meet halfway", x.s_half)):
        p = x[f"p_{st}"] * mult
        top = x[f"p_{st}"] >= x[f"p_{st}"].quantile(0.8)
        print(f"{st:8s} {lab:26s} MAE {np.mean(np.abs(x[st]-p)):.3f} | bias all {np.mean(x[st]-p):+.3f} | bias top-20% receivers {np.mean((x[st]-p)[top]):+.3f}")
    print(f"   median scale to QB: {k.s_qb.median():.3f}\n")

# ---- final recalibration of the production (scaled) projection: actual ~ a + b * scaled, fit on earlier seasons only
print("=== recalibrated after scaling (walk-forward, 2021-2026) ===")
import statsmodels.api as sm
for st, qc in (("rec", "cmp"), ("tgt", "att"), ("rec_yds", "pass_yds")):
    tot = rc.groupby(["game_id", "posteam"])[f"p_{st}"].sum(); q = qb[f"p_{qc}"] * (0.985 if st == "tgt" else 1.0) * 0.97
    k = pd.concat([tot.rename("rcv"), q.rename("qb")], axis=1).dropna(); k["s_qb"] = k.qb / k.rcv
    x = rc.join(k[["s_qb"]], on=["game_id", "posteam"]).dropna(subset=["s_qb"]); x = x[x.pos.isin(["WR", "TE"]) & (x[f"p_{st}"] > 0.5)].copy()
    x["scaled"] = x[f"p_{st}"] * x.s_qb
    outs = []
    for yr in range(2021, 2027):
        tr, te = x[x.season < yr], x[x.season == yr].copy()
        m = sm.OLS(tr[st], sm.add_constant(tr[["scaled"]])).fit(); te["recal"] = m.predict(sm.add_constant(te[["scaled"]], has_constant="add")); outs.append(te)
    t = pd.concat(outs); top = t.scaled >= t.scaled.quantile(0.8)
    for lab, col in (("production (scaled)", "scaled"), ("scaled + recalibrated", "recal")):
        print(f"{st:8s} {lab:24s} MAE {np.mean(np.abs(t[st]-t[col])):.3f} | bias all {np.mean(t[st]-t[col]):+.3f} | top-20% {np.mean((t[st]-t[col])[top]):+.3f}")
    m = sm.OLS(x[st], sm.add_constant(x[["scaled"]])).fit(); print(f"   full-history fit: actual = {m.params['const']:.3f} + {m.params['scaled']:.3f} x scaled\n")
