"""Test 1: when a contributor is absent, where does his volume go? Compares, on held-out seasons:
  none  = projections as-is (no redistribution)
  prop  = spread the vacated share across everyone in proportion to their own share (what the page does now)
  role  = fitted: multiplier depends on your position, the absent player's position, and whether you're next man up"""
import numpy as np, pandas as pd, statsmodels.api as sm, pickle
d = pd.read_parquet("data/inj_test.parquet")
d["sum_t"] = d.groupby(["game_id", "posteam"]).p_tgt.transform("sum"); d["sum_c"] = d.groupby(["game_id", "posteam"]).p_car.transform("sum")
d = d[(d.pr_n >= 1) & d.implied.notna() & d.pos.isin(["RB", "WR", "TE"])].copy()
for g in ["WR", "TE", "RB"]:
    d[f"t_same_{g}"] = 0.0
d["t_same"] = np.select([d.pos == "WR", d.pos == "TE", d.pos == "RB"], [d.vt_WR, d.vt_TE, d.vt_RB]); d["t_other"] = d.vt_all - d.t_same
d["t_same_top"] = d.t_same * (d.rk_t == 1); d["t_same_2"] = d.t_same * (d.rk_t == 2); d["t_same_rest"] = d.t_same * (d.rk_t > 2)
d["c_same"] = np.where(d.pos == "RB", d.vc_RB, 0.0); d["c_other"] = d.vc_all - d.c_same
d["c_same_top"] = d.c_same * (d.rk_c == 1); d["c_same_2"] = d.c_same * (d.rk_c == 2); d["c_same_rest"] = d.c_same * (d.rk_c > 2)
TX = ["t_same_top", "t_same_2", "t_same_rest", "t_other"]; CX = ["c_same_top", "c_same_2", "c_same_rest", "c_other"]

def fit(tr, y, base, X, groups):
    m = {}
    for g in groups:
        x = tr[(tr.pos == g) & (tr[base] > 0.05)]
        m[g] = sm.GLM(x[y], sm.add_constant(x[X]), family=sm.families.Poisson(), offset=np.log(x[base])).fit()
    return m
def mult(te, m, X):
    out = np.ones(len(te))
    for g, mm in m.items():
        k = (te.pos == g).to_numpy()
        out[k] = np.exp(mm.predict(sm.add_constant(te.loc[k, X], has_constant="add"), offset=np.zeros(k.sum())).pipe(np.log)) if k.any() else 1
    return out

import sys
KT, KC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
res = []
for s in range(2022, 2027):
    tr, te = d[d.season < s], d[d.season == s].copy()
    mt = fit(tr, "tgt", "p_tgt", TX, ["WR", "TE", "RB"]); mc = fit(tr, "car", "p_car", CX, ["RB"])
    te["mt_role"] = mult(te, mt, TX); te["mc_role"] = np.where(te.pos == "RB", mult(te, mc, CX), 1.0)
    te["mt_prop"] = (KT * te.v_att * .985 / te.sum_t).clip(.6, 1.8); te["mc_prop"] = (KC * te.v_car / te.sum_c).clip(.6, 1.8)     # normalise the active pool to the team total
    # additive version: next man up gets a slice of the vacated share itself, not a multiple of his own
    for nm, y, base, vol, X in [("t", "tgt", "p_tgt", "v_att", TX), ("c", "car", "p_car", "v_car", CX)]:
        add = np.zeros(len(te))
        for g in (["WR", "TE", "RB"] if nm == "t" else ["RB"]):
            x = tr[tr.pos == g]; k = (te.pos == g).to_numpy()
            m = sm.OLS((x[y] - x[base]) / x[vol], sm.add_constant(x[X])).fit()
            add[k] = m.predict(sm.add_constant(te.loc[k, X], has_constant="add")) * te.loc[k, vol]
        te[f"m{nm}_add"] = ((te[base] + add).clip(lower=0) / te[base].clip(lower=.05)).clip(0, 6)
    # role-aware first, then normalise the pool to the team total
    for nm, base, vol, k in [("t", "p_tgt", "v_att", .985), ("c", "p_car", "v_car", 1.0)]:
        adj = te[base] * te[f"m{nm}_add"]
        tot = te["sum_" + nm] + (adj - te[base]).groupby([te.game_id, te.posteam]).transform("sum")
        te[f"m{nm}_both"] = te[f"m{nm}_add"] * (te[vol] * k * (KT if nm == "t" else KC) / tot).clip(.6, 1.8)
    res.append(te)
o = pd.concat(res)
def report(mask, label):
    x = o[mask]
    print(f"\n{label}  (n={len(x)})")
    for st, base, mk in [("tgt", "p_tgt", "mt"), ("rec_yds", "p_rec_yds", "mt"), ("car", "p_car", "mc"), ("rush_yds", "p_rush_yds", "mc")]:
        y = x[x[base] > (1.5 if st in ("tgt", "car") else 8)]
        e = {k: np.abs(y[st] - y[base] * (1 if k == "none" else y[f"{mk}_{k}"])).mean() for k in ["none", "prop", "role", "add", "both"]}
        b = {k: (y[base] * (1 if k == "none" else y[f"{mk}_{k}"])).sum() / y[st].sum() - 1 for k in ["none", "prop", "role", "add", "both"]}
        print(f"  {st:9s} MAE none {e['none']:.3f}  norm {e['prop']:.3f}  role {e['role']:.3f}  add {e['add']:.3f}  add+norm {e['both']:.3f} | bias none {b['none']:+.1%} norm {b['prop']:+.1%} role {b['role']:+.1%} add {b['add']:+.1%} both {b['both']:+.1%}")
report(o.vt_all + o.vc_all >= 0, "ALL player-games 2022-26")
report(o.vt_all >= .15, "teams missing 15%+ of their targets")
report((o.pos == "RB") & (o.vc_RB >= .25), "RBs when 25%+ of the team's RB carries are out")
report((o.pos == "RB") & (o.vc_RB >= .25) & (o.rk_c == 1), "  ... the next man up")
report((o.pos == "WR") & (o.vt_WR >= .15) & (o.rk_t <= 2), "top-2 WRs when 15%+ of WR targets are out")
report((o.pos == "TE") & (o.vt_TE >= .10), "TEs when the usual TE is out")

full_t = fit(d, "tgt", "p_tgt", TX, ["WR", "TE", "RB"]); full_c = fit(d, "car", "p_car", CX, ["RB"])
print("\nfitted multipliers per 10 points of vacated share (exp(0.1*b)):")
for g, m in full_t.items(): print("  targets", g, {k: round(float(np.exp(.1 * v)), 3) for k, v in m.params.items() if k != "const"}, "const", round(float(np.exp(m.params["const"])), 3))
for g, m in full_c.items(): print("  carries", g, {k: round(float(np.exp(.1 * v)), 3) for k, v in m.params.items() if k != "const"}, "const", round(float(np.exp(m.params["const"])), 3))
pickle.dump(dict(t={g: m.params.to_dict() for g, m in full_t.items()}, c={g: m.params.to_dict() for g, m in full_c.items()}, TX=TX, CX=CX), open("inj_redistribution.pkl", "wb"))
