import numpy as np, pandas as pd, int_model as M
from spread_history import bucket, LABELS
pbp, sched = M.load(); qb, tg, ctx = M.build_tables(pbp, sched)
base = M.prep(M.build_features(qb, tg, ctx))
base = base[base.att.notna() & (base.att >= 10) & base.exp_margin.notna() & (base.prior_games >= 1)].copy()
m = base.exp_margin
base["big_dog"]  = np.clip(-m - 10, 0, None)        # extra slope beyond 10-pt dog
base["big_fav"]  = np.clip(m - 10, 0, None)
base["m2"]       = m**2 / 100
b = bucket(m)
for lab in LABELS:
    base["b_" + lab] = (b == lab).astype(int)
dums = ["b_" + l for l in LABELS if l not in ("Fav 1-3.5", "Pick'em")]
LIN = ["log_proj_att","log_qb_rate","qb_adot","qb_exp_low","off_press_rate","def_press_rate","log_def_rate","wind","exp_margin"]
NOLIN = [c for c in LIN if c != "exp_margin"]
variants = {
    "current (linear spread)": LIN,
    "no spread in INT rate": NOLIN,
    "linear + squared": LIN + ["m2"],
    "linear + hinges at +/-10": LIN + ["big_dog", "big_fav"],
    "spread buckets (historical)": NOLIN + dums,
}
d = base.dropna(subset=M.ATT_X + LIN[1:])
y_all = []
res = {}
for name, xs in variants.items():
    M.INT_X = xs
    preds = []
    for s in range(2019, 2027):
        tr, te = d[d.season < s], d[d.season == s]
        am = M.fit_attempts(tr); im = M.fit_ints(tr, am)
        preds.append(M.predict(te, am, im)[["p_int", "ints", "season"]])
    p = pd.concat(preds); y = (p.ints >= 1).astype(int)
    res[name] = dict(brier=np.mean((p.p_int - y)**2),
                     ll=-np.mean(y*np.log(p.p_int)+(1-y)*np.log(1-p.p_int)),
                     brier_24on=np.mean(((p.p_int - y)**2)[p.season >= 2024]))
const = np.mean((y.mean() - y)**2)
r = pd.DataFrame(res).T
r["skill_vs_const"] = 1 - r.brier / const
print(f"walk-forward 2019-2026, n={len(y)}")
print(r.round(5).to_string())
