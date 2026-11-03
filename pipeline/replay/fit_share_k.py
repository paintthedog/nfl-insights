"""Quick offline fit of the share-concentration exponent K on replayed weeks (means only; the full replay with SHARE_K_*
set is the real check). Every listed non-QB counts (no stat line = 0); expected value = mean x p_play.
Fit on 2024, judged on 2025."""
import numpy as np, pandas as pd
o = pd.read_pickle("replay/out/graded.pkl"); o = o[o.pos != "QB"].copy()
for c in ["m_tgt", "m_rec", "m_rec_yds", "m_car", "m_rush_yds", "m_fp_ppr"]: o[c] = o[c].fillna(0)
for c in ["a_tgt", "a_rec", "a_rec_yds", "a_car", "a_rush_yds", "a_fp_ppr"]: o[c] = o[c].fillna(0)
def apply(d, k, base, cols):
    b = d[base].clip(lower=0.05); mult = (b / b.groupby([d.game_id, d.team]).transform("median")) ** (k - 1)
    out = {}
    for c in cols:
        e0 = d[c] * d.p_play; e1 = d[c] * mult * d.p_play
        sc = e0.groupby([d.game_id, d.team]).transform("sum") / e1.groupby([d.game_id, d.team]).transform("sum").replace(0, np.nan)
        out[c] = (e1 * sc.fillna(1)).values
    return out
for grp, base, cols in [("receiving", "m_tgt", ["m_tgt", "m_rec", "m_rec_yds"]), ("rushing", "m_car", ["m_car", "m_rush_yds"])]:
    print(f"== {grp}: RMSE of expected vs actual (every listed player), 2024 fit / 2025 test")
    for k in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]:
        line = []
        for s in (2024, 2025):
            d = o[o.season == s]; e = apply(d, k, base, cols)
            line.append(" ".join(f"{c[2:]} {np.sqrt(np.mean((d['a_' + c[2:]].values - e[c]) ** 2)):.3f}" for c in cols))
        print(f"  K={k:.1f} | {line[0]} | {line[1]}")
