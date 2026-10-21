"""QB change split (Oct 17): one 'QB changed' flag vs separate 'new/backup QB' and 'starter returning' flags. Walk-forward 2019-2026."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, lines_model as L
pbp, sched = L.load(); g = L.build(pbp, sched); g = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & (g.min_games >= 3)].copy()
BM, BT = list(L.MARGIN_X), list(L.TOTAL_X)
SM = [x if x != "qb_gap" else None for x in BM]; SM = [x for x in SM if x] + ["qb_new_gap", "qb_ret_gap"]
ST = [x for x in BT if x != "qb_any"] + ["qb_new_any", "qb_ret_any"]
res = {}
for lab, (mx, tx) in {"current": (BM, BT), "split new/returning": (SM, ST)}.items():
    L.MARGIN_X, L.TOTAL_X = mx, tx; outs = []
    for yr in range(2019, 2027):
        mm, mt = L.fit(g[g.season < yr]); outs.append(L.predict(g[g.season == yr].copy(), mm, mt))
    res[lab] = pd.concat(outs).set_index("game_id").sort_index()
    mm, mt = L.fit(g); print(lab, "coefs:", {k: round(v, 2) for k, v in mm.params.items() if "qb" in k and "qd" not in k}, "| total:", {k: round(v, 2) for k, v in mt.params.items() if "qb" in k and "qd" not in k})
a, b = res["current"], res["split new/returning"].loc[res["current"].index]
for nm, act, col in (("margin", "result", "m_margin"), ("total", "total", "m_total")):
    d = np.abs(a[act] - a[col]) - np.abs(b[act] - b[col])
    print(f"{nm}: MAE current {np.abs(a[act]-a[col]).mean():.3f} -> split {np.abs(b[act]-b[col]).mean():.3f} | gain {d.mean():+.3f} ±{d.std()/np.sqrt(len(d)):.3f} | seasons better {int((d.groupby(a.season).mean()>0).sum())}/8")
ret = (b.h_qb_ret == 1) | (b.a_qb_ret == 1)
for lab, o in (("current", a), ("split", b)):
    sg = np.where(o.loc[ret].h_qb_ret == 1, 1, -1); print(f"games with a returning starter (n={ret.sum()}): returning team beat {lab} model by {np.mean(sg*(o.loc[ret].result-o.loc[ret].m_margin)):+.2f}")
