"""Does an off-ball-linebacker tackle feature help DROY (and DPOY)? LOSO at THRU and at season's end: winner rank hits + log loss."""
import numpy as np
src = open("awards.py").read().split("PROJ = {")[0]; exec(src)
P5 = project(THRU)
_frame = frame
def frame2(kind, tabs, wide):
    d = _frame(kind, tabs, wide)
    if kind in ("dpoy", "droy"): d["lb_tkl"] = np.where(d.role == "Linebacker", d.tkl, 0.0)
    return d
def run(kind, feats, tabs):
    FEATS[kind] = feats; d, miss = label(kind, zed(frame2(kind, tabs, True), feats)); yrs = [y for y in WIN[kind] if y not in miss]
    rk, ll = {}, 0
    for yr in yrs:
        te = score(kind, fit(kind, d[(d.season != yr) & (d.season < SEASON)], True), d[d.season == yr]); w = te[te.won == 1]
        if not len(w): continue
        rk[yr] = int(w.rk.iloc[0]); ll -= np.log(max(w.p.iloc[0], 1e-9))
    return rk, ll, d
for kind in ("droy", "dpoy"):
    base = list(FEATS[kind])
    for nm, f in (("base", base), ("+lb_tkl", base + ["lb_tkl"])):
        for tag, tabs in (("wk%d" % THRU, P5), ("final", FULL)):
            rk, ll, d = run(kind, f, tabs)
            print(f"{kind} {nm:8s} {tag:5s} top1 {hits(rk, rk, 1)} top3 {hits(rk, rk, 3)} top5 {hits(rk, rk, 5)} of {len(rk)}  logloss {ll:.1f}  ranks {rk}")
        if kind == "droy":
            dw = d; te = score(kind, fit(kind, dw[dw.season < SEASON], True), dw[dw.season == SEASON]).sort_values("p", ascending=False)
            print("   2026:", [(n, round(p * 100, 1)) for n, p in zip(te.name.head(8), te.p.head(8))])
    FEATS[kind] = base
