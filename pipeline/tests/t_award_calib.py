"""Are award 'fit' shares calibrated? Leave-one-season-out at THRU: compare the share the model gave the top pick (and the winner)
with how often it actually won; fit a temperature T (p ~ exp(s/T)) that minimises log loss of the winner, pooled across awards."""
import numpy as np
src = open("awards.py").read().split("PROJ = {")[0]; exec(src)
P5 = project(THRU); res = {}
for kind in VOTES:
    d, miss = label(kind, zed(frame(kind, P5, True), FEATS[kind])); yrs = [y for y in WIN[kind] if y not in miss]
    rows = []
    for yr in yrs:
        m = fit(kind, d[(d.season != yr) & (d.season < SEASON)], True); te = d[d.season == yr]
        s = m.decision_function(te[["z_" + x for x in FEATS[kind]]]); w = te.won.values
        if w.sum() == 0: continue
        rows.append((s, w))
    res[kind] = rows
def stats(rows, T):
    ll, top, hit, wp = 0, 0, 0, 0
    for s, w in rows:
        e = np.exp((s - s.max()) / T); p = e / e.sum(); i = p.argmax(); ll -= np.log(max(p[w == 1][0], 1e-9)); top += p[i]; hit += w[i]; wp += p[w == 1][0]
    return ll, top, hit, wp
for kind, rows in res.items():
    ll1, top, hit, wp = stats(rows, 1.0)
    Ts = np.linspace(0.5, 6, 56); best = min(Ts, key=lambda T: stats(rows, T)[0]); llb, topb, _, wpb = stats(rows, best)
    print(f"{kind:5s} n={len(rows)}  T=1: top pick share avg {top/len(rows):.0%}, top pick won {hit}/{len(rows)}, winner share avg {wp/len(rows):.0%}, logloss {ll1:.1f} | best T={best:.1f}: top share {topb/len(rows):.0%}, logloss {llb:.1f}")
allr = [r for k in res for r in res[k]]; Ts = np.linspace(0.5, 6, 56); bt = min(Ts, key=lambda T: stats(allr, T)[0])
print("pooled best T", round(bt, 2), "logloss", round(stats(allr, bt)[0], 1), "vs T=1", round(stats(allr, 1)[0], 1))
