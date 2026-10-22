"""(1) Are the model's 55-95% 'conservative leg' probabilities calibrated?  (2) Does player consistency add to it?
Walk-forward: projections fit on prior seasons; outcome distributions (ratio tables) built only from earlier seasons' OOS rows."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "."); sys.path.insert(0, "tests")
import proj_model as P
P.RANGE_STATS = ["rec", "rec_yds", "rush_yds", "pass_yds"]
from t_star_cal import run
o = run(seasons=range(2020, 2027))
o = o.sort_values(["pid", "gameday"])
SPECS = [("rec", "WR", [2, 3, 4, 5, 6, 7]), ("rec", "TE", [2, 3, 4, 5, 6]), ("rec", "RB", [1, 2, 3, 4]),
         ("rec_yds", "WR", list(range(20, 121, 10))), ("rec_yds", "TE", list(range(10, 91, 10))), ("rush_yds", "RB", list(range(20, 121, 10))),
         ("pass_yds", "QB", list(range(150, 326, 25)))]
# consistency: player's prior coefficient of variation of the stat (last 16 games, min 6), no look-ahead
for st in ["rec", "rec_yds", "rush_yds", "pass_yds"]:
    g = o.groupby("pid")[st]
    m = g.transform(lambda x: x.shift().rolling(16, min_periods=6).mean())
    s = g.transform(lambda x: x.shift().rolling(16, min_periods=6).std())
    o[f"cv_{st}"] = s / m.replace(0, np.nan)
ev = []
for s in range(2022, 2027):
    tab = P.ratio_table(o[o.season < s].assign(qb_start=o.qb_start))
    te = o[o.season == s]
    for st, grp, ths in SPECS:
        r = te[(te.pos == grp) & (te[f"p_{st}"] > 0.5)]
        if grp == "QB": r = r[r.qb_start]
        edges = [tab[(st, grp, b)]["lo"] for b in range(3)]
        for b in range(3):
            lo = edges[b]; hi = tab[(st, grp, b)]["hi"] if b == 2 else edges[b + 1]
            rb = r[(r[f"p_{st}"] >= (lo if b else -1)) & (r[f"p_{st}"] < (hi if b < 2 else 1e9))]
            rat = np.sort(tab[(st, grp, b)]["ratios"])
            for k in ths:
                need = (k - 0.5) / rb[f"p_{st}"].to_numpy()            # P(actual >= k) = P(ratio > (k-.5)/mu)
                p = 1 - np.searchsorted(rat, need, side="right") / len(rat)
                ev.append(pd.DataFrame(dict(season=s, st=st, grp=grp, k=k, p=p, hit=(rb[st].to_numpy() >= k).astype(int),
                                            cv=rb[f"cv_{st}"].to_numpy(), pid=rb.pid.to_numpy())))
e = pd.concat(ev); e = e[(e.p >= 0.55) & (e.p <= 0.95)]
print(f"conservative-leg events (model 55-95%): {len(e):,}  seasons 2022-2026\n")
e["bin"] = pd.cut(e.p, [.55, .65, .75, .85, .95])
print("=== (1) CALIBRATION: model probability vs actual hit rate")
t = e.groupby(["st", "bin"], observed=True).agg(n=("hit", "size"), model=("p", "mean"), actual=("hit", "mean"))
t["gap"] = t.actual - t.model
print((t.assign(model=lambda x: (x.model*100).round(1), actual=lambda x: (x.actual*100).round(1), gap=lambda x: (x.gap*100).round(1))).to_string())
# (2) consistency beyond the model: fit on 2022-24, test 2025-26
print("\n=== (2) DOES CONSISTENCY ADD? logit(hit) ~ logit(model p) + cv (z within stat/pos)")
e = e.dropna(subset=["cv"]).copy()
e["cvz"] = e.groupby(["st", "grp"]).cv.transform(lambda x: (x - x.mean()) / x.std())
e["lp"] = np.log(e.p / (1 - e.p))
for st in ["rec", "rec_yds", "rush_yds", "pass_yds"]:
    x = e[e.st == st]; tr, te = x[x.season <= 2024], x[x.season >= 2025]
    if len(te) < 300: continue
    m0 = sm.Logit(tr.hit, sm.add_constant(tr[["lp"]])).fit(disp=0)
    m1 = sm.Logit(tr.hit, sm.add_constant(tr[["lp", "cvz"]])).fit(disp=0)
    def ll(m, cols): pp = m.predict(sm.add_constant(te[cols], has_constant="add")).clip(1e-6, 1-1e-6); return -np.mean(te.hit*np.log(pp) + (1-te.hit)*np.log(1-pp))
    raw = -np.mean(te.hit*np.log(te.p.clip(1e-6,1-1e-6)) + (1-te.hit)*np.log((1-te.p).clip(1e-6,1)))
    # effect size: hit rate for most vs least consistent quintile at similar model p
    te2 = te.copy(); te2["cq"] = pd.qcut(te2.cvz, 5, labels=["most steady", "2", "3", "4", "most volatile"])
    hr = te2.groupby("cq", observed=True).apply(lambda g: g.hit.mean() - g.p.mean())
    print(f"{st:<9} cv coef {m1.params['cvz']:+.3f} (z={m1.tvalues['cvz']:+.1f})  test log-loss: raw {raw:.4f} | recal {ll(m0,['lp']):.4f} | +consistency {ll(m1,['lp','cvz']):.4f}"
          f"\n          actual-minus-model by consistency quintile (2025-26): " + "  ".join(f"{k}: {v*100:+.1f}" for k, v in hr.items()))
