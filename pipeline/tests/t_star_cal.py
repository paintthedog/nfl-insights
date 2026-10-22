"""Is the yardage model compressed by its OWN projection? Walk-forward 2021-2026, calibration slope + decile table."""
import os, pickle, sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, ".")
import proj_model as P
CACHE = "/tmp/claude-0/-home-claude/61a3e72b-dbd0-5c34-8049-91e3f7bfb7b5/scratchpad/proj_built.pkl"
if os.path.exists(CACHE): d0 = pickle.load(open(CACHE, "rb"))
else:
    pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
    players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
    d0 = P.build(pbp, sched, players); pickle.dump(d0, open(CACHE, "wb"))
TESTS = [("QB", "pass_yds"), ("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "rec_yds"), ("RB", "rush_yds"), ("WR", "rec"), ("TE", "rec")]
def run(kmult=1.0, keys=("ypt", "ypa", "ypc", "catch"), seasons=range(2021, 2027)):
    K0 = dict(P.K)
    for k in keys: P.K[k] = K0[k] * kmult
    d = d0.copy(); pm = P.position_means(d); d = P.efficiency(d, pm)
    P.K.update(K0)
    out = []
    for s in seasons:
        tr = d[d.season < s].copy(); te = d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); te = P.apply_calibration(te, cal); out.append(te)
    o = pd.concat(out); o = o[(o.pr_n >= 1)]
    return o[((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"])]
def report(o, label):
    print(f"\n--- {label}")
    for grp, st in TESTS:
        r = o[(o.pos == grp) & (o[f"p_{st}"] > 0)]
        slope = sm.OLS(r[st], sm.add_constant(r[f"p_{st}"])).fit().params.iloc[1]
        top = r[r[f"p_{st}"] >= r[f"p_{st}"].quantile(0.9)]
        print(f"{grp} {st:<9} n={len(r):>5}  MAE {np.mean(np.abs(r[st]-r[f'p_{st}'])):6.2f}  slope {slope:4.2f}  "
              f"top-10% proj {top[f'p_{st}'].mean():6.1f} -> actual {top[st].mean():6.1f} ({top[st].mean()/top[f'p_{st}'].mean()-1:+.0%})")
if __name__ == "__main__":
    o = run(); report(o, "CURRENT MODEL (K as is)")
    # decile table for the two that matter most
    for grp, st in [("QB", "pass_yds"), ("WR", "rec_yds")]:
        r = o[(o.pos == grp) & (o[f"p_{st}"] > 0)].copy(); r["dec"] = pd.qcut(r[f"p_{st}"], 10, labels=False) + 1
        t = r.groupby("dec").agg(proj=(f"p_{st}", "mean"), actual=(st, "mean"), n=(st, "size")).round(1)
        print(f"\n{grp} {st} by model-projection decile:\n" + t.T.to_string())
