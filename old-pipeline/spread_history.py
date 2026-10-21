"""Historical INT rates by point spread (team perspective) + model calibration by spread bucket."""
import numpy as np, pandas as pd, int_model as M

BINS   = [-99, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 99]
LABELS = ["Dog 11+", "Dog 7-10.5", "Dog 4-6.5", "Dog 1-3.5", "Pick'em",
          "Fav 1-3.5", "Fav 4-6.5", "Fav 7-10.5", "Fav 11+"]

def bucket(margin):  # exp_margin: + = favored
    return pd.cut(margin, BINS, labels=LABELS)

if __name__ == "__main__":
    pbp, sched = M.load(); qb, tg, ctx = M.build_tables(pbp, sched)
    df = M.prep(M.build_features(qb, tg, ctx))
    df = df[df.att.notna() & (df.att >= 10) & df.exp_margin.notna()].copy()
    df["bucket"] = bucket(df.exp_margin)
    df["hit"] = (df.ints >= 1).astype(int)

    hist = df.groupby("bucket", observed=True).agg(
        starts=("hit", "size"), p_int=("hit", "mean"), ints=("ints", "sum"),
        att=("att", "sum"), att_pg=("att", "mean"))
    hist["int_pct"] = hist.ints / hist.att * 100
    hist["se"] = np.sqrt(hist.p_int * (1 - hist.p_int) / hist.starts)
    print("All starts 2016 - 2026 Wk3 (starter, 10+ att):")
    print(hist[["starts", "p_int", "se", "att_pg", "int_pct"]].round(3).to_string())

    # walk-forward out-of-sample predictions: train on seasons < s, predict s
    d = df[df.prior_games >= 1].dropna(subset=M.ATT_X + M.INT_X[1:])
    oos = []
    for s in range(2019, 2027):
        tr, te = d[d.season < s], d[d.season == s]
        am = M.fit_attempts(tr); im = M.fit_ints(tr, am)
        oos.append(M.predict(te, am, im))
    oos = pd.concat(oos)
    cal = oos.groupby("bucket", observed=True).agg(n=("hit", "size"), model=("p_int", "mean"), actual=("hit", "mean"))
    cal["miss"] = cal.actual - cal.model
    print("\nModel (walk-forward 2019-2026) vs actual by spread bucket:")
    print(cal.round(3).to_string())
    hist.to_csv("int_by_spread.csv")
