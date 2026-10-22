"""Walk-forward TD model calibration by probability band, 2021-2026 (no prices needed)."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, ".")
import td_model as T
tpbp, tsched, tplayers = T.load()
tpg, tteam, tctx = T.build_tables(tpbp, tsched, tplayers)
tf = T.features(tpg, tteam, tctx); tf = tf[(tf.pr_n >= 1) & tf.implied.notna() & tf.rush_share.notna()]
out = []
for s in range(2021, 2027):
    tmm = T.fit_team(tteam[tteam.season < s]); mr, mp = T.fit_players(tf[tf.season < s], tmm)
    out.append(T.predict(tf[tf.season == s].copy(), tmm, mr, mp))
o = pd.concat(out)
o["y_"] = ((o.rtd.fillna(0) + o.ptd.fillna(0)) > 0).astype(int)
y = o.y_
o["p"] = 1 - np.exp(-o.lam); o["y"] = y
o = o[o.pos != "QB"] if "pos" in o.columns else o
o["band"] = pd.cut(o.p, [0, .10, .20, .26, .40, .45, .55, 1], labels=["<10%", "10-20%", "20-26% (~3.9-5.0)", "26-40% (~2.5-3.9)", "40-45%", "45-55%", "55%+"])
t = o.groupby("band", observed=True).agg(n=("y", "size"), model=("p", "mean"), actual=("y", "mean"))
t["gap_pts"] = (t.actual - t.model) * 100
print(t.round(3).to_string())
for s, x in o.groupby("season"):
    z = x[x.p.between(.26, .40)]; print(f"  {s} 26-40% band: model {z.p.mean():.3f} actual {z.y.mean():.3f} (n={len(z)})")
