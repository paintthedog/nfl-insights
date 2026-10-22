"""Does a snap-share JUMP help projections for first- and second-year players (ascending roles), even though it
hurt when pooled across everyone? Adds snap_up_yk = max(snap_trend,0) for players in season 1-2 of their career."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position", "rookie_season"])
d = P.build(pbp, sched, players[["gsis_id","display_name","position"]]); pm = P.position_means(d); d = P.efficiency(d, pm)
rk = players.drop_duplicates("gsis_id").set_index("gsis_id").rookie_season
d["yk"] = (d.season - d.pid.map(rk)).fillna(9)
d["young"] = (d.yk <= 1).astype(float)
d["snap_up_yk"] = d.snap_trend.clip(lower=0) * d.young
BASE_P = ["l_snap", "snap_dn", "l_wopr", "qd"]; BASE_R = ["l_snap", "snap_dn"]
def run(up):
    P.EXTRA = {**{s: BASE_P + (["snap_up_yk"] if up else []) for s in ["tgt", "rec", "rec_yds", "rec_1d"]},
               **{s: BASE_R + (["snap_up_yk"] if up else []) for s in ["car", "rush_yds", "rush_1d"]}}
    out = []
    for s in range(2021, 2027):
        tr = d[d.season < s].copy(); te = d[d.season == s].copy(); teams = tr.drop_duplicates(["game_id", "posteam"])
        va, vc = P.fit_volume(teams); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); out.append(P.apply_calibration(te, cal))
    o = pd.concat(out); return o[(o.pr_n >= 1) & o.implied.notna() & o.pos.isin(["RB", "WR", "TE"])]
A = run(0); B = run(1)
mae = lambda x, st: np.abs(x[st] - x["p_" + st]).mean()
for nm, sel in (("all", lambda x: x.index == x.index), ("yr-1/2 players", lambda x: x.young == 1),
                ("yr-1/2 with snap jump >= .1", lambda x: (x.young == 1) & (x.snap_trend >= .1))):
    a, b = A[sel(A)], B[sel(B)]
    print(nm, "n", len(a), " ".join(f"{st}: {mae(a, st):.3f}->{mae(b, st):.3f}" for st in ["tgt", "rec_yds", "car", "rush_yds"]), flush=True)
x = A[(A.young == 1) & (A.snap_trend >= .1)]
print("bias check, young+jump: actual-minus-proj tgt {:+.2f}, rec_yds {:+.2f} (n={})".format((x.tgt - x.p_tgt).mean(), (x.rec_yds - x.p_rec_yds).mean(), len(x)))
