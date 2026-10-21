"""Do missing defensive regulars on the OPPONENT, split secondary vs front seven, improve player projections?
o_sec joins receiving stats, o_f7 rushing stats, as extra calibration features. Walk-forward 2021-2026."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); pm = P.position_means(d); d = P.efficiency(d, pm)
dm = pd.read_parquet("data/team_dmiss.parquet").rename(columns={"team": "opp", "sec": "o_sec", "f7": "o_f7"})
d = d.merge(dm, on=["game_id", "opp"], how="left"); d[["o_sec", "o_f7"]] = d[["o_sec", "o_f7"]].fillna(0.0)
BASE_R = ["l_snap", "snap_dn"]; BASE_P = ["l_snap", "snap_dn", "l_wopr", "qd"]
def run(sec, f7):
    P.EXTRA = {**{s: BASE_P + (["o_sec"] if sec else []) for s in ["tgt", "rec", "rec_yds", "rec_1d"]},
               **{s: BASE_R + (["o_f7"] if f7 else []) for s in ["car", "rush_yds", "rush_1d"]}}
    out = []
    for s in range(2021, 2027):
        tr = d[d.season < s].copy(); te = d[d.season == s].copy(); teams = tr.drop_duplicates(["game_id", "posteam"])
        va, vc = P.fit_volume(teams); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); out.append(P.apply_calibration(te, cal))
    o = pd.concat(out); return o[(o.pr_n >= 1) & o.implied.notna() & o.pos.isin(["RB", "WR", "TE"])]
A = run(0, 0); B = run(1, 1)
mae = lambda x, st: np.abs(x[st] - x["p_" + st]).mean()
for nm, sel in (("all", lambda x: x.o_sec.notna()), ("opp missing 1.5+ secondary", lambda x: x.o_sec >= 1.5), ("opp missing 1.5+ front7", lambda x: x.o_f7 >= 1.5)):
    a, b = A[sel(A)], B[sel(B)]
    print(nm, "n", len(a), " ".join(f"{st}: {mae(a, st):.3f}->{mae(b, st):.3f}" for st in ["tgt", "rec", "rec_yds", "car", "rush_yds"]), flush=True)
# bias check: do players beat their projection when the unit is depleted?
for st, col, cut in (("rec_yds", "o_sec", 1.5), ("rush_yds", "o_f7", 1.5)):
    x = A[A[col] >= cut]; print(f"bias {st} when {col}>={cut}: actual-minus-proj {np.mean(x[st] - x['p_'+st]):+.2f} (n={len(x)}); everyone {np.mean(A[st]-A['p_'+st]):+.2f}")
