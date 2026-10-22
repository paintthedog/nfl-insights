import numpy as np, pandas as pd, proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.efficiency(P.build(pbp, sched, players), None) if False else None
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
d["snap_up"] = d.snap_trend.clip(lower=0); d["snap_dn"] = d.snap_trend.clip(upper=0)
REC = ["tgt", "rec", "rec_yds", "rec_1d"]; RUSH = ["car", "rush_yds", "rush_1d"]; SNAP = ["l_snap", "snap_trend"]
out = {}
V = {"current": {},
     "snap + WOPR": {**{s: SNAP + ["l_wopr"] for s in REC}, **{s: SNAP for s in RUSH}},
     "split up/down": {**{s: ["l_snap", "snap_up", "snap_dn", "l_wopr"] for s in REC}, **{s: ["l_snap", "snap_up", "snap_dn"] for s in RUSH}},
     "falls only": {**{s: ["l_snap", "snap_dn", "l_wopr"] for s in REC}, **{s: ["l_snap", "snap_dn"] for s in RUSH}}}
for name, extra in V.items():
    P.EXTRA = extra; outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        outs.append(P.apply_calibration(te, P.fit_calibration(tr[tr.pr_n >= 1])))
    o = pd.concat(outs); o = o[(o.pr_n >= 1) & o.implied.notna() & o.pos.isin(["RB", "WR", "TE"])]
    o["pts"] = o.rec + o.rec_yds / 10 + o.rec_1d + o.rush_yds / 10 + o.rush_1d
    o["p_pts"] = o.p_rec + o.p_rec_yds / 10 + o.p_rec_1d + o.p_rush_yds / 10 + o.p_rush_1d
    out[name] = o
c = out["current"]
masks = [("all games", c.snap_trend.notna()), ("snap share jumped 15+ pts", c.snap_trend > 0.15), ("snap share fell 15+ pts", c.snap_trend < -0.15)]
for nm in ["snap + WOPR", "split up/down", "falls only"]:
    n = out[nm]; print(nm)
    for label, m in masks:
        e0 = np.mean(np.abs(c[m].pts - c[m].p_pts)); e1 = np.mean(np.abs(n[m].pts - n[m].p_pts))
        print(f"   {label:28s} n={m.sum():6d}  current {e0:.2f}  new {e1:.2f}  change {100*(e1/e0-1):+.1f}%")
