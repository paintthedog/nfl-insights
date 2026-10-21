import numpy as np, pandas as pd, proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
pts = lambda o, p: (o[p + "rec"] + o[p + "rec_yds"] / 10 + o[p + "rush_yds"] / 10 + np.where(o.pos == "QB", o[p + "pass_yds"] / 25, 0))
rows = []
for name, cfg in {"current (defense: last season full weight)": dict(DEF_SEASON_CARRY=1.0),
                  "defense last season x0.5": dict(DEF_SEASON_CARRY=0.5),
                  "defense last season x0.25": dict(DEF_SEASON_CARRY=0.25),
                  "defense last season x0.25, less averaging": dict(DEF_SEASON_CARRY=0.25, DEF_K_PASS=150, DEF_K_RUSH=100)}.items():
    P.DEF_K_PASS, P.DEF_K_RUSH = 300, 200
    for k, v in cfg.items(): setattr(P, k, v)
    d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
    outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        outs.append(P.apply_calibration(te, P.fit_calibration(tr[tr.pr_n >= 1])))
    o = pd.concat(outs); o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))].copy()
    o["err"] = np.abs(pts(o, "") - pts(o, "p_"))
    row = dict(variant=name, **{"wk 1-4": o[o.week <= 4].err.mean(), "wk 5+": o[o.week >= 5].err.mean(), "all": o.err.mean()})
    for g, st in [("QB", "pass_yds"), ("RB", "rush_yds"), ("WR", "rec_yds"), ("TE", "rec_yds")]:
        r = o[o.pos == g]; row[f"{g} {st}"] = np.mean(np.abs(r[st] - r["p_" + st]))
    rows.append(row); print("done:", name, flush=True)
r = pd.DataFrame(rows).set_index("variant")
print(((r / r.iloc[0] - 1) * 100).round(2).to_string())
