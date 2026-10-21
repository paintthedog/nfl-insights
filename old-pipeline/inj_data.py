"""Builds the injury test set: walk-forward projections 2019-2026 plus, for every team-game, which recent contributors
were absent and how much target / carry share they left behind. Writes data/inj_test.parquet."""
import numpy as np, pandas as pd
import proj_model as P

pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
out = []
for s in range(2019, 2027):
    tr, te = d[d.season < s].copy(), d[d.season == s].copy()
    va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[tr.pr_n >= 1]); out.append(P.apply_calibration(te, cal))
d = pd.concat(out)

# who was on the field: any touch, or any offensive snap
snaps = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_pct"]).rename(columns={"gsis_id": "pid"})
snaps = snaps[snaps.offense_pct > 0][["game_id", "pid"]].drop_duplicates()
tm_of = d[["game_id", "posteam"]].drop_duplicates()
played = pd.concat([d[["game_id", "posteam", "pid"]], snaps.merge(d[["pid", "posteam", "season"]].drop_duplicates().merge(d[["game_id", "season"]].drop_duplicates(), on="season"), on=["game_id", "pid"])[["game_id", "posteam", "pid"]]]).drop_duplicates()
played = played.merge(tm_of, on=["game_id", "posteam"])          # keep only rows where that team actually played that game
gm = d[["game_id", "posteam", "season", "week", "gameday"]].drop_duplicates().sort_values(["posteam", "gameday"])
gm["k"] = gm.groupby(["posteam", "season"]).cumcount()
sh = d[["game_id", "posteam", "pid", "pos", "tgt_sh", "car_sh"]].merge(gm[["game_id", "posteam", "season", "k"]], on=["game_id", "posteam"])
pk = played.merge(gm[["game_id", "posteam", "season", "k"]], on=["game_id", "posteam"])
on_field = set(zip(pk.posteam, pk.season, pk.k, pk.pid))
rows = []
for (tm, ssn), g in sh.groupby(["posteam", "season"]):
    byk = {k: x for k, x in g.groupby("k")}
    for k in sorted(byk):
        if k == 0: continue
        prev = pd.concat([byk[j] for j in (k - 1, k - 2, k - 3) if j in byk])
        cand = pd.concat([byk[j] for j in (k - 1, k - 2) if j in byk]).pid.unique()
        rec = prev[prev.pid.isin(cand)].groupby(["pid", "pos"], as_index=False)[["tgt_sh", "car_sh"]].mean()
        ab = rec[[ (tm, ssn, k, p) not in on_field for p in rec.pid ]]
        r = dict(posteam=tm, season=ssn, k=k)
        for grp in ["WR", "TE", "RB", "QB"]:
            a = ab[ab.pos == grp]
            r[f"vt_{grp}"] = a.tgt_sh.sum(); r[f"vc_{grp}"] = a.car_sh.sum()
        r["top_abs_t"] = ab.tgt_sh.max() if len(ab) else 0.0; r["top_abs_c"] = ab.car_sh.max() if len(ab) else 0.0
        rows.append(r)
v = pd.DataFrame(rows).fillna(0).merge(gm[["game_id", "posteam", "season", "k"]], on=["posteam", "season", "k"])
d = d.merge(v.drop(columns=["season", "k"]), on=["game_id", "posteam"], how="left")
for c in [c for c in v.columns if c.startswith(("vt_", "vc_", "top_abs"))]: d[c] = d[c].fillna(0)
d["vt_all"] = d.vt_WR + d.vt_TE + d.vt_RB; d["vc_all"] = d.vc_RB + d.vc_QB + d.vc_WR
# rank among active players at the same position by prior share
d["rk_t"] = d.groupby(["game_id", "posteam", "pos"]).pr_tgt_sh.rank(ascending=False, method="first")
d["rk_c"] = d.groupby(["game_id", "posteam", "pos"]).pr_car_sh.rank(ascending=False, method="first")
keep = ["game_id", "posteam", "pid", "pos", "season", "week", "gameday", "pr_n", "qb_start", "implied", "pr_tgt_sh", "pr_car_sh", "tgt", "car", "rec", "rec_yds", "rush_yds",
        "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "v_att", "v_car", "tm_tgt", "tm_car", "rk_t", "rk_c"] + [c for c in d.columns if c.startswith(("vt_", "vc_", "top_abs"))]
d[keep].to_parquet("data/inj_test.parquet")
print(len(d), "rows;", "team-games with 10%+ of targets vacated:", (d.drop_duplicates(["game_id", "posteam"]).vt_all >= .10).mean().round(3),
      "| 15%+ of carries:", (d.drop_duplicates(["game_id", "posteam"]).vc_all >= .15).mean().round(3))
