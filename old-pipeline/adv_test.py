"""Walk-forward tests of (1) route participation / targets per route run and (3) draft capital for inexperienced players."""
import numpy as np, pandas as pd, nflreadpy as nfl
import proj_model as P, td_model as T
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
# ---- routes (on the field for a dropback), 2019-2025 only: not published during the season
r = pd.read_parquet("data/routes.parquet").rename(columns={})
d = d.merge(r[["game_id", "pid", "routes", "tm_db"]], on=["game_id", "pid"], how="left")
has = d.game_id.isin(r.game_id.unique())
d.loc[has & d.routes.isna(), "routes"] = 0
d["tm_db"] = d.groupby(["game_id", "posteam"]).tm_db.transform("max")
d["route_sh"] = d.routes / d.tm_db
x = T.decayed_mean(d[["pid", "gameday", "game_id", "route_sh"]].copy(), "pid", ["route_sh"], P.SHARE_DECAY)[["pid", "game_id", "pr_route_sh"]]
lastr = T.decayed_mean(d[["pid", "gameday", "game_id", "route_sh"]].copy(), "pid", ["route_sh"], 0.001).rename(columns={"pr_route_sh": "last_route"})[["pid", "game_id", "last_route"]]
s = P.decayed_sum(d[["pid", "gameday", "game_id", "routes", "tgt"]].fillna({"routes": 0}).copy(), "pid", ["routes", "tgt"], P.EFF_DECAY)[["pid", "game_id", "ds_routes", "ds_tgt"]].rename(columns={"ds_routes": "dsr", "ds_tgt": "dst"})
d = d.merge(x, on=["pid", "game_id"], how="left").merge(lastr, on=["pid", "game_id"], how="left").merge(s, on=["pid", "game_id"], how="left")
lg_tprr = d.groupby("pos").apply(lambda g: g.tgt.sum() / max(g.routes.sum(), 1))
d["tprr"] = (d.dst + 60 * d.pos.map(lg_tprr)) / (d.dsr + 60)
d["has_route"] = d.pr_route_sh.notna()
d["l_route"] = np.log(d.pr_route_sh.fillna(np.exp(d.l_snap)).clip(0.02, 1))
d["route_dn"] = (d.last_route.fillna(d.pr_route_sh) - d.pr_route_sh).clip(-0.6, 0).fillna(0)
d["l_tprr"] = np.log(d.tprr.fillna(d.pos.map(lg_tprr)).clip(0.01, 0.5))
# ---- draft capital for players with little NFL history
dp = nfl.load_draft_picks().to_pandas().dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id").pick
d = d.sort_values(["pid", "gameday"]); d["cg"] = d.groupby("pid").cumcount()
pick = d.pid.map(dp)
d["dc"] = np.where(pick.notna(), 1 - np.log(pick.clip(1, 262)) / np.log(262), 0.0)     # 1.0 = first pick, ~0 = last pick / undrafted
d["young"] = (d.cg < 8).astype(float); d["dc_young"] = d.dc * d.young
d["dc_y4"] = d.dc * (d.cg < 4); d["y4"] = (d.cg < 4).astype(float)
print("route coverage (skill rows 2019-25):", round(d[d.pos.isin(["RB", "WR", "TE"]) & d.season.between(2019, 2025)].has_route.mean(), 3),
      "| inexperienced skill rows:", int((d.pos.isin(["RB", "WR", "TE"]) & (d.cg < 8) & (d.season >= 2021)).sum()))
REC = ["tgt", "rec", "rec_yds", "rec_1d"]; RUSH = ["car", "rush_yds", "rush_1d"]
base = {**{s: ["l_snap", "snap_dn", "l_wopr"] for s in REC}, **{s: ["l_snap", "snap_dn"] for s in RUSH}}
add = lambda rec=[], rush=[]: {**{s: base[s] + rec for s in REC}, **{s: base[s] + rush for s in RUSH}}
variants = {
    "current": base,
    "+ route share": add(["l_route", "route_dn"]),
    "route share instead of snap": {**{s: ["l_route", "route_dn", "l_wopr"] for s in REC}, **{s: base[s] for s in RUSH}},
    "+ targets per route": add(["l_tprr"]),
    "+ route share + targets per route": add(["l_route", "route_dn", "l_tprr"]),
    "+ draft capital (<8 games)": add(["young", "dc_young"], ["young", "dc_young"]),
    "+ draft capital (<4 and <8)": add(["young", "dc_young", "y4", "dc_y4"], ["young", "dc_young", "y4", "dc_y4"]),
}
import sys
only = sys.argv[1:] 
res = []; keepo = {}
for name, extra in variants.items():
    P.EXTRA = extra; outs = []
    for s in range(2022, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); outs.append(P.apply_calibration(te, cal))
    o = pd.concat(outs); o = o[(o.pr_n >= 1) & o.implied.notna() & o.pos.isin(["RB", "WR", "TE"])]
    keepo[name] = o
    for lab, mk in [("2022-25 all", o.season <= 2025), ("2022-26 all", o.season >= 2022), ("2022-26 <8 career games", o.cg < 8), ("2022-26 rookies rd 1-3 <8g", (o.cg < 8) & (o.dc > 0.4))]:
        y = o[mk]; row = dict(variant=name, sample=lab, n=len(y))
        for grp, st in [("WR", "tgt"), ("WR", "rec_yds"), ("TE", "tgt"), ("TE", "rec_yds"), ("RB", "tgt"), ("RB", "car"), ("RB", "rush_yds")]:
            z = y[y.pos == grp]; row[f"{grp} {st}"] = np.mean(np.abs(z[st] - z[f"p_{st}"]))
            row[f"{grp} {st} bias"] = z[f"p_{st}"].sum() / max(z[st].sum(), 1) - 1
        res.append(row)
    print("done", name, flush=True)
R = pd.DataFrame(res); R.to_csv("adv_test_results.csv", index=False)
pd.set_option("display.width", 250)
cols = [c for c in R.columns if c not in ("variant", "sample", "n") and "bias" not in c]
for lab, g in R.groupby("sample", sort=False):
    g = g.set_index("variant"); b = g.loc["current", cols]
    print(f"\n{lab} (n={int(g.n.iloc[0])}): MAE, % change vs current (negative = better)")
    print(pd.concat([g.loc[["current"], cols].round(3), ((g[cols] / b - 1) * 100).round(2).iloc[1:]]).to_string())
    print("bias:", g[[c for c in g.columns if "bias" in c]].loc[["current", "+ draft capital (<8 games)"]].round(3).to_string())
