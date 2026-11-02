"""Pool dilution (Nov 2). Published receiving yards / carries ran ~8% below the raw model for players who played (Weeks 7-8).
Is that because the team total gets spread over players who don't play? Uses the last pre-kickoff archive for each game
and counts EVERY published pool player (a player who never touched the ball scores 0), so there is no selection on playing."""
import sys; sys.path.insert(0, ".")
import glob, json, numpy as np, pandas as pd
import proj_model as PM
pg = PM.player_games(PM.load())[0]
sched = pd.read_parquet("data/sched.parquet"); sched = sched[sched.season == 2026]
rows = []
for wk in sorted({int(f.split("_wk")[1].split("_")[0]) for f in glob.glob("archive/data_wk*_*.json")}):
    sw = sched[(sched.week == wk) & sched.home_score.notna()]
    snaps = sorted(glob.glob(f"archive/data_wk{wk}_*.json"))
    st = lambda f: f.split("_")[-2] + f.split("_")[-1][:4]
    for r in sw.itertuples():
        pre = [f for f in snaps if st(f) < r.gameday.replace("-", "") + r.gametime.replace(":", "")]
        if not pre: continue
        D = json.load(open(pre[-1])); act = pg[pg.game_id == r.game_id].set_index("pid")
        for p in D["proj"]["players"]:
            if p["team"] not in (r.home_team, r.away_team) or p["pos"] == "QB": continue
            m = p.get("mean") or {}; pp = p.get("p_play") or 1.0
            a = act.loc[p["pid"]] if p["pid"] in act.index else None
            if isinstance(a, pd.DataFrame): a = a.iloc[0]
            rows.append(dict(wk=wk, game_id=r.game_id, team=p["team"], pid=p["pid"], name=p["name"], pos=p["pos"], p_play=pp,
                             proj_ry=(m.get("rec_yds") or 0) * pp, proj_car=(m.get("car") or 0) * pp, proj_tgt=(m.get("tgt") or 0) * pp,
                             played=a is not None and (a.tgt + a.car) > 0, ry=0.0 if a is None else float(a.rec_yds), car=0.0 if a is None else float(a.car), tgt=0.0 if a is None else float(a.tgt)))
o = pd.DataFrame(rows)
# team receiving/rushing that went to players NOT in the published pool
_pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]).drop_duplicates("gsis_id").set_index("gsis_id").position
pg["pos"] = pg.pid.map(_pl)
tm = pg[pg.game_id.isin(o.game_id) & (pg.pos != "QB")].groupby(["game_id", "posteam"])[["rec_yds", "car", "tgt"]].sum()
po = o.groupby(["game_id", "team"])[["proj_ry", "proj_car", "proj_tgt", "ry", "car", "tgt"]].sum()
po = po.join(tm.rename_axis(["game_id", "team"]).add_prefix("all_"))
print(f"{len(o)} pool rows, {o.game_id.nunique()} games")
print("team totals (non-QB): projected rec yds %.0f | actual by pool players %.0f | actual all players %.0f" % (po.proj_ry.sum(), po.ry.sum(), po.all_rec_yds.sum()))
print("                      projected carries %.0f | actual by pool players %.0f | actual all players %.0f" % (po.proj_car.sum(), po.car.sum(), po.all_car.sum()))
o["band"] = pd.cut(o.proj_ry, [-0.01, 2, 8, 20, 40, 400])
print("\nreceiving yards by published band (every pool player; non-players count 0):")
print(o.groupby("band", observed=True).agg(n=("ry", "size"), played=("played", "mean"), proj=("proj_ry", "sum"), actual=("ry", "sum")).assign(ratio=lambda x: x.actual / x.proj).round(2).to_string())
o["cband"] = pd.cut(o.proj_car, [-0.01, 0.5, 3, 8, 14, 60])
r = o[o.pos == "RB"]
print("\ncarries by published band (RBs):")
print(r.groupby("cband", observed=True).agg(n=("car", "size"), played=("played", "mean"), proj=("proj_car", "sum"), actual=("car", "sum")).assign(ratio=lambda x: x.actual / x.proj).round(2).to_string())
o.to_csv("/tmp/claude-0/pool_rows.csv", index=False)
