"""Grade replayed weeks (replay/out/proj_<S>_wk<W>.json) the way the scorecard grades live weeks: every listed player who
took an offensive snap (a 0-touch day counts), randomized-PIT range coverage, and the volume-sharing check (every listed
player, non-players score 0, by published band). Usage: python replay/grade.py [2024 2025]   (from pipeline/)"""
import sys; sys.path.insert(0, ".")
import glob, json, re, numpy as np, pandas as pd
import proj_model as PM
import os
TAG = os.environ.get("REPLAY_TAG", "")
SEAS = [int(x) for x in sys.argv[1:]] or [2024, 2025]
pg = PM.player_games(PM.load())[0]
pg["fp_ppr"] = pg.pass_yds * .04 + pg.pass_td * 4 - pg.ints * 2 + pg.rush_yds * .1 + pg.rush_td * 6 + pg.rec + pg.rec_yds * .1 + pg.rec_td * 6
pg = pg[pg.game_id.str[:4].astype(int).isin(SEAS)]
act = pg.set_index(["game_id", "pid"])
sn = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_snaps"])
on = set(map(tuple, sn[sn.offense_snaps.fillna(0) > 0][["game_id", "gsis_id"]].values))
sched = pd.read_parquet("data/sched.parquet"); sched = sched[sched.season.isin(SEAS)]
QL = PM.QUANTS
STATS = ["rec", "tgt", "rec_yds", "rush_yds", "car", "pass_yds", "fp_ppr"]
rows = []
for f in sorted(glob.glob(f"replay/out/{TAG}proj_*_wk*.json")):
    S, W = map(int, re.findall(r"proj_(\d+)_wk(\d+)", f)[0])
    if S not in SEAS: continue
    sw = sched[(sched.season == S) & (sched.week == W)]
    gid = {t: g for g, h, a in zip(sw.game_id, sw.home_team, sw.away_team) for t in (h, a)}
    for p in json.load(open(f))["players"]:
        g = gid.get(p["team"])
        if g is None: continue
        k = (g, p["pid"])
        if k in act.index: a = act.loc[k]; a = a.iloc[0] if isinstance(a, pd.DataFrame) else a; played = True
        else: a = None; played = k in on
        r = dict(season=S, week=W, game_id=g, team=p["team"], pid=p["pid"], name=p["name"], pos=p["pos"], p_play=p.get("p_play") or 1.0, flag=p.get("flag") or "",
                 on=played or a is not None)
        for st in STATS:
            m = (p.get("mean") or {}).get(st); q = (p.get("q") or {}).get(st)
            v = None if a is None else float(a[st]) if st in a else None
            if v is None and r["on"]: v = 0.0
            r[f"m_{st}"] = m; r[f"a_{st}"] = v
            if q and v is not None and max(q) > 0:
                flo = max([QL[i] for i, x in enumerate(q) if x < v], default=0.0); fhi = min([QL[i] for i, x in enumerate(q) if x > v], default=1.0); w = max(fhi - flo, 1e-9)
                part = lambda lo, hi: max(0.0, min(fhi, hi) - max(flo, lo)) / w
                r[f"b_{st}"], r[f"u_{st}"], r[f"i_{st}"] = part(0, .1), part(.9, 1), part(.25, .75)
        rows.append(r)
o = pd.DataFrame(rows)
o.to_pickle(f"replay/out/{TAG}graded.pkl")
pd.set_option("display.width", 220)
print(f"{o.groupby(['season','week']).ngroups} replayed weeks, {len(o)} listed player-games, {int(o.on.sum())} on the field")
g = o[o.on & (o.p_play >= 0.9)]
res = []
for st in STATS:
    x = g[g[f"m_{st}"].notna() & g[f"a_{st}"].notna() & (g[f"m_{st}"] >= 0.5 if st != "fp_ppr" else g[f"m_{st}"] >= 3)]
    if st in ("rec", "tgt", "rec_yds"): x = x[x.pos != "QB"]
    if st == "pass_yds": x = x[x.pos == "QB"]
    e = x[f"a_{st}"] - x[f"m_{st}"]
    res.append(dict(stat=st, n=len(x), proj=x[f"m_{st}"].mean(), act=x[f"a_{st}"].mean(), bias=e.mean(), mae=e.abs().mean(),
                    below=x[f"b_{st}"].mean(), above=x[f"u_{st}"].mean(), mid=x[f"i_{st}"].mean()))
print("\nby stat (on the field, p_play 0.9+):"); print(pd.DataFrame(res).round(3).to_string(index=False))
nq = o[o.pos != "QB"].copy()
nq["proj_ry"] = nq.m_rec_yds.fillna(0) * nq.p_play; nq["ry"] = nq.a_rec_yds.fillna(0)
nq["band"] = pd.cut(nq.proj_ry, [-0.01, 2, 8, 20, 40, 400])
print("\nvolume sharing, receiving yards by published band (every listed player; no stat line = 0):")
print(nq.groupby("band", observed=True).agg(n=("ry", "size"), played=("on", "mean"), proj=("proj_ry", "sum"), actual=("ry", "sum")).assign(ratio=lambda x: x.actual / x.proj).round(3).to_string())
rb = o[o.pos == "RB"].copy(); rb["pc"] = rb.m_car.fillna(0) * rb.p_play; rb["ac"] = rb.a_car.fillna(0)
rb["band"] = pd.cut(rb.pc, [-0.01, 0.5, 3, 8, 14, 60])
print("\nvolume sharing, RB carries by published band:")
print(rb.groupby("band", observed=True).agg(n=("ac", "size"), played=("on", "mean"), proj=("pc", "sum"), actual=("ac", "sum")).assign(ratio=lambda x: x.actual / x.proj).round(3).to_string())
tm = pg[pg.game_id.isin(o.game_id)]
tm = tm.assign(pos=tm.pid.map(pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]).drop_duplicates("gsis_id").set_index("gsis_id").position))
allry = tm[tm.pos != "QB"].rec_yds.sum()
print(f"\nteam receiving yards: projected (listed, x p_play) {nq.proj_ry.sum():.0f} | caught by listed players {nq.ry.sum():.0f} | by all non-QBs {allry:.0f}")
