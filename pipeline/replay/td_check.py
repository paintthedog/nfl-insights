"""Replay check of touchdown chances and multi-TD games (Nov 3). Uses tagged replay outputs (REPLAY_TAG, default cb_ = the
adopted config). For every listed player who took a snap: the site's anytime-TD chance (TD model, p_td if he plays), the
simulator's chance, and how often he scored 1+ and 2+."""
import sys; sys.path.insert(0, ".")
import glob, json, os, re, numpy as np, pandas as pd
import proj_model as PM
TAG = os.environ.get("REPLAY_TAG", "cb_")
pg = PM.player_games(PM.load())[0]; pg = pg[pg.game_id.str[:4].isin(["2024", "2025"])]
pg["td"] = pg.rush_td + pg.rec_td
act = pg.set_index(["game_id", "pid"])
sn = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_snaps"])
on = set(map(tuple, sn[sn.offense_snaps.fillna(0) > 0][["game_id", "gsis_id"]].values))
sched = pd.read_parquet("data/sched.parquet")
rows = []
for f in sorted(glob.glob(f"replay/out/{TAG}proj_*_wk*.json")):
    S, W = map(int, re.findall(r"proj_(\d+)_wk(\d+)", f)[0])
    sw = sched[(sched.season == S) & (sched.week == W)]
    gid = {t: g for g, h, a in zip(sw.game_id, sw.home_team, sw.away_team) for t in (h, a)}
    for p in json.load(open(f))["players"]:
        g = gid.get(p["team"]); k = (g, p["pid"])
        if g is None or (k not in act.index and k not in on): continue
        a = act.loc[k] if k in act.index else None
        if isinstance(a, pd.DataFrame): a = a.iloc[0]
        sim = p.get("sim") or {}
        rows.append(dict(season=S, week=W, pos=p["pos"], p_play=p.get("p_play") or 1, p_td=p.get("p_td"), sim_p=sim.get("p_td"), sim_mean=(sim.get("mean") or {}).get("td"),
                         td=0 if a is None else int(a.td), ppr_q90=(p.get("q") or {}).get("fp_ppr", [None] * 21)[18], ppr=None if a is None else float(a.pass_yds * .04 + a.pass_td * 4 - a.ints * 2 + a.rush_yds * .1 + a.rush_td * 6 + a.rec + a.rec_yds * .1 + a.rec_td * 6)))
o = pd.DataFrame(rows); o = o[o.pos.isin(["RB", "WR", "TE"]) & o.p_td.notna()]
o["y1"] = (o.td >= 1).astype(float); o["y2"] = (o.td >= 2).astype(float)
pd.set_option("display.width", 200)
print(f"{len(o)} player-games (RB/WR/TE on the field), {TAG}")
o["band"] = pd.cut(o.p_td, [0, .1, .2, .3, .4, .5, .6, 1])
print("\nanytime TD, the site's chance (TD model) by band:")
print(o.groupby("band", observed=True).agg(n=("y1", "size"), said=("p_td", "mean"), scored=("y1", "mean"), sim_said=("sim_p", "mean")).round(3).to_string())
for s in (2024, 2025):
    x = o[o.season == s]; print(f"  {s}: said {x.p_td.mean():.3f} scored {x.y1.mean():.3f} | Brier site {((x.p_td - x.y1) ** 2).mean():.4f} sim {((x.sim_p - x.y1) ** 2).mean():.4f}")
top = o[o.p_td >= 0.4]; z = (top.y1.mean() - top.p_td.mean()) / np.sqrt(top.p_td.mean() * (1 - top.p_td.mean()) / len(top))
print(f"  40%+: n={len(top)} said {top.p_td.mean():.3f} scored {top.y1.mean():.3f} (z {z:+.1f})")
lam = -np.log(1 - o.p_td.clip(upper=0.95)); o["p2_pois"] = 1 - np.exp(-lam) * (1 + lam)
print("\n2+ touchdowns: Poisson from the site's chance vs actual, by position:")
print(o.groupby("pos").agg(n=("y2", "size"), p2_said=("p2_pois", "mean"), p2_actual=("y2", "mean")).round(4).to_string())
r = o[(o.pos == "RB") & o.ppr.notna() & o.ppr_q90.notna()]
ab = r[r.ppr > r.ppr_q90]
print(f"\nRB games above the PPR ceiling: {len(ab)} of {len(r)} ({len(ab) / len(r):.1%}); of those, 2+ TDs {ab.y2.mean():.1%}, 1 TD {(ab.td == 1).mean():.1%}, 0 TD {(ab.td == 0).mean():.1%}")
print(f"RB 2+ TD rate overall {r.y2.mean():.3%} vs Poisson {r.p2_pois.mean():.3%}")
