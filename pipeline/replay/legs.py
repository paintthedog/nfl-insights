"""Leg calibration on replayed weeks (Nov 14): for every listed player on the field (p_play 0.85+), pick the round threshold the
combos would pick (chance closest to 70%, combo_slate rules) and check how often it hit; plus how often results fall below each
published percentile. Usage: REPLAY_TAG=x_ python replay/legs.py [2024 2025]   (from pipeline/)"""
import sys; sys.path.insert(0, ".")
import glob, json, re, os, numpy as np, pandas as pd
import proj_model as PM
TAG = os.environ.get("REPLAY_TAG", ""); SEAS = [int(x) for x in sys.argv[1:]] or [2024, 2025]
pg = PM.player_games(PM.load())[0]; pg = pg[pg.game_id.str[:4].astype(int).isin(SEAS)]; act = pg.set_index(["game_id", "pid"])
act = act[~act.index.duplicated()]
sn = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_snaps"])
on = set(map(tuple, sn[sn.offense_snaps.fillna(0) > 0][["game_id", "gsis_id"]].values))
sched = pd.read_parquet("data/sched.parquet")
KS = {"rec": list(range(1, 13)), "pass_yds": list(range(150, 351, 25)), "rush_yds": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]}
KS["rec_yds"] = KS["rush_yds"]
def p_over(arr, line, QS):
    arr = np.asarray(arr, float)
    if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
    for i in range(1, len(arr)):
        if line <= arr[i]:
            t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
    return max(0, (1 - QS[-1]) * .5)
rows = []; pit = []
for f in sorted(glob.glob(f"replay/out/{TAG}proj_*_wk*.json")):
    S, W = map(int, re.findall(r"proj_(\d+)_wk(\d+)", f)[0])
    if S not in SEAS: continue
    D = json.load(open(f)); QS = D["quants"]; sw = sched[(sched.season == S) & (sched.week == W)]
    gid = {t: g for g, h, a in zip(sw.game_id, sw.home_team, sw.away_team) for t in (h, a)}
    for p in D["players"]:
        g = gid.get(p["team"]); k = (g, p["pid"])
        if g is None or (p.get("p_play") or 1) < .85 or not (k in act.index or k in on): continue
        for st in (["pass_yds"] if p["pos"] == "QB" else ["rush_yds", "rec"] if p["pos"] == "RB" else ["rec", "rec_yds"]):
            q = (p.get("q") or {}).get(st); m = (p.get("mean") or {}).get(st) or 0
            if not q or max(q) <= 0: continue
            v = float(act.loc[k][st]) if k in act.index else 0.0
            for T in (0.70, 0.5, 0.85):
                kk = min(KS[st], key=lambda x: abs(p_over(q, x - .5, QS) - T)); pr = p_over(q, kk - .5, QS)
                if abs(pr - T) <= .15 and (T != .7 or .55 <= pr <= .85):
                    rows.append(dict(season=S, pos=p["pos"], stat=st, T=T, k=kk, p=pr, hit=v > kk - .5, mean=m, tgt=(p["mean"].get("tgt") or 0), car=(p["mean"].get("car") or 0)))
            pit.append(dict(season=S, pos=p["pos"], stat=st, mean=m, **{f"q{int(x*100)}": (v < q[i]) + .5 * (v == q[i]) for i, x in enumerate(QS) if x in (.1, .2, .3, .4, .5, .7, .9)}))
L = pd.DataFrame(rows); L.to_pickle(f"replay/out/{TAG}legs.pkl"); PT = pd.DataFrame(pit)
pd.set_option("display.width", 200)
reg = L[((L.stat.isin(["rec", "rec_yds"])) & (L.tgt >= 4)) | ((L.stat == "rush_yds") & (L.car >= 8)) | (L.stat == "pass_yds")]
print("legs nearest each target chance, regulars (4+ proj targets / 8+ carries / QBs):")
print(reg.groupby(["stat", "T"]).agg(n=("hit", "size"), said=("p", "mean"), hit=("hit", "mean")).round(3).to_string())
print("\n70% catches legs by threshold (regulars):"); print(reg[(reg.stat == "rec") & (reg["T"] == .7)].groupby("k").agg(n=("hit", "size"), said=("p", "mean"), hit=("hit", "mean")).round(3).to_string())
print("\n70% legs by season:"); print(reg[reg["T"] == .7].groupby(["stat", "season"]).agg(n=("hit", "size"), said=("p", "mean"), hit=("hit", "mean")).round(3).to_string())
R = PT[((PT.stat.isin(["rec", "rec_yds"])) & (PT["mean"] >= (2.5 if True else 0)) & (PT.stat == "rec")) | ((PT.stat == "rec_yds") & (PT["mean"] >= 30)) | ((PT.stat == "rush_yds") & (PT["mean"] >= 35)) | (PT.stat == "pass_yds")]
print("\nshare of results below each published percentile (ties half), regulars:"); print(R.groupby("stat")[[c for c in PT if c.startswith("q")]].mean().round(3).assign(n=R.groupby("stat").size()).to_string())
