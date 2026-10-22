"""Correlated same-game parlays from the joint game simulation. For each game: milestone legs (DK-style X+), joint hit rate
from the SAME simulated games, vs the product of the legs' own rates (what an independent pricing would assume)."""
import os, sys, itertools, numpy as np, pandas as pd
os.environ.setdefault("NFL_WEEK", "6")
src = open("proj_project.py").read().split("sim_players = {}")[0]
exec(src)
MS = {"rec": [3, 4, 5, 6, 7], "rec_yds": [25, 40, 50, 60, 70, 80, 100], "rush_yds": [25, 40, 50, 60, 70, 80, 100], "pass_yds": [175, 200, 225, 250, 275, 300], "td": [1]}
LBL = {"rec": "receptions", "rec_yds": "rec yds", "rush_yds": "rush yds", "pass_yds": "pass yds", "td": "anytime TD"}
SKIP_TEAMS_QB = {"ATL", "BAL"}            # starting QB unresolved
out = []
for gid, g in f.groupby("game_id"):
    team_rows = {}
    for tm, x in g.groupby("posteam"):
        x = x.copy(); x["name"] = x["name"].fillna(x.pid)
        team_rows[tm] = x[["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc",
                           "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share", "p_ints", "p_play", "p_pass_yds"]]
    sim = S.simulate_game(team_rows, n_sim=20000, rng=np.random.default_rng(7), pool_frac=J.POOL_FRAC)
    legs = []
    for tm, ss in sim.items():
        pp = dict(zip(team_rows[tm].pid, team_rows[tm].p_play.fillna(1)))
        for i, pid in enumerate(ss["pids"]):
            if pp.get(pid, 1) < 0.95: continue
            pos = ss["pos"][i]; nm = team_rows[tm].set_index("pid").name.get(pid, pid)
            stats = {"pass_yds": ss["pass_yds"][:, i]} if pos == "QB" else {"rec": ss["rec"][:, i], "rec_yds": ss["rec_yds"][:, i], "rush_yds": ss["rush_yds"][:, i], "td": ss["rec_td"][:, i] + ss["rush_td"][:, i]}
            if pos == "QB" and tm in SKIP_TEAMS_QB: continue
            for k, v in stats.items():
                for t in MS[k]:
                    hit = v >= t; p = hit.mean()
                    if 0.55 <= p <= 0.85: legs.append(dict(tm=tm, name=nm, pos=pos, stat=k, t=t, p=p, hit=hit))
    # one leg per player-stat: keep the milestone closest to 70%
    best = {}
    for L_ in legs:
        key = (L_["name"], L_["stat"])
        if key not in best or abs(L_["p"] - .70) < abs(best[key]["p"] - .70): best[key] = L_
    legs = list(best.values())
    for n in (2, 3):
        for combo in itertools.combinations(legs, n):
            if len({c["name"] for c in combo}) < n: continue
            if n == 3 and len({c["tm"] for c in combo}) > 1 and False: pass
            joint = np.logical_and.reduce([c["hit"] for c in combo]).mean(); indep = np.prod([c["p"] for c in combo])
            if 0.36 <= joint <= 0.52:
                out.append(dict(game=gid, n=n, legs=" + ".join(f"{c['name']} {c['t']}+ {LBL[c['stat']]}" if c["stat"] != "td" else f"{c['name']} anytime TD" for c in combo),
                                leg_ps=" / ".join(f"{c['p']:.0%}" for c in combo), joint=joint, indep=indep, lift=joint / indep, fair=1 / joint, naive=1 / indep))
o = pd.DataFrame(out); o.to_csv("sgp_wk6_all.csv", index=False)
pick = o.sort_values("lift", ascending=False).groupby("game").head(1).sort_values("lift", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(pick.head(12)[["game", "legs", "leg_ps", "joint", "indep", "lift", "fair", "naive"]].round(3).to_string(index=False))
