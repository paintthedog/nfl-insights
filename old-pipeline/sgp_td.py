"""Correlated anytime-TD parlays (2 and 3 legs, same game) from the joint simulation."""
import os, itertools, numpy as np, pandas as pd
os.environ.setdefault("NFL_WEEK", "6")
exec(open("proj_project.py").read().split("sim_players = {}")[0])
SKIP = {"BAL", "CHI"}   # QB unresolved
out, single = [], []
for gid, g in f.groupby("game_id"):
    team_rows = {}
    for tm, x in g.groupby("posteam"):
        x = x.copy(); x["name"] = x["name"].fillna(x.pid)
        team_rows[tm] = x[["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc",
                           "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share", "p_ints", "p_play", "p_pass_yds"]]
    sim = S.simulate_game(team_rows, n_sim=40000, rng=np.random.default_rng(11), pool_frac=J.POOL_FRAC)
    legs = []
    for tm, ss in sim.items():
        if tm in SKIP: continue
        pp = dict(zip(team_rows[tm].pid, team_rows[tm].p_play.fillna(1))); nmap = team_rows[tm].set_index("pid").name
        for i, pid in enumerate(ss["pids"]):
            if pp.get(pid, 1) < 0.95: continue
            td = (ss["rec_td"][:, i] + ss["rush_td"][:, i]) >= 1; p = td.mean()
            if p >= 0.28: legs.append(dict(tm=tm, name=nmap.get(pid, pid), pos=ss["pos"][i], p=p, hit=td)); single.append((gid, tm, nmap.get(pid, pid), ss["pos"][i], p))
    for n in (2, 3):
        for c in itertools.combinations(legs, n):
            j = np.logical_and.reduce([x["hit"] for x in c]).mean(); ind = np.prod([x["p"] for x in c])
            out.append(dict(game=gid.replace("2026_06_", ""), n=n, same_team=len({x["tm"] for x in c}) == 1,
                            legs=" + ".join(f"{x['name']} ({x['tm']} {x['pos']}, {x['p']:.0%})" for x in c), joint=j, indep=ind, lift=j / ind, fair=1 / j))
o = pd.DataFrame(out); o.to_csv("sgp_td_wk6_all.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 140)
print("lift by structure:"); print(o.assign(kind=np.where(o.same_team, "same team", "opposite teams")).groupby(["n", "kind"]).lift.describe()[["count", "mean", "min", "max"]].round(3))
for n in (2, 3):
    print(f"\n== {n}-leg, best lift (joint >= {'0.15' if n==2 else '0.05'})")
    print(o[(o.n == n) & (o.joint >= (0.15 if n == 2 else 0.05))].sort_values("lift", ascending=False).head(12)[["game", "legs", "joint", "indep", "lift", "fair"]].round(3).to_string(index=False))
