"""Does game-day weather improve the projections? Walk-forward 2021-2026, actual game-day conditions."""
import numpy as np, pandas as pd, proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
pts = lambda o, p: (o[p + "rec"] + o[p + "rec_yds"] / 10 + o[p + "rush_yds"] / 10 + np.where(o.pos == "QB", o[p + "pass_yds"] / 25, 0))
rows = []
for name, use in [("no weather", False), ("wind + cold", True)]:
    P.USE_WEATHER = use
    d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
    outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); outs.append(P.apply_calibration(te, cal))
        if s == 2026 and use:
            m = cal[("pass_yds", "QB")]; print("QB pass yds: wind coef", round(m.params["wind_eff"], 4), "cold", round(m.params["cold"], 3))
    o = pd.concat(outs); o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))].copy()
    o["err"] = np.abs(pts(o, "") - pts(o, "p_"))
    windy = o.wind_eff >= 15
    row = dict(variant=name, all=o.err.mean(), windy_games=o[windy].err.mean(), n_windy=int(windy.sum()), cold_games=o[o.cold == 1].err.mean())
    for g, st in [("QB", "pass_yds"), ("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "rush_yds")]:
        r = o[o.pos == g]; row[f"{g} {st}"] = np.mean(np.abs(r[st] - r["p_" + st])); rw = r[r.wind_eff >= 15]; row[f"{g} {st} windy"] = np.mean(np.abs(rw[st] - rw["p_" + st]))
    rows.append(row)
r = pd.DataFrame(rows).set_index("variant"); pd.set_option("display.width", 220)
print(r.round(3).to_string()); print("\n% vs no weather (negative = better)\n", ((r / r.iloc[0] - 1) * 100).round(2).to_string())
