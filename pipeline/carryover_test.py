"""How much should last season (and a player's old team) count? Walk-forward 2021-2026."""
import numpy as np, pandas as pd
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
DEF = dict(SEASON_CARRY=1.0, TEAM_CARRY=1.0, EFF_SEASON_CARRY=1.0, TEAM_VOL_CARRY=1.0)
V = {
    "current (no extra discount)":            {},
    "last season x0.6":                       dict(SEASON_CARRY=0.6),
    "last season x0.35":                      dict(SEASON_CARRY=0.35),
    "last season x0.15":                      dict(SEASON_CARRY=0.15),
    "old team x0.3":                          dict(TEAM_CARRY=0.3),
    "old team x0.1":                          dict(TEAM_CARRY=0.1),
    "last season x0.35 + old team x0.3":      dict(SEASON_CARRY=0.35, TEAM_CARRY=0.3),
    "+ team volume last season x0.5":         dict(SEASON_CARRY=0.35, TEAM_CARRY=0.3, TEAM_VOL_CARRY=0.5),
    "+ efficiency last season x0.5":          dict(SEASON_CARRY=0.35, TEAM_CARRY=0.3, EFF_SEASON_CARRY=0.5),
}
pts = lambda o, p: (o[p + "rec"] + o[p + "rec_yds"] / 10 + o[p + "rush_yds"] / 10
                    + np.where(o.pos == "QB", o[p + "pass_yds"] / 25, 0))
rows = []
for name, cfg in V.items():
    for k, v in {**DEF, **cfg}.items(): setattr(P, k, v)
    d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
    outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        outs.append(P.apply_calibration(te, P.fit_calibration(tr[tr.pr_n >= 1])))
    o = pd.concat(outs)
    o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))].copy()
    o["err"] = np.abs(pts(o, "") - pts(o, "p_"))
    moved = o.groupby("pid").posteam.transform(lambda t: t != t.shift()) & (o.groupby("pid").cumcount() > 0)
    row = dict(variant=name, **{f"weeks 1-4": o[o.week <= 4].err.mean(), "weeks 5-9": o[o.week.between(5, 9)].err.mean(),
                                "weeks 10+": o[o.week >= 10].err.mean(), "all": o.err.mean()})
    for g in ["QB", "RB", "WR", "TE"]:
        row[f"{g} wk1-4"] = o[(o.pos == g) & (o.week <= 4)].err.mean()
    rows.append(row); print("done:", name, flush=True)
r = pd.DataFrame(rows).set_index("variant")
pd.set_option("display.width", 220)
print("\nMean abs error, PPR points excl. TDs (lower = better)\n", r.round(3).to_string())
print("\n% vs current (negative = better)\n", ((r / r.loc["current (no extra discount)"] - 1) * 100).round(2).to_string())
