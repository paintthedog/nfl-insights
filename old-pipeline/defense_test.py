"""Walk-forward test: how much should projections lean on opponent yards allowed?"""
import numpy as np, pandas as pd
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
V = {
    "no opponent adjustment":                    dict(USE_DEF=False),
    "current (heavy averaging, short memory)":   dict(),
    "less averaging":                            dict(DEF_K_PASS=100, DEF_K_RUSH=70),
    "much less averaging":                       dict(DEF_K_PASS=30, DEF_K_RUSH=20),
    "longer memory":                             dict(DEF_DECAY=0.96),
    "longer memory + less averaging":            dict(DEF_DECAY=0.96, DEF_K_PASS=100, DEF_K_RUSH=70),
    "position-specific receiving":               dict(POS_DEF=True),
    "position-specific + longer memory":         dict(POS_DEF=True, DEF_DECAY=0.96),
}
DEF = dict(USE_DEF=True, DEF_K_PASS=300, DEF_K_RUSH=200, DEF_DECAY=0.90, POS_DEF=False, POS_DEF_K=60)
pts = lambda o, p: (o[p + "rec"] + o[p + "rec_yds"] / 10 + o[p + "rec_1d"] + o[p + "rush_yds"] / 10 + o[p + "rush_1d"]
                    + np.where(o.pos == "QB", o[p + "pass_yds"] / 25 + 0.5 * o[p + "pass_1d"] - o[p + "sacks"], 0))
res = []
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
    o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))]
    row = dict(variant=name)
    for g in ["QB", "RB", "WR", "TE"]:
        r = o[o.pos == g]; row[g] = np.mean(np.abs(pts(r, "") - pts(r, "p_")))
    late = o[o.week >= 8]
    row["all, weeks 8+"] = np.mean(np.abs(pts(late, "") - pts(late, "p_")))
    for g, st in [("QB", "pass_yds"), ("RB", "rush_yds"), ("WR", "rec_yds"), ("TE", "rec_yds")]:
        r = o[o.pos == g]; row[f"{g} {st}"] = np.mean(np.abs(r[st] - r[f"p_{st}"]))
    res.append(row); print("done:", name, flush=True)
r = pd.DataFrame(res).set_index("variant")
pd.set_option("display.width", 220)
print("\nMean absolute error, Legends points excl. TDs (lower = better)\n", r.round(3).to_string())
print("\n% vs current (negative = better)\n", ((r / r.loc["current (heavy averaging, short memory)"] - 1) * 100).round(2).to_string())
r.to_csv("defense_test_results.csv")
