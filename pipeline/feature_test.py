"""Walk-forward test of new projection features: snap share, air-yards share, WOPR, pass rate over expected."""
import numpy as np, pandas as pd
import proj_model as P

pbp = P.load()
sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players)
pm = P.position_means(d)
d = P.efficiency(d, pm)
print("snap coverage on skill rows:", round(d[d.pos.isin(["RB", "WR", "TE"])].snap.notna().mean(), 3),
      "| PROE coverage:", round(d.pg_proe.ne(0).mean(), 3))

REC = ["tgt", "rec", "rec_yds", "rec_1d"]; RUSH = ["car", "rush_yds", "rush_1d"]
SNAP = ["l_snap", "snap_trend"]
variants = {
    "current": ({}, False),
    "+ PROE": ({}, True),
    "+ snap share": ({**{s: SNAP for s in REC}, **{s: SNAP for s in RUSH}}, False),
    "+ air-yards share": ({s: ["l_ay_sh"] for s in REC}, False),
    "+ WOPR": ({s: ["l_wopr"] for s in REC}, False),
    "+ snap + air share": ({**{s: SNAP + ["l_ay_sh"] for s in REC}, **{s: SNAP for s in RUSH}}, False),
    "+ snap + WOPR": ({**{s: SNAP + ["l_wopr"] for s in REC}, **{s: SNAP for s in RUSH}}, False),
    "+ snap + air share + PROE": ({**{s: SNAP + ["l_ay_sh"] for s in REC}, **{s: SNAP for s in RUSH}}, True),
}
nontd = lambda df, pre: (df[pre + "rec"] + df[pre + "rec_yds"] / 10 + df[pre + "rec_1d"] + df[pre + "rush_yds"] / 10
                         + df[pre + "rush_1d"] + np.where(df.pos == "QB", df[pre + "pass_yds"] / 25 + 0.5 * df[pre + "pass_1d"] - df[pre + "sacks"], 0))
res = []
for name, (extra, proe) in variants.items():
    P.EXTRA, P.USE_PROE = extra, proe
    outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1])
        outs.append(P.apply_calibration(te, cal))
    o = pd.concat(outs)
    o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))]
    row = dict(variant=name)
    for grp in ["QB", "RB", "WR", "TE"]:
        r = o[o.pos == grp]
        row[f"{grp} pts"] = np.mean(np.abs(nontd(r, "") - nontd(r, "p_")))
    for grp, st in [("WR", "tgt"), ("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "car"), ("RB", "rush_yds"), ("QB", "att"), ("QB", "pass_yds")]:
        r = o[o.pos == grp]
        row[f"{grp} {st}"] = np.mean(np.abs(r[st] - r[f"p_{st}"]))
    res.append(row)
    print("done", name, flush=True)
r = pd.DataFrame(res).set_index("variant")
pd.set_option("display.width", 220)
print("\nMean absolute error (lower is better); 'pts' = Legends points excluding TDs/INTs")
print(r.round(3).to_string())
print("\n% change vs current (negative = better)")
print(((r / r.loc["current"] - 1) * 100).round(2).to_string())
r.to_csv("feature_test_results.csv")
