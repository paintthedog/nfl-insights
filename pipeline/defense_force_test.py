"""Force the opponent effect bigger than the data says, and measure what happens."""
import numpy as np, pandas as pd
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
P.DEF_K_PASS, P.DEF_K_RUSH = 100, 70     # let defenses spread out, then scale the learned effect
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
pts = lambda o, p: (o[p + "rec"] + o[p + "rec_yds"] / 10 + o[p + "rec_1d"] + o[p + "rush_yds"] / 10 + o[p + "rush_1d"]
                    + np.where(o.pos == "QB", o[p + "pass_yds"] / 25 + 0.5 * o[p + "pass_1d"] - o[p + "sacks"], 0))
res = []
for mult in [0, 1, 1.5, 2, 3]:
    outs, eff = [], []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"]))
        tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1])
        for m in cal.values():
            if "ldef" in m.params.index:
                p = m.params.copy(); p.loc["ldef"] = p.loc["ldef"] * mult; m._results.params = p.to_numpy()
        o = P.apply_calibration(te, cal)
        outs.append(o)
    o = pd.concat(outs)
    o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))]
    row = dict(opponent_effect=f"{mult}x what the data says")
    for g in ["QB", "RB", "WR", "TE"]:
        r = o[o.pos == g]; row[g] = np.mean(np.abs(pts(r, "") - pts(r, "p_")))
    # how much the adjustment moves WR yards between the softest and toughest pass defense (10th-90th pct)
    lo, hi = np.quantile(o.def_pass_ratio, [0.1, 0.9])
    wr = cal[("rec_yds", "WR")]._results.params[list(cal[("rec_yds", "WR")].params.index).index("ldef")]
    row["WR yds: soft vs tough D"] = f"{(hi / lo) ** wr * 100 - 100:+.0f}%"
    res.append(row)
r = pd.DataFrame(res).set_index("opponent_effect")
base = r.loc["1x what the data says", ["QB", "RB", "WR", "TE"]]
print(r.round(3).to_string())
print("\n% vs data-chosen (negative = better)\n", ((r[["QB", "RB", "WR", "TE"]] / base - 1) * 100).round(2).to_string())
