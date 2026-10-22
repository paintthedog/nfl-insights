"""Small-parlay study (Oct 16): which pairs of LIKELY legs hit together more often than independence predicts?
Each leg is defined relative to the model's walk-forward projection so a single leg hits ~78% (about a 1.25-1.30 alt-line price).
Roles (QB, WR1, WR2, TE1, RB1) come from PROJECTIONS, so they are knowable before kickoff. 2019-2026 regular season."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
outs = []
for s in range(2019, 2027):
    tr, te = d[d.season < s].copy(), d[d.season == s].copy()
    va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[tr.pr_n >= 1]); outs.append(P.apply_calibration(te, cal))
o = pd.concat(outs); o = o[o.pr_n >= 1]; o.to_pickle("tests/_wf_proj.pkl")
TARGET = 0.78
rank = lambda df, pos, col, k: df[df.pos.isin(pos)].sort_values(col, ascending=False).groupby(["game_id", "posteam"]).nth(k - 1).reset_index(drop=False) if False else \
    df[df.pos.isin(pos)].sort_values(col, ascending=False).groupby(["game_id", "posteam"]).head(k).groupby(["game_id", "posteam"]).tail(1)
qb = o[(o.pos == "QB") & o.qb_start.astype(bool)].drop_duplicates(["game_id", "posteam"])
roles = {"QB": qb, "WR1": rank(o, ["WR"], "p_tgt", 1), "WR2": rank(o, ["WR"], "p_tgt", 2), "TE1": rank(o, ["TE"], "p_tgt", 1), "RB1": rank(o, ["RB"], "p_car", 1)}
def leg(df, stat, name):
    r = np.quantile(df[stat] / df[f"p_{stat}"].clip(lower=0.1), 1 - TARGET)
    x = df[["game_id", "posteam"]].copy(); x[name] = (df[stat] >= r * df[f"p_{stat}"]).astype(int).values
    return x, r
L = {}
for nm, rl, st in (("QB pass yds", "QB", "pass_yds"), ("WR1 rec", "WR1", "rec"), ("WR1 rec yds", "WR1", "rec_yds"), ("WR2 rec", "WR2", "rec"),
                   ("TE1 rec", "TE1", "rec"), ("RB1 rush yds", "RB1", "rush_yds"), ("RB1 rec", "RB1", "rec")):
    L[nm], r = leg(roles[rl], st, nm); print(f"{nm:13s} = clears {r:.2f}x his projection -> hits {L[nm][nm].mean():.1%} alone", flush=True)
T = None
for nm, x in L.items(): T = x if T is None else T.merge(x, on=["game_id", "posteam"], how="inner")
s = sched[["game_id", "home_team", "away_team", "result", "spread_line"]]
T = T.merge(s, on="game_id"); T["home"] = (T.posteam == T.home_team).astype(int)
T["fav_by"] = np.where(T.home == 1, T.spread_line, -T.spread_line)
T["win"] = np.where(T.home == 1, T.result > 0, T.result < 0).astype(int)
T["Team wins (fav 3+)"] = np.where(T.fav_by >= 3, T.win, np.nan)
T["opp"] = np.where(T.home == 1, T.away_team, T.home_team)
OP = ["QB pass yds", "WR1 rec", "RB1 rush yds"]
opp = T[["game_id", "posteam"] + OP].rename(columns={**{c: "OPP " + c for c in OP}, "posteam": "opp"})
T = T.merge(opp, on=["game_id", "opp"], how="left")
cols = list(L) + ["Team wins (fav 3+)"] + ["OPP " + c for c in OP]
rows = []
for a, b in itertools.combinations(cols, 2):
    if a.startswith("OPP") and b.startswith("OPP"): continue
    if {a, b} == {"WR1 rec", "WR1 rec yds"}: continue
    x = T[[a, b]].dropna()
    pa, pb, pj = x[a].mean(), x[b].mean(), (x[a] * x[b]).mean()
    rows.append(dict(leg_a=a, leg_b=b, n=len(x), p_a=pa, p_b=pb, if_independent=pa * pb, together=pj, lift=pj / (pa * pb)))
R = pd.DataFrame(rows).sort_values("lift", ascending=False)
pd.set_option("display.width", 220); print(R.round(3).to_string(index=False)); R.to_csv("tests/combo_lift.csv", index=False)
w = T[["game_id", "WR1 rec"]].copy(); w["wk"] = w.game_id.str[:7]; m = w.merge(w, on="wk"); m = m[m.game_id_x < m.game_id_y].sample(20000, random_state=1)
print("\ncross-game baseline (two WR1 rec legs, different games): together", round((m["WR1 rec_x"] * m["WR1 rec_y"]).mean(), 3),
      "| if independent", round(m["WR1 rec_x"].mean() * m["WR1 rec_y"].mean(), 3))
T.to_pickle("tests/_combo_T.pkl")
