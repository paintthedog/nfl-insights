"""Bundle test (Oct 16): QB dropback success rate, receiver target success rate, neutral pace (team + opp), QB aggressiveness,
applied together on top of the current walk-forward projections. Each stat gets one Poisson GLM (offset = log projection) with
its bundle features, fit on earlier seasons only, scored on 2022-2026. Scores each stat and total non-TD fantasy points."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, glob, statsmodels.api as sm, warnings
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
o = pd.concat(outs); o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))].copy()
gd = pd.to_datetime(sched.set_index("game_id").gameday)
def prior(df, key, num, den, K, lg, decay=0.95):
    df = df.assign(gameday=df.game_id.map(gd)).sort_values([key, "gameday"]); out = []
    for k, x in df.groupby(key):
        a = b = 0.0
        for r in x.itertuples():
            out.append((r.game_id, k, (a + K * lg) / (b + K))); a = a * decay + getattr(r, num); b = b * decay + getattr(r, den)
    return pd.DataFrame(out, columns=["game_id", key, "v"])
cols = ["game_id", "play_id", "season_type", "posteam", "defteam", "play_type", "epa", "receiver_player_id", "passer_player_id", "qb_dropback",
        "two_point_attempt", "wp", "half_seconds_remaining", "game_seconds_remaining", "drive"]
pb = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
pb = pb[(pb.season_type == "REG") & pb.play_type.isin(["pass", "run"]) & (pb.two_point_attempt != 1) & pb.posteam.notna()].copy()
e = pb[pb.epa.notna()].copy(); e["s"] = (e.epa > 0).astype(float); e["n"] = 1.0
F = {}
c = e[e.play_type == "pass"].dropna(subset=["receiver_player_id"]).groupby(["game_id", "receiver_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"receiver_player_id": "pid"})
F["tgt_sr"] = prior(c, "pid", "s", "n", 40, c.s.sum() / c.n.sum())
q = e[e.qb_dropback == 1].dropna(subset=["passer_player_id"]).groupby(["game_id", "passer_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"passer_player_id": "pid"})
F["qb_sr"] = prior(q, "pid", "s", "n", 150, q.s.sum() / q.n.sum())
pb = pb.sort_values(["game_id", "drive", "game_seconds_remaining"], ascending=[True, True, False])
pb["dt"] = pb.groupby(["game_id", "drive"]).game_seconds_remaining.diff(-1)
nz = pb[pb.wp.between(.2, .8) & (pb.half_seconds_remaining > 120) & pb.dt.between(1, 50)]
pc = nz.groupby(["game_id", "posteam"]).agg(ds=("dt", "sum"), dn=("dt", "size")).reset_index()
F["pace"] = prior(pc, "posteam", "ds", "dn", 60, nz.dt.mean(), decay=0.9)
tg = pd.concat([sched[["season", "week", "game_id", "home_team"]].rename(columns={"home_team": "team"}), sched[["season", "week", "game_id", "away_team"]].rename(columns={"away_team": "team"})])
x = pd.read_parquet("data/ngs_passing.parquet"); x = x[x.week > 0].copy(); x["team"] = x.team_abbr.replace({"LAR": "LA"})
x = x.merge(tg, on=["season", "week", "team"]).rename(columns={"player_gsis_id": "pid"}); x["ag"] = x.aggressiveness.fillna(0) * x.attempts
F["aggr"] = prior(x, "pid", "ag", "attempts", 150, x.ag.sum() / x.attempts.sum())
for nm in ("tgt_sr", "aggr"): o = o.merge(F[nm].rename(columns={"v": nm}), on=["game_id", "pid"], how="left")
o = o.merge(F["pace"].rename(columns={"v": "pace"}), on=["game_id", "posteam"], how="left")
o = o.merge(F["pace"].rename(columns={"v": "opp_pace", "posteam": "opp"}), on=["game_id", "opp"], how="left")
# team's starting QB success rate for everyone on the offense (receivers' yards depend on their QB too)
qs = o[o.pos == "QB"][["game_id", "posteam", "pid"]].merge(F["qb_sr"].rename(columns={"v": "qb_sr"}), on=["game_id", "pid"], how="left")[["game_id", "posteam", "qb_sr"]].drop_duplicates(["game_id", "posteam"])
o = o.merge(qs, on=["game_id", "posteam"], how="left")
for c_ in ("tgt_sr", "aggr", "pace", "opp_pace", "qb_sr"): o[c_] = o[c_].fillna(o[c_].median())
B = {  # stat -> bundle features, by position
    ("QB", "att"): ["pace", "opp_pace"], ("QB", "cmp"): ["pace", "opp_pace", "qb_sr"], ("QB", "pass_yds"): ["qb_sr", "pace", "opp_pace"],
    ("QB", "ints"): ["aggr"], ("QB", "pass_1d"): ["qb_sr", "pace"],
    ("WR", "tgt"): ["tgt_sr", "pace"], ("WR", "rec"): ["tgt_sr", "pace", "qb_sr"], ("WR", "rec_yds"): ["tgt_sr", "qb_sr"], ("WR", "rec_1d"): ["tgt_sr", "qb_sr"],
    ("TE", "tgt"): ["tgt_sr", "pace"], ("TE", "rec"): ["tgt_sr", "pace", "qb_sr"], ("TE", "rec_yds"): ["tgt_sr", "qb_sr"], ("TE", "rec_1d"): ["tgt_sr", "qb_sr"],
    ("RB", "car"): ["pace", "opp_pace"], ("RB", "tgt"): ["tgt_sr", "pace"], ("RB", "rec"): ["tgt_sr", "pace"], ("RB", "rec_yds"): ["tgt_sr", "qb_sr"],
}
o = o[o.season >= 2022].copy() if False else o
for col in [c_ for (_, c_) in B]: o[f"b_{col}"] = o[f"p_{col}"]
z_all = []
for (pos, st), X in B.items():
    if f"p_{st}" not in o or st not in o: print("skip", pos, st); continue
    z = o[o.pos == pos]
    for s in range(2022, 2027):
        tr = z[(z.season < s) & (z[f"p_{st}"] > 0.05)]; te_i = z.index[(z.season == s)]
        mu = tr[X].mean(); m = sm.GLM(tr[st], sm.add_constant(tr[X] - mu), family=sm.families.Poisson(), offset=np.log(tr[f"p_{st}"])).fit()
        o.loc[te_i, f"b_{st}"] = o.loc[te_i, f"p_{st}"] * np.exp(((o.loc[te_i, X] - mu) * m.params[X]).sum(axis=1))
t = o[o.season >= 2022].copy()
def pts(df, pre):
    g = lambda c_: df[pre + c_] if pre + c_ in df else df["p_" + c_]
    return (g("rec") + g("rec_yds") / 10 + g("rec_1d") + g("rush_yds") / 10 + g("rush_1d") + np.where(df.pos == "QB", g("pass_yds") / 25 + 0.5 * g("pass_1d") - g("sacks"), 0))
actual = (t.rec + t.rec_yds / 10 + t.rec_1d + t.rush_yds / 10 + t.rush_1d + np.where(t.pos == "QB", t.pass_yds / 25 + 0.5 * t.pass_1d - t.sacks, 0))
rows = []
for pos in ["QB", "RB", "WR", "TE"]:
    m_ = t.pos == pos; a0 = np.abs(actual[m_] - pts(t[m_], "p_")); a1 = np.abs(actual[m_] - pts(t[m_], "b_"))
    dd = a0 - a1
    rows.append(dict(item=f"{pos} fantasy pts (no TDs)", n=int(m_.sum()), mae_now=a0.mean(), mae_bundle=a1.mean(), change_pct=(a1.mean() / a0.mean() - 1) * 100,
                     seasons_better=f"{int((dd.groupby(t[m_].season).mean() > 0).sum())}/5", se_pct=dd.std() / np.sqrt(len(dd)) / a0.mean() * 100))
for (pos, st) in B:
    m_ = t.pos == pos; a0 = np.abs(t[m_][st] - t[m_][f"p_{st}"]); a1 = np.abs(t[m_][st] - t[m_][f"b_{st}"]); dd = a0 - a1
    rows.append(dict(item=f"{pos} {st}", n=int(m_.sum()), mae_now=a0.mean(), mae_bundle=a1.mean(), change_pct=(a1.mean() / a0.mean() - 1) * 100,
                     seasons_better=f"{int((dd.groupby(t[m_].season).mean() > 0).sum())}/5", se_pct=dd.std() / np.sqrt(len(dd)) / a0.mean() * 100))
R = pd.DataFrame(rows).round(3); R.to_csv("tests/bundle_results.csv", index=False)
pd.set_option("display.width", 220); print(R.to_string(index=False))
