"""Player projections: explosive plays and garbage-time usage, tested ON TOP of the success-rate layer (Oct 16).
For each stat: GLM(offset log proj, SR bundle) vs GLM(offset, SR bundle + candidate), walk-forward 2022-2026."""
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
        "two_point_attempt", "wp", "half_seconds_remaining", "game_seconds_remaining", "drive", "yards_gained", "rusher_player_id"]
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

# ---- candidates
e["x"] = (e.yards_gained >= 20).astype(float)
rr = e[e.play_type == "run"].dropna(subset=["rusher_player_id"]).groupby(["game_id", "rusher_player_id"])[["x", "n"]].sum().reset_index().rename(columns={"rusher_player_id": "pid"})
rc = e[e.play_type == "pass"].dropna(subset=["receiver_player_id"]).groupby(["game_id", "receiver_player_id"])[["x", "n"]].sum().reset_index().rename(columns={"receiver_player_id": "pid"})
qx = e[e.qb_dropback == 1].dropna(subset=["passer_player_id"]).groupby(["game_id", "passer_player_id"])[["x", "n"]].sum().reset_index().rename(columns={"passer_player_id": "pid"})
dp = e[e.play_type == "pass"].groupby(["game_id", "defteam"])[["x", "n"]].sum().reset_index(); dr = e[e.play_type == "run"].groupby(["game_id", "defteam"])[["x", "n"]].sum().reset_index()
C = {"rush_x": prior(rr, "pid", "x", "n", 60, rr.x.sum() / rr.n.sum()), "rec_x": prior(rc, "pid", "x", "n", 40, rc.x.sum() / rc.n.sum()),
     "qb_x": prior(qx, "pid", "x", "n", 150, qx.x.sum() / qx.n.sum()),
     "opp_px": prior(dp, "defteam", "x", "n", 150, dp.x.sum() / dp.n.sum(), decay=0.92), "opp_rx": prior(dr, "defteam", "x", "n", 120, dr.x.sum() / dr.n.sum(), decay=0.92)}
# garbage-time usage: share of a player's touches/targets that came with the game decided (wp outside 10-90%), vs his team's
e["gtm"] = (~e.wp.between(0.10, 0.90)).astype(float)
tch = pd.concat([e[e.play_type == "pass"].dropna(subset=["receiver_player_id"]).rename(columns={"receiver_player_id": "pid"})[["game_id", "pid", "gtm", "n"]],
                 e[e.play_type == "run"].dropna(subset=["rusher_player_id"]).rename(columns={"rusher_player_id": "pid"})[["game_id", "pid", "gtm", "n"]]])
tch = tch.groupby(["game_id", "pid"])[["gtm", "n"]].sum().reset_index()
C["garbage_share"] = prior(tch, "pid", "gtm", "n", 30, tch.gtm.sum() / tch.n.sum())
for nm, f in C.items():
    key = f.columns[1]
    if key == "pid": o = o.merge(f.rename(columns={"v": nm}), on=["game_id", "pid"], how="left")
    else: o = o.merge(f.rename(columns={"v": nm, "defteam": "opp"}), on=["game_id", "opp"], how="left")
for c_ in list(C): o[c_] = o[c_].fillna(o[c_].median())
BASE = {("QB", "pass_yds"): ["qb_sr", "pace", "opp_pace"], ("WR", "rec_yds"): ["tgt_sr", "qb_sr"], ("TE", "rec_yds"): ["tgt_sr", "qb_sr"],
        ("RB", "rec_yds"): ["tgt_sr", "qb_sr"], ("RB", "rush_yds"): ["pace", "opp_pace"], ("RB", "car"): ["pace", "opp_pace"],
        ("WR", "tgt"): ["tgt_sr", "pace"], ("TE", "tgt"): ["tgt_sr", "pace"]}
TESTS = [("Player explosive rate (rushing)", "rush_x", [("RB", "rush_yds")]),
         ("Player explosive rate (receiving)", "rec_x", [("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "rec_yds")]),
         ("QB explosive rate", "qb_x", [("QB", "pass_yds"), ("WR", "rec_yds")]),
         ("Opp explosive passes allowed", "opp_px", [("QB", "pass_yds"), ("WR", "rec_yds"), ("TE", "rec_yds")]),
         ("Opp explosive runs allowed", "opp_rx", [("RB", "rush_yds")]),
         ("Garbage-time share of touches", "garbage_share", [("RB", "car"), ("RB", "rush_yds"), ("WR", "tgt"), ("TE", "tgt"), ("WR", "rec_yds")])]
def run(pos, st, X):
    z = o[(o.pos == pos) & (o[f"p_{st}"] > 0.05)].copy(); res = []
    for s in range(2022, 2027):
        tr, te = z[z.season < s], z[z.season == s].copy()
        mu = tr[X].mean(); m = sm.GLM(tr[st], sm.add_constant(tr[X] - mu), family=sm.families.Poisson(), offset=np.log(tr[f"p_{st}"])).fit()
        te["adj"] = te[f"p_{st}"] * np.exp(((te[X] - mu) * m.params[X]).sum(axis=1)); res.append(te)
    t = pd.concat(res); return t, np.abs(t[st] - t.adj)
rows = []
for lab, c_, targets in TESTS:
    for pos, st in targets:
        t0, e0 = run(pos, st, BASE[(pos, st)]); t1, e1 = run(pos, st, BASE[(pos, st)] + [c_]); dlt = e0 - e1
        rows.append(dict(feature=lab, target=f"{pos} {st}", n=len(t1), mae_now=round(e0.mean(), 3), mae_with=round(e1.mean(), 3),
                         change_pct=round((e1.mean() / e0.mean() - 1) * 100, 2), se_pct=round(dlt.std() / np.sqrt(len(dlt)) / e0.mean() * 100, 2),
                         seasons_better=f"{int((dlt.groupby(t1.season).mean() > 0).sum())}/5"))
R = pd.DataFrame(rows); R.to_csv("tests/proj_expl_results.csv", index=False); pd.set_option("display.width", 220); print(R.to_string(index=False))
# ---- ceiling: with the same projection, do explosive players blow past it more often? (big games = actual >= 1.75x projection)
print("\nCEILING: share of games at >= 1.75x projection (and >= 2.5x), by explosive-rate third within projection band, 2019-2026")
o["band"] = None
for pos, st, c_ in [("WR", "rec_yds", "rec_x"), ("TE", "rec_yds", "rec_x"), ("RB", "rush_yds", "rush_x"), ("RB", "rec_yds", "rec_x"), ("QB", "pass_yds", "qb_x")]:
    z = o[(o.pos == pos) & (o[f"p_{st}"] > (150 if pos == "QB" else 20))].copy()
    z["pband"] = pd.qcut(z[f"p_{st}"], 4, labels=False, duplicates="drop")
    z["xt"] = z.groupby("pband")[c_].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    z["big"] = z[st] >= 1.75 * z[f"p_{st}"]; z["huge"] = z[st] >= 2.5 * z[f"p_{st}"]; z["bust"] = z[st] <= 0.4 * z[f"p_{st}"]
    t = z.groupby("xt").agg(n=("big", "size"), big=("big", "mean"), huge=("huge", "mean"), bust=("bust", "mean"), avg_ratio=(st, "sum"))
    t["avg_ratio"] = z.groupby("xt")[st].sum() / z.groupby("xt")[f"p_{st}"].sum()
    se = np.sqrt(t.big * (1 - t.big) / t.n)
    print(f"\n{pos} {st} (explosive measure: {c_})"); print((t.assign(big_se=se)).round(3).to_string())

# ---- ceiling test: given the projection, does the candidate predict a boom game (actual >= the 85th-percentile
#      actual/projection ratio for that position, set from earlier seasons)? Logistic, walk-forward; Brier vs position-only.
import statsmodels.api as sm2
CEIL = [("WR", "rec_yds", "rec_x"), ("TE", "rec_yds", "rec_x"), ("RB", "rush_yds", "rush_x"), ("RB", "rec_yds", "rec_x"), ("QB", "pass_yds", "qb_x"),
        ("WR", "rec_yds", "qb_x"), ("WR", "rec_yds", "opp_px"), ("RB", "rush_yds", "opp_rx")]
rows = []
for pos, st, c_ in CEIL:
    z = o[(o.pos == pos) & (o[f"p_{st}"] > (150 if pos == "QB" else 15))].copy(); res = []
    for s in range(2022, 2027):
        tr, te = z[z.season < s].copy(), z[z.season == s].copy()
        thr = (tr[st] / tr[f"p_{st}"]).quantile(0.85); tr["boom"] = (tr[st] >= thr * tr[f"p_{st}"]).astype(int); te["boom"] = (te[st] >= thr * te[f"p_{st}"]).astype(int)
        X = [c_]; mu, sd = tr[X].mean(), tr[X].std()
        m = sm2.GLM(tr.boom, sm2.add_constant((tr[X] - mu) / sd), family=sm2.families.Binomial()).fit()
        te["p1"] = m.predict(sm2.add_constant((te[X] - mu) / sd, has_constant="add")); te["p0"] = tr.boom.mean()
        te["top"] = te[c_] >= tr[c_].quantile(0.8); te["bot"] = te[c_] <= tr[c_].quantile(0.2); res.append(te)
    t = pd.concat(res); b0 = ((t.boom - t.p0) ** 2).mean(); b1 = ((t.boom - t.p1) ** 2).mean()
    rows.append(dict(target=f"{pos} {st}", feature=c_, n=len(t), boom_rate=round(t.boom.mean(), 3),
                     boom_top20pct=round(t[t.top].boom.mean(), 3), boom_bottom20pct=round(t[t.bot].boom.mean(), 3),
                     brier_gain_pct=round((1 - b1 / b0) * 100, 2),
                     seasons_better=f"{int(sum(((g.boom - g.p0) ** 2).mean() > ((g.boom - g.p1) ** 2).mean() for _, g in t.groupby('season')))}/5"))
R2 = pd.DataFrame(rows); R2.to_csv("tests/proj_ceiling_results.csv", index=False); print("\nCEILING"); print(R2.to_string(index=False))
