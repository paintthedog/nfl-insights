"""Follow-ups on the two second-tier stats that showed anything: neutral pace inside the volume model, QB aggressiveness inside the INT model."""
import numpy as np, pandas as pd, glob, warnings
warnings.filterwarnings("ignore")
import proj_model as P, int_model as M
exec(open("adv_test2.py").read().split("# ---- team game ids for NGS rows")[0].split("outs = []")[0])      # data + d + prior()
gd = pd.to_datetime(sched.set_index("game_id").gameday)
def prior(df, key, num, den, K, lg, decay=0.95):
    df = df.assign(gameday=df.game_id.map(gd)).sort_values([key, "gameday"]); out = []
    for k, x in df.groupby(key):
        a = b = 0.0
        for r in x.itertuples():
            out.append((r.game_id, k, (a + K * lg) / (b + K))); a = a * decay + getattr(r, num); b = b * decay + getattr(r, den)
    return pd.DataFrame(out, columns=["game_id", key, "v"])
cols = ["game_id", "posteam", "play_type", "wp", "half_seconds_remaining", "game_seconds_remaining", "drive", "two_point_attempt"]
pb = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
pb = pb[pb.play_type.isin(["pass", "run"]) & (pb.two_point_attempt == 0) & pb.posteam.notna()].sort_values(["game_id", "drive", "game_seconds_remaining"], ascending=[True, True, False])
pb["dt"] = pb.groupby(["game_id", "drive"]).game_seconds_remaining.diff(-1)
nz = pb[pb.wp.between(.2, .8) & (pb.half_seconds_remaining > 120) & pb.dt.between(1, 50)]
pc = nz.groupby(["game_id", "posteam"]).agg(ds=("dt", "sum"), dn=("dt", "size")).reset_index()
pace = prior(pc, "posteam", "ds", "dn", 60, nz.dt.mean(), decay=0.9)
d = d.merge(pace.rename(columns={"v": "pace"}), on=["game_id", "posteam"], how="left").merge(pace.rename(columns={"v": "opp_pace", "posteam": "opp"}), on=["game_id", "opp"], how="left")
base = list(P.VOL_X); res = []
for nm, vx in [("current", base), ("+ team pace", base + ["pace"]), ("+ team and opponent pace", base + ["pace", "opp_pace"])]:
    P.VOL_X = vx; outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); outs.append(P.apply_calibration(te, cal))
    o = pd.concat(outs); t = o.drop_duplicates(["game_id", "posteam"]); t = t[t.implied.notna()]
    o = o[(o.pr_n >= 1) & o.implied.notna() & (((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"]))]
    row = dict(variant=nm, team_att=np.abs(t.tm_att - t.v_att).mean(), team_car=np.abs(t.tm_car - t.v_car).mean())
    for g, st in [("QB", "att"), ("QB", "pass_yds"), ("WR", "tgt"), ("WR", "rec_yds"), ("TE", "tgt"), ("RB", "car"), ("RB", "rush_yds")]:
        z = o[o.pos == g]; row[f"{g} {st}"] = np.abs(z[st] - z[f"p_{st}"]).mean()
    res.append(row); print("done", nm, flush=True)
R = pd.DataFrame(res).set_index("variant"); pd.set_option("display.width", 220)
print(R.round(3).to_string()); print(((R / R.loc["current"] - 1) * 100).round(2).iloc[1:].to_string())
P.VOL_X = base
# ---- QB aggressiveness (share of throws into tight coverage, Next Gen Stats) in the interception model
ipbp, isched = M.load(); qb, tgm, ctx = M.build_tables(ipbp, isched)
df = M.prep(M.build_features(qb, tgm, ctx)); df = df[(df.prior_games >= 1) & df.att.notna() & (df.att >= 10)].copy()
tg = pd.concat([sched[["season", "week", "game_id", "home_team"]].rename(columns={"home_team": "team"}), sched[["season", "week", "game_id", "away_team"]].rename(columns={"away_team": "team"})])
q = pd.read_parquet("data/ngs_passing.parquet"); q = q[q.week > 0].copy(); q["team"] = q.team_abbr.replace({"LAR": "LA"})
q = q.merge(tg, on=["season", "week", "team"]).rename(columns={"player_gsis_id": "pid"}); q["ag"] = q.aggressiveness.fillna(0) * q.attempts
lg = q.ag.sum() / q.attempts.sum()
a = prior(q, "pid", "ag", "attempts", 150, lg).rename(columns={"v": "qb_aggr", "pid": "passer_player_id"})
df = df.merge(a, on=["game_id", "passer_player_id"], how="left"); df["qb_aggr"] = df.qb_aggr.fillna(lg)
b0 = list(M.INT_X)
for nm, X in [("current", b0), ("+ QB aggressiveness", b0 + ["qb_aggr"])]:
    M.INT_X = X; ps = []
    for s in range(2022, 2027):
        tr, te = df[df.season < s], df[df.season == s].dropna(subset=M.ATT_X + [c for c in X if c != "log_proj_att"])
        am = M.fit_attempts(tr); im = M.fit_ints(tr, am); ps.append(M.predict(te, am, im)[["season", "ints", "p_int"]])
    o = pd.concat(ps); y = (o.ints >= 1).astype(int); bb = np.mean((y.mean() - y) ** 2)
    print(f"INT model {nm:22s} n={len(o)} brier {np.mean((o.p_int - y) ** 2):.5f} skill {1 - np.mean((o.p_int - y) ** 2) / bb:.4f}", {k: (round(float(im.params[k]), 3), round(float(im.pvalues[k]), 3)) for k in X if k not in b0})
