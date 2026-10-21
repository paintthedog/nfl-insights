"""Success rate (EPA > 0) for player projections. Same method as adv_test2: take the current model's walk-forward
projection, ask whether a prior-games success-rate stat explains what it missed (Poisson GLM, offset = log projection),
fit on earlier seasons only, score on the next (2022-2026)."""
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
cols = ["game_id", "season_type", "posteam", "defteam", "play_type", "epa", "rusher_player_id", "receiver_player_id", "passer_player_id", "qb_dropback", "two_point_attempt"]
pb = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
pb = pb[(pb.season_type == "REG") & pb.play_type.isin(["pass", "run"]) & (pb.two_point_attempt != 1) & pb.posteam.notna() & pb.epa.notna()].copy()
pb["s"] = (pb.epa > 0).astype(float); pb["n"] = 1.0; pb["isp"] = (pb.play_type == "pass").astype(float)
F = {}
r = pb[pb.play_type == "run"].dropna(subset=["rusher_player_id"]).groupby(["game_id", "rusher_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"rusher_player_id": "pid"})
F["rush_sr"] = prior(r, "pid", "s", "n", 60, r.s.sum() / r.n.sum())
c = pb[pb.play_type == "pass"].dropna(subset=["receiver_player_id"]).groupby(["game_id", "receiver_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"receiver_player_id": "pid"})
F["tgt_sr"] = prior(c, "pid", "s", "n", 40, c.s.sum() / c.n.sum())
q = pb[pb.qb_dropback == 1].dropna(subset=["passer_player_id"]).groupby(["game_id", "passer_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"passer_player_id": "pid"})
F["qb_sr"] = prior(q, "pid", "s", "n", 150, q.s.sum() / q.n.sum())
t = pb.groupby(["game_id", "posteam"])[["s", "n"]].sum().reset_index()
F["team_sr"] = prior(t, "posteam", "s", "n", 150, t.s.sum() / t.n.sum(), decay=0.92)
dfn = pb.groupby(["game_id", "defteam"])[["s", "n"]].sum().reset_index()
dP = pb[pb.isp == 1].groupby(["game_id", "defteam"])[["s", "n"]].sum().reset_index(); dR = pb[pb.isp == 0].groupby(["game_id", "defteam"])[["s", "n"]].sum().reset_index()
F["opp_def_sr"] = prior(dfn, "defteam", "s", "n", 150, dfn.s.sum() / dfn.n.sum(), decay=0.92)
F["opp_dpass_sr"] = prior(dP, "defteam", "s", "n", 100, dP.s.sum() / dP.n.sum(), decay=0.92)
F["opp_drush_sr"] = prior(dR, "defteam", "s", "n", 80, dR.s.sum() / dR.n.sum(), decay=0.92)
for nm, f in F.items():
    key = f.columns[1]
    if key == "pid": o = o.merge(f.rename(columns={"v": nm}), on=["game_id", "pid"], how="left")
    elif key == "posteam": o = o.merge(f.rename(columns={"v": nm}), on=["game_id", "posteam"], how="left")
    else: o = o.merge(f.rename(columns={"v": nm, "defteam": "opp"}), on=["game_id", "opp"], how="left")
TESTS = [
    ("Rushing success rate (player)", ["rush_sr"], [("RB", "rush_yds"), ("RB", "car")]),
    ("Target success rate (player)", ["tgt_sr"], [("WR", "rec_yds"), ("WR", "tgt"), ("TE", "rec_yds"), ("RB", "rec_yds")]),
    ("Dropback success rate (QB)", ["qb_sr"], [("QB", "pass_yds"), ("QB", "att")]),
    ("Team offensive success rate", ["team_sr"], [("QB", "att"), ("QB", "pass_yds"), ("WR", "tgt"), ("RB", "car"), ("RB", "rush_yds")]),
    ("Opponent success rate allowed", ["opp_def_sr"], [("QB", "pass_yds"), ("RB", "rush_yds"), ("WR", "rec_yds")]),
    ("Opp pass SR allowed", ["opp_dpass_sr"], [("QB", "pass_yds"), ("WR", "rec_yds"), ("TE", "rec_yds")]),
    ("Opp rush SR allowed", ["opp_drush_sr"], [("RB", "rush_yds"), ("RB", "car")]),
]
rows = []
for lab, X, targets in TESTS:
    for pos, st in targets:
        z = o[(o.pos == pos) & (o[f"p_{st}"] > 0.05)].dropna(subset=X).copy(); res = []
        for s in range(2022, 2027):
            tr, te = z[z.season < s], z[z.season == s].copy()
            mu = tr[X].mean(); m = sm.GLM(tr[st], sm.add_constant(tr[X] - mu), family=sm.families.Poisson(), offset=np.log(tr[f"p_{st}"])).fit()
            te["adj"] = te[f"p_{st}"] * np.exp(((te[X] - mu) * m.params[X]).sum(axis=1)); res.append(te)
        t = pd.concat(res); full = sm.GLM(z[st], sm.add_constant(z[X] - z[X].mean()), family=sm.families.Poisson(), offset=np.log(z[f"p_{st}"])).fit()
        e0, e1 = np.abs(t[st] - t[f"p_{st}"]).mean(), np.abs(t[st] - t.adj).mean()
        rows.append(dict(feature=lab, target=f"{pos} {st}", n=len(t), mae_now=round(e0, 3), mae_with=round(e1, 3), change_pct=round((e1 / e0 - 1) * 100, 2),
                         p=" / ".join(f"{full.pvalues[x]:.3f}" for x in X), effect=" / ".join(f"{np.exp(full.params[x] * z[x].std()) - 1:+.1%}" for x in X)))
R = pd.DataFrame(rows); R.to_csv("tests/success_player_results.csv", index=False)
pd.set_option("display.width", 220); print(R.to_string(index=False))
