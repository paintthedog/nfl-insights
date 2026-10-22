"""Second-tier feature tests. Method: take the current model's walk-forward projection for each player-game, then ask
whether a candidate stat explains what the projection missed (Poisson GLM, offset = log projection), fitted only on
earlier seasons and scored on the next one (2022-2026)."""
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
    """shrunk, decayed rate from games BEFORE each game: (sum num + K*lg) / (sum den + K)"""
    df = df.assign(gameday=df.game_id.map(gd)).sort_values([key, "gameday"]); out = []
    for k, x in df.groupby(key):
        a = b = 0.0
        for r in x.itertuples():
            out.append((r.game_id, k, (a + K * lg) / (b + K))); a = a * decay + getattr(r, num); b = b * decay + getattr(r, den)
    return pd.DataFrame(out, columns=["game_id", key, "v"])

# ---- team game ids for NGS rows
tg = pd.concat([sched[["season", "week", "game_id", "home_team"]].rename(columns={"home_team": "team"}), sched[["season", "week", "game_id", "away_team"]].rename(columns={"away_team": "team"})])
def ngs(kind):
    x = pd.read_parquet(f"data/ngs_{kind}.parquet"); x = x[x.week > 0].copy(); x["team"] = x.team_abbr.replace({"LAR": "LA"})
    return x.merge(tg, on=["season", "week", "team"]).rename(columns={"player_gsis_id": "pid"})
F = {}
r = ngs("rushing"); r["ry"] = r.rush_yards_over_expected.fillna(0); r["b8"] = r.percent_attempts_gte_eight_defenders.fillna(0) / 100 * r.rush_attempts
F["ryoe"] = prior(r, "pid", "ry", "rush_attempts", 80, 0.0); F["box8"] = prior(r, "pid", "b8", "rush_attempts", 80, r.b8.sum() / r.rush_attempts.sum())
c = ngs("receiving"); c["sp"] = c.avg_separation * c.targets; c["cu"] = c.avg_cushion * c.targets; c["yx"] = c.avg_yac_above_expectation.fillna(0) * c.receptions
F["sep"] = prior(c, "pid", "sp", "targets", 40, c.sp.sum() / c.targets.sum()); F["cushion"] = prior(c, "pid", "cu", "targets", 40, c.cu.sum() / c.targets.sum())
F["yacoe"] = prior(c, "pid", "yx", "receptions", 40, 0.0)
q = ngs("passing"); q["tt"] = q.avg_time_to_throw * q.attempts; q["ag"] = q.aggressiveness.fillna(0) * q.attempts
F["ttt"] = prior(q, "pid", "tt", "attempts", 150, q.tt.sum() / q.attempts.sum()); F["aggr"] = prior(q, "pid", "ag", "attempts", 150, q.ag.sum() / q.attempts.sum())
# ---- from play-by-play: CPOE per QB, neutral-situation pace per team, play-action / screen / motion per offense (FTN 2022+)
cols = ["game_id", "play_id", "posteam", "defteam", "passer_player_id", "cpoe", "qb_dropback", "play_type", "wp", "half_seconds_remaining", "game_seconds_remaining", "drive", "two_point_attempt", "no_huddle"]
pb = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
pb = pb[pb.play_type.isin(["pass", "run"]) & (pb.two_point_attempt == 0) & pb.posteam.notna()].copy()
cp = pb.dropna(subset=["cpoe", "passer_player_id"]).groupby(["game_id", "passer_player_id"]).cpoe.agg(["sum", "count"]).reset_index().rename(columns={"passer_player_id": "pid", "sum": "cs", "count": "cn"})
F["cpoe"] = prior(cp, "pid", "cs", "cn", 200, 0.0)
pb = pb.sort_values(["game_id", "drive", "game_seconds_remaining"], ascending=[True, True, False])
pb["dt"] = pb.groupby(["game_id", "drive"]).game_seconds_remaining.diff(-1)          # seconds until the next snap of the same drive
nz = pb[pb.wp.between(.2, .8) & (pb.half_seconds_remaining > 120) & pb.dt.between(1, 50)]
pc = nz.groupby(["game_id", "posteam"]).agg(ds=("dt", "sum"), dn=("dt", "size"), nh=("no_huddle", "sum")).reset_index()
F["pace"] = prior(pc, "posteam", "ds", "dn", 60, nz.dt.mean(), decay=0.9); F["nohuddle"] = prior(pc, "posteam", "nh", "dn", 60, nz.no_huddle.mean(), decay=0.9)
ftn = pd.concat([pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_screen_pass", "is_motion", "is_rpo"]) for f in glob.glob("data/ftn_*.parquet")]).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
ftn["play_id"] = ftn.play_id.astype(float)
fp = pb.merge(ftn.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"])
dbk = fp[fp.qb_dropback == 1]
tf = dbk.groupby(["game_id", "posteam"]).agg(pa=("is_play_action", "sum"), sc=("is_screen_pass", "sum"), n=("is_play_action", "size")).reset_index()
tm = fp.groupby(["game_id", "posteam"]).agg(mo=("is_motion", "sum"), n=("is_motion", "size")).reset_index()
F["pa"] = prior(tf, "posteam", "pa", "n", 100, tf.pa.sum() / tf.n.sum(), decay=0.9); F["screen"] = prior(tf, "posteam", "sc", "n", 100, tf.sc.sum() / tf.n.sum(), decay=0.9)
F["motion"] = prior(tm, "posteam", "mo", "n", 150, tm.mo.sum() / tm.n.sum(), decay=0.9)
opp = d.drop_duplicates(["game_id", "posteam"]).set_index(["game_id", "posteam"]).opp
for nm, f in F.items():
    key = f.columns[1]
    o = o.merge(f.rename(columns={"v": nm, key: "pid" if key == "pid" else "posteam"}), on=["game_id", "pid" if key == "pid" else "posteam"], how="left")
o = o.merge(F["pace"].rename(columns={"v": "opp_pace", "posteam": "opp"}), on=["game_id", "opp"], how="left")
print("coverage 2022+:", {nm: round(o[o.season >= 2022][nm].notna().mean(), 2) for nm in list(F) + ["opp_pace"]})

TESTS = [  # (label, features, [(position, stat)], first season with data)
    ("Rush yards over expected (player)", ["ryoe"], [("RB", "rush_yds")], 2019),
    ("Stacked-box rate faced (player)", ["box8"], [("RB", "rush_yds")], 2019),
    ("RYOE + stacked box", ["ryoe", "box8"], [("RB", "rush_yds")], 2019),
    ("Neutral pace (team + opponent)", ["pace", "opp_pace"], [("QB", "att"), ("QB", "pass_yds"), ("WR", "tgt"), ("RB", "car")], 2019),
    ("No-huddle rate (team)", ["nohuddle"], [("QB", "att"), ("WR", "tgt"), ("RB", "car")], 2019),
    ("Play-action rate (team)", ["pa"], [("QB", "pass_yds"), ("WR", "rec_yds"), ("TE", "rec_yds")], 2023),
    ("Screen rate (team)", ["screen"], [("RB", "tgt"), ("RB", "rec_yds"), ("WR", "rec_yds")], 2023),
    ("Motion rate (team)", ["motion"], [("QB", "pass_yds"), ("WR", "rec_yds"), ("RB", "rush_yds")], 2023),
    ("Completion % over expected (QB)", ["cpoe"], [("QB", "cmp"), ("QB", "pass_yds")], 2019),
    ("Time to throw (QB)", ["ttt"], [("QB", "sacks"), ("QB", "pass_yds")], 2019),
    ("Aggressiveness (QB)", ["aggr"], [("QB", "ints"), ("QB", "pass_yds")], 2019),
    ("Separation (receiver)", ["sep"], [("WR", "rec_yds"), ("WR", "tgt"), ("TE", "rec_yds")], 2019),
    ("Cushion (receiver)", ["cushion"], [("WR", "rec_yds"), ("WR", "tgt")], 2019),
    ("YAC over expected (receiver)", ["yacoe"], [("WR", "rec_yds"), ("TE", "rec_yds"), ("RB", "rec_yds")], 2019),
]
rows = []
for lab, X, targets, first in TESTS:
    for pos, st in targets:
        z = o[(o.pos == pos) & (o[f"p_{st}"] > 0.05) & (o.season >= first)].dropna(subset=X).copy()
        res = []; 
        for s in range(max(2022, first + 1), 2027):
            tr, te = z[z.season < s], z[z.season == s].copy()
            if len(tr) < 300 or not len(te): continue
            mu = tr[X].mean(); m = sm.GLM(tr[st], sm.add_constant(tr[X] - mu), family=sm.families.Poisson(), offset=np.log(tr[f"p_{st}"])).fit()
            te["adj"] = te[f"p_{st}"] * np.exp(((te[X] - mu) * m.params[X]).sum(axis=1))     # feature effect only, no re-centering
            res.append(te)
        if not res: continue
        t = pd.concat(res); full = sm.GLM(z[st], sm.add_constant(z[X] - z[X].mean()), family=sm.families.Poisson(), offset=np.log(z[f"p_{st}"])).fit()
        e0, e1 = np.abs(t[st] - t[f"p_{st}"]).mean(), np.abs(t[st] - t.adj).mean()
        rows.append(dict(feature=lab, target=f"{pos} {st}", n=len(t), mae_now=round(e0, 3), mae_with=round(e1, 3), change_pct=round((e1 / e0 - 1) * 100, 2),
                         p=" / ".join(f"{full.pvalues[x]:.3f}" for x in X), effect=" / ".join(f"{np.exp(full.params[x] * z[x].std()) - 1:+.1%}" for x in X)))
R = pd.DataFrame(rows); R.to_csv("adv_test2_results.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 40)
print(R.to_string(index=False))
