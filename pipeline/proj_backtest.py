"""Walk-forward backtest of stat projections (2021 - 2026 Wk3) + outcome-range calibration."""
import pickle
import numpy as np, pandas as pd
import proj_model as P, td_model as T

pbp = P.load()
sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])

d = P.build(pbp, sched, players)
pm = P.position_means(d)
d = P.efficiency(d, pm)

# ---- touchdown means from the TD model, walk-forward (same seasons)
tpbp, tsched, tplayers = T.load()
tpg, tteam, tctx = T.build_tables(tpbp, tsched, tplayers)
tf = T.features(tpg, tteam, tctx)
tf = tf[(tf.pr_n >= 1) & tf.implied.notna() & tf.rush_share.notna()]
tds = []
for s in range(2021, 2027):
    tmm = T.fit_team(tteam[tteam.season < s]); mr, mp = T.fit_players(tf[tf.season < s], tmm)
    tds.append(T.predict(tf[tf.season == s], tmm, mr, mp)[["game_id", "pid", "lam_rush", "lam_rec", "exp_td", "rush_share"]])
tds = pd.concat(tds)
d = d.merge(tds, on=["game_id", "pid"], how="left")

# ---- baselines: season-to-date average and last-3 average of each stat (prior games only)
d = d.sort_values(["pid", "gameday"])
BASE_STATS = ["tgt", "rec", "rec_yds", "rec_1d", "car", "rush_yds", "rush_1d", "att", "cmp", "pass_yds", "sacks", "pass_1d",
              "ints", "pass_td", "rush_td", "rec_td"]
g = d.groupby(["pid", "season"])
for st in BASE_STATS:
    d[f"b_std_{st}"] = g[st].transform(lambda x: x.shift().expanding().mean())
    d[f"b_l3_{st}"] = d.groupby("pid")[st].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    prev = d.groupby(["pid", "season"])[st].mean().rename("prev").reset_index()
    prev["season"] += 1
    d = d.merge(prev.rename(columns={"prev": f"b_prev_{st}"}), on=["pid", "season"], how="left")
    d[f"b_std_{st}"] = d[f"b_std_{st}"].fillna(d[f"b_prev_{st}"])
    d = d.drop(columns=[f"b_prev_{st}"])

# ---- walk-forward
out = []
for s in range(2021, 2027):
    tr = d[(d.season < s)].copy(); te = d[d.season == s].copy()
    teams = tr.drop_duplicates(["game_id", "posteam"])
    va, vc = P.fit_volume(teams)
    tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[(tr.pr_n >= 1)])
    te = P.apply_calibration(te, cal)
    out.append(te)
o = pd.concat(out)
o = o[(o.pr_n >= 1) & o.implied.notna()]
o = o[((o.pos == "QB") & o.qb_start) | o.pos.isin(["RB", "WR", "TE"])]

# TDs + INTs
o["p_rush_td"] = o.lam_rush.fillna(0); o["p_rec_td"] = o.lam_rec.fillna(0)
o["p_pass_td"] = np.where(o.pos == "QB", o.exp_td.fillna(2.3) * (1 - o.rush_share.fillna(0.4)) * 0.97, 0)
for f, v in P.fantasy(o, "p_").items():
    o[f"p_fp_{f}"] = v
act = o.rename(columns={c: f"a_{c}" for c in []})
for f, v in P.fantasy(o.assign(**{f"a_{c}": o[c] for c in BASE_STATS}), "a_").items():
    o[f"a_fp_{f}"] = v
for kind in ["std", "l3"]:
    for f, v in P.fantasy(o.assign(**{f"{kind}_{c}": o[f"b_{kind}_{c}"] for c in BASE_STATS}), f"{kind}_").items():
        o[f"b_{kind}_fp_{f}"] = v

# ---- MAE report
rows = []
def mae(a, b): return np.mean(np.abs(a - b))
for grp, stats in [("QB", ["att", "cmp", "pass_yds", "pass_1d", "rush_yds", "fp_legends", "fp_std"]),
                   ("RB", ["car", "rush_yds", "tgt", "rec", "rec_yds", "fp_legends", "fp_ppr", "fp_std"]),
                   ("WR", ["tgt", "rec", "rec_yds", "rec_1d", "fp_legends", "fp_ppr", "fp_std"]),
                   ("TE", ["tgt", "rec", "rec_yds", "fp_legends", "fp_ppr", "fp_std"])]:
    r = o[o.pos == grp]
    for st in stats:
        a = r[f"a_{st}"] if st.startswith("fp_") else r[st]
        bs = r[f"b_std_{st}"].fillna(r[f"p_{st}"]); bl = r[f"b_l3_{st}"].fillna(r[f"p_{st}"])
        rows.append(dict(pos=grp, stat=st, n=len(r), model=mae(a, r[f"p_{st}"]), season_avg=mae(a, bs), last3=mae(a, bl)))
rep = pd.DataFrame(rows)
rep["vs_season_avg"] = 1 - rep.model / rep.season_avg
rep["vs_last3"] = 1 - rep.model / rep.last3
pd.set_option("display.width", 200)
print(rep.round(3).to_string(index=False))

# ---- outcome ranges: build ratio tables from 2021-22 residuals, test P(over) calibration on 2023+
build_rows = o[o.season <= 2023]; test_rows = o[o.season >= 2024].copy()
tab = P.ratio_table(build_rows)
checks = []
for st in ["rec_yds", "rush_yds", "rec", "pass_yds", "fp_legends"]:
    for grp in ["QB", "RB", "WR", "TE"]:
        r = test_rows[test_rows.pos == grp]
        if r.empty or not any(k[0] == st and k[1] == grp for k in tab):
            continue
        base = r[f"b_std_{st}"] if f"b_std_{st}" in r else None
        if base is None:
            continue
        line = (np.floor(base * 2) / 2 + 0.5).where(base.notna())   # stand-in for a book line
        m = line.notna() & (r[f"p_{st}"] > 0.5)
        r, line = r[m], line[m]
        act_v = r[f"a_{st}"] if st.startswith("fp_") else r[st]
        pov = np.array([P.p_over(tab, st, grp, mu, L) for mu, L in zip(r[f"p_{st}"], line)])
        checks.append(pd.DataFrame(dict(stat=st, pos=grp, pred=pov, hit=(act_v.to_numpy() > line.to_numpy()).astype(int))))
c = pd.concat(checks).dropna()
c["band"] = pd.cut(c.pred, [0, .3, .4, .5, .6, .7, 1])
print("\nP(over) calibration on 2024-26, raw (lines = season-to-date average + 0.5):")
print(c.groupby("band", observed=True).agg(n=("hit", "size"), pred=("pred", "mean"), actual=("hit", "mean")).round(3))
print("\nby stat:", c.groupby("stat").apply(lambda x: pd.Series(dict(pred=x.pred.mean(), actual=x.hit.mean()))).round(3).to_dict("index"))

# ---- recalibration map for P(over): fit on half the test games, check on the other half
rng = np.random.default_rng(0)
c["half"] = rng.integers(0, 2, len(c))
import statsmodels.api as sm
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
def fit_recal(df):
    """Smooth monotone recalibration: logit(p_adj) = a + b * logit(p)  (Platt scaling)."""
    m = sm.GLM(df.hit, sm.add_constant(lg(df.pred.to_numpy())), family=sm.families.Binomial()).fit()
    a, b = m.params
    return float(a), float(max(b, 0.05))
def apply_recal(p, ab):
    a, b = ab; z = a + b * lg(np.asarray(p)); return 1 / (1 + np.exp(-z))
maps = {st: fit_recal(g[g.half == 0]) for st, g in c.groupby("stat")}
chk = c[c.half == 1].copy()
chk["adj"] = [float(apply_recal(p, maps[st])) for p, st in zip(chk.pred, chk.stat)]
chk["band"] = pd.cut(chk.adj, [0, .3, .4, .5, .6, .7, 1])
print("\nAfter recalibration (held-out half):")
print(chk.groupby("band", observed=True).agg(n=("hit", "size"), pred=("adj", "mean"), actual=("hit", "mean")).round(3))
print("by stat:", chk.groupby("stat").apply(lambda x: pd.Series(dict(pred=x.adj.mean(), actual=x.hit.mean()))).round(3).to_dict("index"))
recal = {st: fit_recal(g) for st, g in c.groupby("stat")}   # production: all test games
recal["_pooled"] = fit_recal(c)
print("recal (a, b):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in recal.items()})
qb = o[(o.pos == "QB")]
print("QB pass_yds mean actual vs projected:", round(qb.pass_yds.mean(), 1), round(qb.p_pass_yds.mean(), 1))
pickle.dump(dict(tab=P.ratio_table(o), recal=recal, report=rep),
            open("ratio_table.pkl", "wb"))
