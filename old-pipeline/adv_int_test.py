"""Test 2: do blitz / true-pressure / man-coverage tendencies improve the interception model? Walk-forward 2023-2026."""
import numpy as np, pandas as pd, glob
import int_model as M
pbp, sched = M.load(); qb, tg, ctx = M.build_tables(pbp, sched)
df = M.prep(M.build_features(qb, tg, ctx)); df = df[(df.prior_games >= 1) & df.att.notna() & (df.att >= 10)].copy()
# per-play tags
pb = pd.concat([pd.read_parquet(f, columns=["game_id", "play_id", "posteam", "defteam", "qb_dropback", "two_point_attempt", "season"]) for f in sorted(glob.glob("data/pbp_20*.parquet")) if int(f[-12:-8]) >= 2019])
pb = pb[(pb.qb_dropback == 1) & (pb.two_point_attempt == 0) & pb.posteam.notna()]
ftn = pd.concat([pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers"]) for f in glob.glob("data/ftn_*.parquet")]).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
ftn["play_id"] = ftn.play_id.astype(float)
cov = pd.concat([pd.read_parquet(f) for f in glob.glob("data/part_cov_*.parquet")])[["game_id", "play_id", "defense_man_zone_type", "was_pressure"]]
pb = pb.merge(ftn.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left").merge(cov.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
pb["blitz"] = np.where(pb.n_blitzers.notna(), (pb.n_blitzers > 0).astype(float), np.nan)
mz = pb.defense_man_zone_type.fillna("")
pb["man"] = np.where(mz.str.contains("MAN"), 1.0, np.where(mz.str.contains("ZONE"), 0.0, np.nan))
pb["tp"] = np.where(pb.was_pressure.notna(), pb.was_pressure.astype(float), np.nan)
gd = sched.set_index("game_id").gameday
def prior_rate(side, col, K, decay=0.9):
    g = pb.dropna(subset=[col]).groupby(["game_id", side])[col].agg(["sum", "count"]).reset_index()
    g["gameday"] = pd.to_datetime(g.game_id.map(gd)); g = g.sort_values([side, "gameday"])
    lg = g["sum"].sum() / g["count"].sum(); out = []
    for t, x in g.groupby(side):
        s = n = 0.0
        for r in x.itertuples():
            out.append((r.game_id, t, (s + K * lg) / (n + K))); s = s * decay + r.sum; n = n * decay + r.count
    return pd.DataFrame(out, columns=["game_id", side, "v"]), lg
feats = {}
for nm, side, col, K in [("def_blitz", "defteam", "blitz", 150), ("off_blitzed", "posteam", "blitz", 150), ("def_tp", "defteam", "tp", 150), ("off_tp", "posteam", "tp", 150), ("def_man", "defteam", "man", 150)]:
    f, lg = prior_rate(side, col, K); df = df.merge(f.rename(columns={"v": nm}), on=["game_id", side], how="left"); df[nm] = df[nm].fillna(lg); feats[nm] = lg
df["blitz_x_risk"] = (df.def_blitz - feats["def_blitz"]) * (df.log_qb_risk - df.log_qb_risk.mean())
print("league rates:", {k: round(v, 3) for k, v in feats.items()}, "| spread of def blitz rate (10-90%):", df[df.season >= 2023].def_blitz.quantile([.1, .9]).round(3).tolist())
base = list(M.INT_X)
variants = {"current": base, "+ defense blitz rate (live)": base + ["def_blitz"], "+ blitz rate x QB risk (live)": base + ["def_blitz", "blitz_x_risk"],
            "+ offense blitzed rate (live)": base + ["def_blitz", "off_blitzed"],
            "true pressure instead of sack+hit (history only)": [c for c in base if c not in ("off_press_rate", "def_press_rate")] + ["def_tp", "off_tp"],
            "+ true pressure (history only)": base + ["def_tp", "off_tp"], "+ man-coverage rate (history only)": base + ["def_man"]}
rows = []
for nm, X in variants.items():
    M.INT_X = X; ps = []
    for s in range(2023, 2027):
        tr, te = df[df.season < s], df[df.season == s].dropna(subset=M.ATT_X + [c for c in X if c != "log_proj_att"])
        am = M.fit_attempts(tr); im = M.fit_ints(tr, am); t = M.predict(te, am, im); ps.append(t[["season", "ints", "p_int", "exp_ints"]])
    o = pd.concat(ps); y = (o.ints >= 1).astype(int); base_b = np.mean((y.mean() - y) ** 2)
    for lab, mk in [("2023-26", o.season >= 2023), ("2023-25", o.season <= 2025)]:
        yy, pp = y[mk], o.p_int[mk]; bb = np.mean((yy.mean() - yy) ** 2)
        rows.append(dict(variant=nm, sample=lab, n=int(mk.sum()), brier=np.mean((pp - yy) ** 2), skill=1 - np.mean((pp - yy) ** 2) / bb,
                         logloss=-np.mean(yy * np.log(pp.clip(1e-6, 1)) + (1 - yy) * np.log((1 - pp).clip(1e-6, 1)))))
    last = im
    if nm != "current": print(nm, {k: (round(float(last.params[k]), 3), round(float(last.pvalues[k]), 3)) for k in X if k not in base})
R = pd.DataFrame(rows); pd.set_option("display.width", 200)
for lab, g in R.groupby("sample"):
    print("\n", lab, "n =", g.n.iloc[0]); print(g.set_index("variant")[["brier", "skill", "logloss"]].round(5).to_string())
