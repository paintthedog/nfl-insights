"""Ceilings (Nov 2, Clinton). Live Weeks 7-8 (scorecard, randomized PIT): ~14-15% of rushing yards, receiving yards and PPR
results beat the published ceiling (90th pct of the game simulator), target 10%; floors are fine (6-10% below).
Question: is that a property of the simulator over past seasons, and does more upside dispersion fix it without
breaking floors? Grid over sim_model.GAME_EFF_SD (per-game yards-per-touch multiplier) and a multiplier on the gamma
shapes YPT_SHAPE / YPC_SHAPE (lower shape = longer right tail). Walk-forward inputs exactly as sim_backtest.py.
Fit seasons 2023-2024, held out 2025-2026. Coverage by randomized PIT (ties split), same as the scorecard."""
import sys; sys.path.insert(0, ".")
import itertools, numpy as np, pandas as pd
import proj_model as P, td_model as T, sim_model as S

pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
tpbp, tsched, tpl = T.load(); tpg, tteam, tctx = T.build_tables(tpbp, tsched, tpl)
tf = T.features(tpg, tteam, tctx); tf = tf[(tf.pr_n >= 1) & tf.implied.notna() & tf.rush_share.notna()]
SEAS = [2023, 2024, 2025, 2026]; GPS = 180
COLS = ["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc", "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share", "p_ints"]
games = []
rs = np.random.default_rng(11)
for s in SEAS:
    tmm = T.fit_team(tteam[tteam.season < s]); mr, mp = T.fit_players(tf[tf.season < s], tmm)
    tds = T.predict(tf[tf.season == s], tmm, mr, mp)[["game_id", "pid", "lam_rush", "lam_rec", "exp_td", "rush_share"]]
    tr, te = d[d.season < s].copy(), d[d.season == s].copy()
    va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[tr.pr_n >= 1]); te = P.apply_calibration(te, cal)
    te = te[(te.pr_n >= 1) & te.implied.notna() & (((te.pos == "QB") & te.qb_start) | te.pos.isin(["RB", "WR", "TE"]))].merge(tds, on=["game_id", "pid"], how="left").dropna(subset=["exp_td"])
    te["p_ints"] = te.p_ints.fillna(0)
    te["actual_ppr"] = P.fantasy(te.assign(**{f"a_{c}": te[c] for c in ["pass_yds", "pass_td", "ints", "sacks", "pass_1d", "rush_yds", "rush_td", "rush_1d", "rec", "rec_yds", "rec_td", "rec_1d"]}), "a_")["ppr"]
    gids = te.game_id.unique(); gids = rs.choice(gids, min(GPS, len(gids)), replace=False)
    for gid in gids:
        g = te[te.game_id == gid]; tr_ = {}
        for tm, x in g.groupby("posteam"):
            x = x.copy(); x["name"] = x.pid; tr_[tm] = x[COLS]
        games.append((s, gid, tr_, g.set_index("pid")))
print("games", len(games), flush=True)

def pit(x, v, lo, hi):
    flo, fhi = (x < v).mean(), (x <= v).mean(); w = fhi - flo
    if w < 1e-9: return float(lo <= flo <= hi)
    return max(0.0, min(fhi, hi) - max(flo, lo)) / w
STATS = [("rec_yds", "rec_yds"), ("rush_yds", "rush_yds"), ("fp_ppr", "actual_ppr"), ("rec", "rec"), ("car", "car")]
base = (S.GAME_EFF_SD, S.YPT_SHAPE, S.YPC_SHAPE)
out = []
for eff, sh in itertools.product([0.30, 0.38, 0.46], [1.0, 0.75, 0.55]):
    S.GAME_EFF_SD, S.YPT_SHAPE, S.YPC_SHAPE = eff, base[1] * sh, base[2] * sh
    rng = np.random.default_rng(7)
    for s, gid, tr_, act in games:
        sim = S.simulate_game(tr_, n_sim=600, rng=rng)
        for tm, ss in sim.items():
            for i, pid in enumerate(ss["pids"]):
                a = act.loc[pid]; a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
                for st, ac in STATS:
                    x = ss[st][:, i]
                    if x.mean() < 0.5: continue
                    if st in ("rec", "rec_yds") and a.pos == "QB": continue
                    if st in ("rush_yds", "car") and a.pos not in ("RB",): continue
                    v = float(a[ac])
                    out.append((eff, sh, s, st, pit(x, v, 0, .1), pit(x, v, .9, 1), pit(x, v, .25, .75), abs(v - np.median(x))))
    print("done", eff, sh, flush=True)
S.GAME_EFF_SD, S.YPT_SHAPE, S.YPC_SHAPE = base
o = pd.DataFrame(out, columns=["eff", "shape", "season", "stat", "below", "above", "mid", "abs_med"])
o["set"] = np.where(o.season <= 2024, "fit 2023-24", "held out 2025-26")
o.to_csv("tests/t_ceiling_sim_rows.csv.gz", index=False)
pd.set_option("display.width", 220)
t = o.groupby(["set", "stat", "eff", "shape"]).agg(n=("below", "size"), below=("below", "mean"), above=("above", "mean"), mid=("mid", "mean"), mae_med=("abs_med", "mean")).round(3)
print(t.to_string())
