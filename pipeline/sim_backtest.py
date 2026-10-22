"""Walk-forward check of the simulator (2024-2026): are its means and P(over) as good as the direct projections,
and do its ranges match reality? Uses the projection model's per-player inputs for past games, simulates, compares."""
import numpy as np, pandas as pd
import proj_model as P, td_model as T, sim_model as S

pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); d = P.efficiency(d, P.position_means(d))
# TD model inputs for the same games (walk-forward)
tpbp, tsched, tpl = T.load(); tpg, tteam, tctx = T.build_tables(tpbp, tsched, tpl)
tf = T.features(tpg, tteam, tctx); tf = tf[(tf.pr_n >= 1) & tf.implied.notna() & tf.rush_share.notna()]
tds = []
for s in range(2024, 2027):
    tmm = T.fit_team(tteam[tteam.season < s]); mr, mp = T.fit_players(tf[tf.season < s], tmm)
    tds.append(T.predict(tf[tf.season == s], tmm, mr, mp)[["game_id", "pid", "lam_rush", "lam_rec", "exp_td", "rush_share"]])
tds = pd.concat(tds)

rng = np.random.default_rng(7)
rows = []
for s in range(2024, 2027):
    tr, te = d[d.season < s].copy(), d[d.season == s].copy()
    va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[tr.pr_n >= 1]); te = P.apply_calibration(te, cal)
    te = te[(te.pr_n >= 1) & te.implied.notna() & (((te.pos == "QB") & te.qb_start) | te.pos.isin(["RB", "WR", "TE"]))]
    te = te.merge(tds, on=["game_id", "pid"], how="left").dropna(subset=["exp_td"])
    games = te.game_id.unique()
    if s == 2026: games = games
    for gid in games[:400] if s < 2026 else games:   # cap per season for speed
        g = te[te.game_id == gid]
        team_rows = {}
        for tm, x in g.groupby("posteam"):
            x = x.copy(); x["name"] = x.pid
            # per-player inputs: projected targets/carries (calibrated), efficiency, TD rates; team volume from the volume model
            team_rows[tm] = x[["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc", "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share"]]
        sim = S.simulate_game(team_rows, n_sim=1500, rng=rng)
        for tm, ss in sim.items():
            x = g[g.posteam == tm].set_index("pid")
            for i, pid in enumerate(ss["pids"]):
                a = x.loc[pid]
                rows.append(dict(season=s, pid=pid, pos=a.pos,
                    rec_yds=a.rec_yds, p_rec_yds=a.p_rec_yds, s_rec_yds=ss["rec_yds"][:, i].mean(), s_rec_yds_q10=np.quantile(ss["rec_yds"][:, i], .1), s_rec_yds_q90=np.quantile(ss["rec_yds"][:, i], .9),
                    rush_yds=a.rush_yds, p_rush_yds=a.p_rush_yds, s_rush_yds=ss["rush_yds"][:, i].mean(),
                    rec=a.rec, p_rec=a.p_rec, s_rec=ss["rec"][:, i].mean(),
                    pass_yds=a.pass_yds, p_pass_yds=a.p_pass_yds, s_pass_yds=ss["pass_yds"][:, i].mean(),
                    td=(a.rtd + a.ptd) if "rtd" in a else np.nan,
                    s_p_td=((ss["rec_td"][:, i] + ss["rush_td"][:, i]) >= 1).mean(),
                    over_rec_yds_line=a.p_rec_yds, s_p_over=(ss["rec_yds"][:, i] > a.p_rec_yds).mean(), hit_over=float(a.rec_yds > a.p_rec_yds)))
o = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"n = {len(o)} player-games")
for st, grp in [("rec_yds", "WR"), ("rec_yds", "TE"), ("rec", "WR"), ("rush_yds", "RB"), ("pass_yds", "QB")]:
    r = o[o.pos == grp]
    print(f"{grp} {st:9s} MAE  direct {np.mean(np.abs(r[st] - r['p_' + st])):6.2f}   sim {np.mean(np.abs(r[st] - r['s_' + st])):6.2f}   | mean actual {r[st].mean():.1f} direct {r['p_' + st].mean():.1f} sim {r['s_' + st].mean():.1f}")
w = o[o.pos.isin(["WR", "TE", "RB"]) & (o.p_rec_yds > 10)]
print(f"\nrange check (rec yds, players projected 10+): actual inside sim 10-90% band {((w.rec_yds >= w.s_rec_yds_q10) & (w.rec_yds <= w.s_rec_yds_q90)).mean():.1%} (target 80%)")
w["band"] = pd.cut(w.s_p_over, [0, .3, .4, .5, .6, 1])
print("\nP(over own projection) calibration from sim:\n", w.groupby("band", observed=True).agg(n=("hit_over", "size"), pred=("s_p_over", "mean"), actual=("hit_over", "mean")).round(3))

# ---- the point of the simulator: teammate correlations. Compare sim-implied corr to actual corr (same game, same team).
import itertools
ex = []
for s in [2025]:
    tr, te = d[d.season < s].copy(), d[d.season == s].copy()
    va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[tr.pr_n >= 1]); te = P.apply_calibration(te, cal)
    te = te[(te.pr_n >= 1) & te.implied.notna() & (((te.pos == "QB") & te.qb_start) | te.pos.isin(["RB", "WR", "TE"]))].merge(tds, on=["game_id", "pid"], how="left").dropna(subset=["exp_td"])
    for gid in te.game_id.unique()[:150]:
        g = te[te.game_id == gid]; team_rows = {}
        for tm, x in g.groupby("posteam"):
            x = x.copy(); x["name"] = x.pid
            team_rows[tm] = x[["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc", "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share"]]
        sim = S.simulate_game(team_rows, n_sim=800, rng=rng)
        for tm, ss in sim.items():
            x = g[g.posteam == tm].set_index("pid"); pos = dict(zip(ss["pids"], ss["pos"]))
            qb = [i for i, p in enumerate(ss["pids"]) if pos[p] == "QB"]
            wrs = [i for i, p in enumerate(ss["pids"]) if pos[p] == "WR" and x.loc[p, "p_tgt"] >= 4]
            if qb and wrs:
                i = qb[0]
                for j in wrs:
                    ex.append(dict(pair="QB pass yds vs WR rec yds", sim=np.corrcoef(ss["pass_yds"][:, i], ss["rec_yds"][:, j])[0, 1],
                                   a=x.loc[ss["pids"][i], "pass_yds"], b=x.loc[ss["pids"][j], "rec_yds"]))
            for j, k in itertools.combinations(wrs, 2):
                ex.append(dict(pair="WR1 rec yds vs WR2 rec yds", sim=np.corrcoef(ss["rec_yds"][:, j], ss["rec_yds"][:, k])[0, 1],
                               a=x.loc[ss["pids"][j], "rec_yds"], b=x.loc[ss["pids"][k], "rec_yds"]))
            rbs = [i for i, p in enumerate(ss["pids"]) if pos[p] == "RB" and x.loc[p, "p_car"] >= 8]
            if qb and rbs:
                ex.append(dict(pair="QB pass yds vs RB rush yds", sim=np.corrcoef(ss["pass_yds"][:, qb[0]], ss["rush_yds"][:, rbs[0]])[0, 1],
                               a=x.loc[ss["pids"][qb[0]], "pass_yds"], b=x.loc[ss["pids"][rbs[0]], "rush_yds"]))
e = pd.DataFrame(ex)
print("\nTeammate correlations: simulator vs actual (2025 games)")
for pr, grp in e.groupby("pair"):
    # actual corr: residualize on nothing, just raw corr across games (what a parlay faces)
    print(f"  {pr:28s} sim {grp.sim.mean():+.2f}   actual {np.corrcoef(grp.a, grp.b)[0, 1]:+.2f}   (n={len(grp)})")

# ---- PPR points: simulated mean vs direct projection, and boom/bust calibration
rows2 = []
for s in [2025, 2026]:
    tr, te = d[d.season < s].copy(), d[d.season == s].copy()
    va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
    cal = P.fit_calibration(tr[tr.pr_n >= 1]); te = P.apply_calibration(te, cal)
    te = te[(te.pr_n >= 1) & te.implied.notna() & (((te.pos == "QB") & te.qb_start) | te.pos.isin(["RB", "WR", "TE"]))].merge(tds, on=["game_id", "pid"], how="left").dropna(subset=["exp_td"])
    te["p_rush_td"] = te.lam_rush; te["p_rec_td"] = te.lam_rec; te["p_pass_td"] = np.where(te.pos == "QB", te.exp_td * (1 - te.rush_share) * 0.97, 0)
    te["p_ints"] = te.p_ints.fillna(0)
    te["direct_ppr"] = P.fantasy(te, "p_")["ppr"]; te["actual_ppr"] = P.fantasy(te.assign(**{f"a_{c}": te[c] for c in ["pass_yds", "pass_td", "ints", "sacks", "pass_1d", "rush_yds", "rush_td", "rush_1d", "rec", "rec_yds", "rec_td", "rec_1d"]}), "a_")["ppr"]
    for gid in te.game_id.unique()[:260]:
        g = te[te.game_id == gid]; team_rows = {}
        for tm, x in g.groupby("posteam"):
            x = x.copy(); x["name"] = x.pid
            team_rows[tm] = x[["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc", "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share", "p_ints"]]
        sim = S.simulate_game(team_rows, n_sim=1500, rng=rng)
        for tm, ss in sim.items():
            x = g[g.posteam == tm].set_index("pid")
            for i, pid in enumerate(ss["pids"]):
                a = x.loc[pid]; f = ss["fp_ppr"][:, i]
                rows2.append(dict(pos=a.pos, actual=a.actual_ppr, direct=a.direct_ppr, sim=f.mean(), q10=np.quantile(f, .1), q90=np.quantile(f, .9)))
o2 = pd.DataFrame(rows2)
print("\nPPR points, 2025-26:")
for g in ["QB", "RB", "WR", "TE"]:
    r = o2[o2.pos == g]
    print(f"  {g}: MAE direct {np.mean(np.abs(r.actual - r.direct)):.2f}  sim {np.mean(np.abs(r.actual - r.sim)):.2f} | mean actual {r.actual.mean():.1f} direct {r.direct.mean():.1f} sim {r.sim.mean():.1f} | inside sim 10-90 band {((r.actual >= r.q10) & (r.actual <= r.q90)).mean():.0%}")
