"""
Anytime touchdown scorer model.

Two layers:
  1. TEAM: expected offensive TDs from the Vegas implied team total (Poisson GLM),
     split into rushing vs receiving TDs by the team's recent rush-TD share.
  2. PLAYER: each player's share of those TDs from his recent opportunity shares
     (carries, carries inside the 10, targets, red-zone targets) via two Poisson
     GLMs (rushing TDs, receiving TDs) with the team's expected TDs as exposure.

P(anytime TD) = 1 - exp(-(rush lambda + rec lambda)). Offensive TDs only
(return and defensive TDs are ~3% of scores and ignored).
All priors use only games BEFORE the one being predicted.
"""
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "data"
PL_DECAY = 0.80        # per-game decay for player shares (~3-game half-life)
TEAM_DECAY = 0.90      # team rush-TD share
EPS = 0.01             # floor inside logs for zero shares
K_TD_SH = 4.0          # pseudo-TDs shrinking a player's own TD share toward his opportunity share
RUSH_X = ["l_car_sh", "l_i10_sh", "l_rtd_sh", "pos_RB", "l_dvp_td", "over_xtd"]      # RB/WR/TE rushing (+ opp rushing TDs allowed, finishing skill)
K_OVER = 6.0   # pseudo-games pulling season-to-date (actual - expected TDs) toward 0; last season carried at 0.5
DVP_DECAY, DVP_CARRY, DVP_K = 0.9, 0.3, 8   # opponent TDs-allowed-by-position: recent-weighted, last season x0.3, 8-game averaging
QB_RUSH_X = ["l_car_sh", "l_i10_sh", "l_rtd_sh", "over_xtd"]            # QB rushing gets its own model
GL_BOOST = 2.0   # goal-line (inside-10) carry weight for non-QB rushers, x the fitted value (td_redzone_test2.py)
TD_SEASON_CARRY = {"RB": 0.6, "WR": 0.6, "TE": 0.6, "QB": 1.0}   # chosen by td_carry_test.py
TD_TEAM_CARRY = 0.3
REC_X = ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB", "over_xtd"]


# ---------------------------------------------------------------- load
def load():
    cols = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
            "yardline_100", "rush_attempt", "pass_attempt", "sack", "qb_kneel",
            "rusher_player_id", "rusher_player_name", "receiver_player_id",
            "receiver_player_name", "rush_touchdown", "pass_touchdown", "two_point_attempt", "air_yards"]
    pbp = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{DATA}/pbp_*.parquet"))])
    sched = pd.read_parquet(f"{DATA}/sched.parquet")
    players = pd.read_parquet(f"{DATA}/players.parquet", columns=["gsis_id", "display_name", "position"])
    return pbp, sched, players


# ---------------------------------------------------------------- tables
def build_tables(pbp, sched, players):
    p = pbp[(pbp.two_point_attempt != 1) & pbp.posteam.notna()].copy()
    car = p[(p.rush_attempt == 1) & p.rusher_player_id.notna() & (p.qb_kneel != 1)]
    tgt = p[(p.pass_attempt == 1) & (p.sack != 1) & p.receiver_player_id.notna()]

    r = (car.assign(i10=(car.yardline_100 <= 10).astype(int), i5=(car.yardline_100 <= 5).astype(int))
         .groupby(["game_id", "posteam", "rusher_player_id"], as_index=False)
         .agg(car=("rush_attempt", "size"), i10=("i10", "sum"), i5=("i5", "sum"), rtd=("rush_touchdown", "sum"),
              name_r=("rusher_player_name", "last"))
         .rename(columns={"rusher_player_id": "pid"}))
    t = (tgt.assign(rz=(tgt.yardline_100 <= 20).astype(int),
                    ez=(tgt.air_yards.fillna(-99) >= tgt.yardline_100).astype(int))   # thrown into the end zone
         .groupby(["game_id", "posteam", "receiver_player_id"], as_index=False)
         .agg(tgt=("pass_attempt", "size"), rz=("rz", "sum"), ez=("ez", "sum"), ptd=("pass_touchdown", "sum"),
              name_t=("receiver_player_name", "last"))
         .rename(columns={"receiver_player_id": "pid"}))
    pg = r.merge(t, on=["game_id", "posteam", "pid"], how="outer")
    for c in ["car", "i10", "i5", "rtd", "tgt", "rz", "ez", "ptd"]:
        pg[c] = pg[c].fillna(0)
    pg["name"] = pg.name_r.fillna(pg.name_t)
    pg = pg.drop(columns=["name_r", "name_t"])

    team = pg.groupby(["game_id", "posteam"], as_index=False)[["car", "i10", "i5", "rtd", "tgt", "rz", "ez", "ptd"]].sum()
    team = team.rename(columns={c: "tm_" + c for c in ["car", "i10", "i5", "rtd", "tgt", "rz", "ez", "ptd"]})
    pg = pg.merge(team, on=["game_id", "posteam"])
    for a, b in [("car", "car"), ("i10", "i10"), ("i5", "i5"), ("rtd", "rtd"), ("tgt", "tgt"), ("rz", "rz"), ("ez", "ez"), ("ptd", "ptd")]:
        pg[f"{a}_sh"] = np.where(pg[f"tm_{b}"] > 0, pg[a] / pg[f"tm_{b}"], np.nan)

    pos = players.drop_duplicates("gsis_id").set_index("gsis_id").position
    pg["pos"] = pg.pid.map(pos).fillna("UNK")

    # game context
    s = sched[["game_id", "season", "week", "gameday", "home_team", "away_team", "spread_line", "total_line"]]
    ctx = pd.concat([s.assign(team=s.home_team, opp=s.away_team, home=1, margin=s.spread_line),
                     s.assign(team=s.away_team, opp=s.home_team, home=0, margin=-s.spread_line)])
    ctx["implied"] = ctx.total_line / 2 + ctx.margin / 2
    ctx["gameday"] = pd.to_datetime(ctx.gameday)
    ctx = ctx[["game_id", "season", "week", "gameday", "team", "opp", "home", "margin", "implied"]]
    team = team.merge(ctx.rename(columns={"team": "posteam"}), on=["game_id", "posteam"])
    team["tds"] = team.tm_rtd + team.tm_ptd
    pg = pg.merge(ctx[["game_id", "team", "gameday", "season", "week"]].rename(columns={"team": "posteam"}),
                  on=["game_id", "posteam"])
    return pg, team, ctx


def decayed_mean(df, key, cols, decay, date_col="gameday"):
    """Decayed average of cols over each key's PRIOR rows (NaN-aware). Adds pr_<col>, pr_n."""
    df = df.sort_values([key, date_col]).copy()
    vals = {c: df[c].to_numpy(float) for c in cols}
    out = {c: np.full(len(df), np.nan) for c in cols}
    n_out = np.zeros(len(df))
    for _, idx in df.groupby(key).indices.items():
        num = {c: 0.0 for c in cols}; den = {c: 0.0 for c in cols}; n = 0.0
        for i in idx:
            for c in cols:
                out[c][i] = num[c] / den[c] if den[c] > 0 else np.nan
            n_out[i] = n
            for c in cols:
                v = vals[c][i]
                num[c] *= decay; den[c] *= decay
                if not np.isnan(v):
                    num[c] += v; den[c] += 1
            n = n * decay + 1
    for c in cols:
        df[f"pr_{c}"] = out[c]
    df["pr_n"] = n_out
    return df


# ---------------------------------------------------------------- features
def features(pg, team, ctx, predict_rows=None):
    """predict_rows: (game_id, posteam, pid, name, pos, gameday, season, week) for unplayed games."""
    # team rush-TD share, decayed & shrunk to league
    lg_rush_share = team.tm_rtd.sum() / team.tds.sum()
    tm = decayed_mean(team, "posteam", ["tm_rtd", "tds"], TEAM_DECAY)
    tm["rush_share"] = (tm.pr_tm_rtd * tm.pr_n + 3 * lg_rush_share * 1) / (tm.pr_tds * tm.pr_n + 3)
    tm["rush_share"] = tm.rush_share.fillna(lg_rush_share)

    d = pg.copy()
    d["_pred"] = False
    if predict_rows is not None:
        d = pd.concat([d, predict_rows.assign(_pred=True)], ignore_index=True)
    share_cols = ["car_sh", "i10_sh", "i5_sh", "rtd_sh", "tgt_sh", "rz_sh", "ez_sh", "ptd_sh", "car", "tgt"]
    d = decayed_mean(d, "pid", share_cols, PL_DECAY)
    if TD_SEASON_CARRY != 1.0 or TD_TEAM_CARRY != 1.0 or isinstance(TD_SEASON_CARRY, dict):
        import proj_model as _P
        sc = TD_SEASON_CARRY
        if isinstance(sc, dict):
            sc = {k: v for k, v in sc.items()}; sc.update({"FB": sc.get("RB", 1.0)})
        keep = [c for c in share_cols if c not in ("car", "tgt")]
        d = _P.decayed_carry(d.drop(columns=["pr_" + c for c in keep]), "pid", keep, PL_DECAY, sc, TD_TEAM_CARRY)

    # shrink a player's TD share toward his opportunity share (TDs are lumpy)
    d["rtd_sh_s"] = (d.pr_rtd_sh * d.pr_n + K_TD_SH / 4 * d.pr_i10_sh.fillna(d.pr_car_sh)) / (d.pr_n + K_TD_SH / 4)
    d["ptd_sh_s"] = (d.pr_ptd_sh * d.pr_n + K_TD_SH / 4 * d.pr_rz_sh.fillna(d.pr_tgt_sh)) / (d.pr_n + K_TD_SH / 4)
    for a, b in [("l_car_sh", "pr_car_sh"), ("l_i10_sh", "pr_i10_sh"), ("l_i5_sh", "pr_i5_sh"), ("l_rtd_sh", "rtd_sh_s"),
                 ("l_tgt_sh", "pr_tgt_sh"), ("l_rz_sh", "pr_rz_sh"), ("l_ez_sh", "pr_ez_sh"), ("l_ptd_sh", "ptd_sh_s")]:
        d[a] = np.log(d[b].fillna(0).clip(lower=0) + EPS)
    # opponent's TDs allowed to this player's position group (season-to-date, recent-weighted)
    import proj_model as _P
    gp = lambda pos: np.select([pos == "QB", pos.isin(["RB", "FB"]), pos == "WR", pos == "TE"], ["QB", "RB", "WR", "TE"], "OTH")
    oppmap = ctx[["game_id", "team", "opp"]].rename(columns={"team": "posteam"})
    gg = pg.merge(oppmap, on=["game_id", "posteam"]).assign(grp=lambda x: gp(x.pos))
    gg["td"] = np.where(gg.grp == "QB", gg.rtd, gg.rtd + gg.ptd)
    alw = gg[gg.grp != "OTH"].groupby(["game_id", "opp", "grp", "gameday", "season"], as_index=False).td.sum()
    dg = alw[["game_id", "opp", "gameday", "season"]].drop_duplicates()
    full = dg.merge(pd.DataFrame({"grp": ["QB", "RB", "WR", "TE"]}), how="cross").merge(alw, how="left").fillna({"td": 0})
    full["one"] = 1.0; full["k"] = full.opp + "|" + full.grp; full["posteam"] = ""
    lg = full.groupby("grp").td.mean()
    t = _P.decayed_carry(full, "k", ["td", "one"], DVP_DECAY, season_carry=DVP_CARRY, mode="sum")
    t["l_dvp_td"] = np.log(((t.ds_td + DVP_K * t.grp.map(lg)) / (t.ds_one + DVP_K)) / t.grp.map(lg))
    d = d.merge(oppmap, on=["game_id", "posteam"], how="left")
    d["grp"] = gp(d.pos)
    d = d.merge(t[["game_id", "opp", "grp", "l_dvp_td"]], on=["game_id", "opp", "grp"], how="left")
    if predict_rows is not None:   # upcoming games: roll each defense's state forward one game
        tl = _P.roll_forward(t, "k", ["td", "one"], DVP_DECAY)
        tl["v"] = np.log(((tl.ds_td + DVP_K * tl.grp.map(lg)) / (tl.ds_one + DVP_K)) / tl.grp.map(lg))
        m = d._pred
        d.loc[m, "l_dvp_td"] = (d.loc[m, "opp"] + "|" + d.loc[m, "grp"]).map(tl.v)
    d["l_dvp_td"] = d.l_dvp_td.fillna(0.0)
    # finishing skill: TDs scored minus TDs expected from where touches happened (season-to-date, prior games; shrunk)
    x = pd.read_parquet("data/xtd.parquet")[["game_id", "pid", "season", "td_std", "xtd_std", "games_std", "td_season", "xtd_season", "games_season"]]
    prev = x.sort_values("game_id").groupby(["pid", "season"]).tail(1)[["pid", "season", "td_season", "xtd_season"]].copy()
    prev["season"] += 1; prev = prev.rename(columns={"td_season": "td_prev", "xtd_season": "xtd_prev"})
    x = x.merge(prev, on=["pid", "season"], how="left"); x[["td_prev", "xtd_prev"]] = x[["td_prev", "xtd_prev"]].fillna(0)
    x["over_xtd"] = ((x.td_std - x.xtd_std) + 0.5 * (x.td_prev - x.xtd_prev)) / (x.games_std + K_OVER)
    d = d.merge(x[["game_id", "pid", "over_xtd", "td_std", "xtd_std"]], on=["game_id", "pid"], how="left")
    if predict_rows is not None:   # upcoming games: each player's season-to-date AFTER his last game
        cur = x[x.season == x.season.max()].sort_values("game_id").groupby("pid").tail(1).set_index("pid")
        over_now = ((cur.td_season - cur.xtd_season) + 0.5 * (cur.td_prev - cur.xtd_prev)) / (cur.games_season + K_OVER)
        m = d._pred.to_numpy()
        d.loc[m, "over_xtd"] = d.loc[m, "pid"].map(over_now).to_numpy()
        d.loc[m, "td_std"] = d.loc[m, "pid"].map(cur.td_season).to_numpy()
        d.loc[m, "xtd_std"] = d.loc[m, "pid"].map(cur.xtd_season).to_numpy()
    d["over_xtd"] = d.over_xtd.fillna(0.0)
    d["is_qb"] = (d.pos == "QB").astype(int)
    d["pos_TE"] = (d.pos == "TE").astype(int)
    d["pos_RB"] = (d.pos.isin(["RB", "FB"])).astype(int)

    # attach team context + rush share (for future games, roll team state forward)
    d = d.merge(ctx[["game_id", "team", "implied", "margin"]].rename(columns={"team": "posteam"}),
                on=["game_id", "posteam"], how="left")
    hist_rs = tm[["game_id", "posteam", "rush_share"]]
    d = d.merge(hist_rs, on=["game_id", "posteam"], how="left")
    if predict_rows is not None:
        last = tm.sort_values("gameday").groupby("posteam").tail(1).copy()
        n = last.pr_n * TEAM_DECAY + 1
        rtd = (last.pr_tm_rtd * last.pr_n * TEAM_DECAY + last.tm_rtd) / n
        tds = (last.pr_tds * last.pr_n * TEAM_DECAY + last.tds) / n
        last["rs_next"] = (rtd * n + 3 * lg_rush_share) / (tds * n + 3)
        m = last.set_index("posteam").rs_next
        d.loc[d._pred, "rush_share"] = d.loc[d._pred, "posteam"].map(m)
    return d


# ---------------------------------------------------------------- models
def fit_team(team_hist):
    tr = team_hist.dropna(subset=["implied"])
    X = sm.add_constant(np.log(tr[["implied"]].clip(lower=5)))
    return sm.GLM(tr.tds, X, family=sm.families.Poisson()).fit()


def team_exp_td(tm_model, implied):
    X = sm.add_constant(np.log(pd.DataFrame({"implied": implied}).clip(lower=5)), has_constant="add")
    return tm_model.predict(X).to_numpy()


def fit_players(d, tm_model):
    d = d.dropna(subset=["implied", "rush_share"]).copy()
    d = d[d.pr_n >= 1]
    d["exp_td"] = team_exp_td(tm_model, d.implied)
    off_r = np.log(d.exp_td * d.rush_share)
    off_p = np.log(d.exp_td * (1 - d.rush_share))
    nq, q = (d.is_qb == 0).to_numpy(), (d.is_qb == 1).to_numpy()
    mr = sm.GLM(d.rtd[nq], sm.add_constant(d.loc[nq, RUSH_X]), family=sm.families.Poisson(), offset=off_r[nq]).fit()
    mr.qb_rush = sm.GLM(d.rtd[q], sm.add_constant(d.loc[q, QB_RUSH_X]), family=sm.families.Poisson(), offset=off_r[q]).fit()
    mp = sm.GLM(d.ptd, sm.add_constant(d[REC_X]), family=sm.families.Poisson(), offset=off_p).fit()
    # QB rushing TDs cluster (multi-TD games), so a Poisson mean misstates P(1+), and by different amounts
    # for pocket vs running QBs. Learn one scale per QB-role band (by carry share) on the TRAINING rows.
    qbr = d[d.is_qb == 1].copy()
    lam = (np.exp(sm.add_constant(qbr[QB_RUSH_X], has_constant="add") @ mr.qb_rush.params) * qbr.exp_td * qbr.rush_share
           + np.exp(sm.add_constant(qbr[REC_X], has_constant="add") @ mp.params) * qbr.exp_td * (1 - qbr.rush_share))
    hit = ((qbr.rtd + qbr.ptd) >= 1).to_numpy()
    cs = qbr.pr_car_sh.fillna(0).to_numpy()
    mr.qb_cuts = list(np.quantile(cs, [1 / 3, 2 / 3]))
    band = np.digitize(cs, mr.qb_cuts)
    grid = np.linspace(0.4, 1.4, 201)
    mr.qb_scales = [float(grid[np.argmin([abs((1 - np.exp(-c * lam[band == b])).mean() - hit[band == b].mean()) for c in grid])])
                    for b in range(3)]
    mr.qb_scale = 1.0
    return mr, mp


def predict(d, tm_model, mr, mp):
    d = d.copy()
    d["exp_td"] = team_exp_td(tm_model, d.implied)
    pr = mr.params.copy(); pr["l_i10_sh"] = pr["l_i10_sh"] * GL_BOOST
    r_all = np.exp(sm.add_constant(d[RUSH_X], has_constant="add") @ pr)
    r_qb = np.exp(sm.add_constant(d[QB_RUSH_X], has_constant="add") @ mr.qb_rush.params)
    d["lam_rush"] = np.where(d.is_qb == 1, r_qb, r_all) * d.exp_td * d.rush_share
    d["lam_rec"] = np.exp(sm.add_constant(d[REC_X], has_constant="add") @ mp.params) * d.exp_td * (1 - d.rush_share)
    qs = np.array(getattr(mr, "qb_scales", [1.0, 1.0, 1.0]))[np.digitize(d.pr_car_sh.fillna(0), getattr(mr, "qb_cuts", [9, 9]))]
    d["qb_mult"] = np.where(d.is_qb == 1, qs, 1.0)
    d["lam"] = d.qb_mult * (d.lam_rush + d.lam_rec)
    d["p_td"] = 1 - np.exp(-d.lam)
    return d


def american(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)
