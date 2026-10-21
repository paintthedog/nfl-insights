"""
QB interception prop model.

Predicts each starting QB's expected interceptions in a game with a Poisson GLM,
then converts to P(>=1 INT) and fair odds for the over/under 0.5 INT prop.

Every feature is computed only from games played BEFORE the game being predicted
(no leakage), using exponentially decayed history with empirical-Bayes shrinkage.
"""
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "data"
QB_DECAY = 0.97      # per-game decay for QB history (~23-game half-life)
TEAM_DECAY = 0.90    # per-game decay for team offense/defense (~6-game half-life)
K_QB_INT = 600       # shrinkage strength (pseudo-attempts) for QB INT rate
K_DEF_INT = 1500     # defense INT rate is mostly noise -> shrink hard
K_PRESS = 150        # pseudo-dropbacks for pressure (sack + hit) rates
K_ADOT = 150         # pseudo-attempts for aDOT
K_QB_PD = 600        # pseudo-attempts for QB passes-defensed rate
K_IW = 200           # pseudo-charted-attempts pulling INT-worthy rate toward the QB's INT-implied level


# ---------------------------------------------------------------- load
def load():
    cols = ["game_id", "season", "week", "season_type", "posteam", "defteam",
            "home_team", "passer_player_id", "passer_player_name",
            "complete_pass", "incomplete_pass", "interception", "sack",
            "qb_hit", "qb_dropback", "air_yards", "two_point_attempt",
            "play_id", "pass_defense_1_player_id"]
    pbp = pd.concat([pd.read_parquet(f, columns=cols)
                     for f in sorted(glob.glob(f"{DATA}/pbp_*.parquet"))])
    # FTN charting (2022+): interception-worthy throws, joined play-by-play
    ftn_files = sorted(glob.glob(f"{DATA}/ftn_*.parquet"))
    if ftn_files:
        ftn = pd.concat([pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id",
                                                     "is_interception_worthy"])
                         for f in ftn_files]).rename(
            columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        ftn["play_id"] = ftn.play_id.astype(float)
        pbp["play_id"] = pbp.play_id.astype(float)
        pbp = pbp.merge(ftn.drop_duplicates(["game_id", "play_id"]),
                        on=["game_id", "play_id"], how="left")
    else:
        pbp["is_interception_worthy"] = np.nan
    sched = pd.read_parquet(f"{DATA}/sched.parquet")
    return pbp, sched


# ---------------------------------------------------------------- game tables
def build_tables(pbp, sched):
    p = pbp[(pbp.two_point_attempt != 1) & pbp.posteam.notna()].copy()
    p["att"] = (p.complete_pass + p.incomplete_pass + p.interception).fillna(0)
    p["pressure"] = ((p.sack == 1) | (p.qb_hit == 1)).astype(int) * p.qb_dropback.fillna(0)
    p["ay"] = np.where(p.att == 1, p.air_yards, np.nan)
    # breakups on the QB's throws (excluding the INT play itself)
    p["pd_q"] = (p.pass_defense_1_player_id.notna() & (p.interception != 1)).astype(int) * p.att
    p["iw"] = (p.is_interception_worthy == True).astype(int) * p.att
    p["charted"] = p.is_interception_worthy.notna().astype(int) * p.att

    # QB-game
    qb = (p[p.passer_player_id.notna()]
          .groupby(["game_id", "season", "week", "posteam", "defteam",
                    "passer_player_id"], as_index=False)
          .agg(name=("passer_player_name", "last"), att=("att", "sum"),
               ints=("interception", "sum"), ay_sum=("ay", "sum"),
               ay_n=("ay", "count"), pd_q=("pd_q", "sum"), iw=("iw", "sum"),
               charted=("charted", "sum")))
    # starter = most attempts for that team in that game
    qb = qb.sort_values("att", ascending=False)
    qb["is_starter"] = ~qb.duplicated(["game_id", "posteam"])

    # team-game offense / defense
    tg = (p.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False)
          .agg(db=("qb_dropback", "sum"), press=("pressure", "sum"),
               att=("att", "sum"), ints=("interception", "sum")))

    # game context from schedule (home/away perspective)
    s = sched[["game_id", "season", "week", "gameday", "home_team", "away_team",
               "spread_line", "total_line", "roof", "wind", "temp"]].copy()
    home = s.assign(team=s.home_team, opp=s.away_team, home=1,
                    exp_margin=s.spread_line)
    away = s.assign(team=s.away_team, opp=s.home_team, home=0,
                    exp_margin=-s.spread_line)
    ctx = pd.concat([home, away])[["game_id", "season", "week", "gameday", "team",
                                   "opp", "home", "exp_margin", "total_line",
                                   "roof", "wind", "temp"]]
    ctx["implied_pts"] = ctx.total_line / 2 + ctx.exp_margin / 2
    outdoor = ~ctx.roof.isin(["dome", "closed"])
    ctx["wind"] = np.where(outdoor, ctx.wind.fillna(8), 0)  # 8 mph ~ median outdoor
    ctx["gameday"] = pd.to_datetime(ctx.gameday)
    return qb, tg, ctx


# ---------------------------------------------------------------- decayed priors
def decayed_prior(df, key, date_col, num_cols, decay):
    """For each row, decayed sums of num_cols over that key's PRIOR rows only."""
    df = df.sort_values([key, date_col]).copy()
    out = {c: np.zeros(len(df)) for c in num_cols}
    out["n_prior"] = np.zeros(len(df))
    for _, idx in df.groupby(key).indices.items():
        acc = {c: 0.0 for c in num_cols}
        n = 0
        for i in idx:
            for c in num_cols:
                out[c][i] = acc[c]
            out["n_prior"][i] = n
            for c in num_cols:
                acc[c] = acc[c] * decay + (df[c].iat[i] if pd.notna(df[c].iat[i]) else 0)
            n += 1
    for c in num_cols:
        df[f"prior_{c}"] = out[c]
    df["prior_games"] = out["n_prior"]
    return df


def build_features(qb, tg, ctx, predict_rows=None):
    """predict_rows: DataFrame of (game_id, season, week, gameday, posteam, defteam,
    passer_player_id, name) for games not yet played -- appended so they get priors."""
    league_int = qb.ints.sum() / qb.att.sum()
    league_ay = qb.ay_sum.sum() / qb.ay_n.sum()
    league_press = tg.press.sum() / tg.db.sum()
    league_pd = qb.pd_q.sum() / qb.att.sum()
    league_iw = qb.iw.sum() / max(qb.charted.sum(), 1)
    gd = ctx[["game_id", "gameday"]].drop_duplicates()

    # --- QB history: use ALL QB appearances (starts and relief) for priors
    q = qb.merge(gd, on="game_id")
    if predict_rows is not None:
        q = pd.concat([q, predict_rows.assign(att=np.nan, ints=np.nan, ay_sum=np.nan,
                                               ay_n=np.nan, pd_q=np.nan, iw=np.nan,
                                               charted=np.nan, is_starter=True,
                                               _pred=True)])
    q["_pred"] = q.get("_pred", False)
    q["_pred"] = q["_pred"].fillna(False).astype(bool)
    q["gpa"] = np.where(q.att > 0, q.att, np.nan)  # attempts/game when he played
    q = decayed_prior(q, "passer_player_id", "gameday",
                      ["att", "ints", "ay_sum", "ay_n", "pd_q", "iw", "charted"], QB_DECAY)
    # decayed attempts-per-start average (starts only)
    q["start_att"] = np.where(q.is_starter, q.att, np.nan)
    q["start_ct"] = np.where(q.is_starter & q.att.notna(), 1.0, 0.0)
    q = decayed_prior(q, "passer_player_id", "gameday", ["start_att", "start_ct"], 0.85)
    q["qb_att_pg"] = q.prior_start_att / q.prior_start_ct.replace(0, np.nan)

    q["qb_int_rate"] = (q.prior_ints + K_QB_INT * league_int) / (q.prior_att + K_QB_INT)
    q["qb_adot"] = (q.prior_ay_sum + K_ADOT * league_ay) / (q.prior_ay_n + K_ADOT)
    q["qb_exp"] = np.log1p(q.prior_att)  # experience (undecayed-ish proxy)
    q["qb_pd_rate"] = (q.prior_pd_q + K_QB_PD * league_pd) / (q.prior_att + K_QB_PD)
    # QB risk: INT-worthy throw rate, shrunk toward the INT-worthy rate implied by his
    # own INT history. Uncharted (pre-2022) = INT history; heavily charted = INT-worthy.
    implied_iw = q.qb_int_rate * (league_iw / league_int)
    q["qb_iw_rate"] = (q.prior_iw + K_IW * implied_iw) / (q.prior_charted + K_IW)

    # --- team offense / defense history
    t = tg.merge(gd, on="game_id")
    t["one"] = 1.0
    off = decayed_prior(t.rename(columns={"posteam": "team"}), "team", "gameday",
                        ["db", "press", "att", "one"], TEAM_DECAY)
    off["off_press_rate"] = (off.prior_press + K_PRESS * league_press) / (off.prior_db + K_PRESS)
    off["off_att_pg"] = off.prior_att / off.prior_one.replace(0, np.nan)
    defn = decayed_prior(t.rename(columns={"defteam": "team"}), "team", "gameday",
                         ["db", "press", "att", "ints"], TEAM_DECAY)
    defn["def_press_rate"] = (defn.prior_press + K_PRESS * league_press) / (defn.prior_db + K_PRESS)
    defn["def_int_rate"] = (defn.prior_ints + K_DEF_INT * league_int) / (defn.prior_att + K_DEF_INT)

    # For upcoming games the team rows don't exist yet: take each team's
    # decayed state AFTER its latest game.
    def latest_state(df, cols):
        df = df.sort_values("gameday")
        last = df.groupby("team").tail(1).copy()
        return last

    starters = q[q.is_starter].copy()
    starters = starters.merge(ctx.rename(columns={"team": "posteam"}),
                              on=["game_id", "posteam"], how="left",
                              suffixes=("", "_c"))

    hist = starters[~starters._pred]
    hist = hist.merge(off[["game_id", "team", "off_press_rate", "off_att_pg", "prior_games"]]
                      .rename(columns={"team": "posteam", "prior_games": "team_games"}),
                      on=["game_id", "posteam"], how="left")
    hist = hist.merge(defn[["game_id", "team", "def_press_rate", "def_int_rate"]]
                      .rename(columns={"team": "defteam"}),
                      on=["game_id", "defteam"], how="left")

    fut = starters[starters._pred]
    if len(fut):
        # roll team state forward one game past its last observation
        def roll(df, cols, decay):
            df = df.sort_values("gameday")
            last = df.groupby("team").tail(1).copy()
            for c in cols:
                last[f"prior_{c}"] = last[f"prior_{c}"] * decay + last[c].fillna(0)
            last["prior_games"] += 1
            return last
        o = roll(t.rename(columns={"posteam": "team"}).pipe(
            lambda d: decayed_prior(d, "team", "gameday", ["db", "press", "att", "one"], TEAM_DECAY)),
            ["db", "press", "att", "one"], TEAM_DECAY)
        o["off_press_rate"] = (o.prior_press + K_PRESS * league_press) / (o.prior_db + K_PRESS)
        o["off_att_pg"] = o.prior_att / o.prior_one
        d = roll(t.rename(columns={"defteam": "team"}).pipe(
            lambda d: decayed_prior(d, "team", "gameday", ["db", "press", "att", "ints"], TEAM_DECAY)),
            ["db", "press", "att", "ints"], TEAM_DECAY)
        d["def_press_rate"] = (d.prior_press + K_PRESS * league_press) / (d.prior_db + K_PRESS)
        d["def_int_rate"] = (d.prior_ints + K_DEF_INT * league_int) / (d.prior_att + K_DEF_INT)
        fut = fut.merge(o[["team", "off_press_rate", "off_att_pg", "prior_games"]]
                        .rename(columns={"team": "posteam", "prior_games": "team_games"}),
                        on="posteam", how="left")
        fut = fut.merge(d[["team", "def_press_rate", "def_int_rate"]]
                        .rename(columns={"team": "defteam"}), on="defteam", how="left")

    allrows = pd.concat([hist, fut], ignore_index=True)
    # QBs with no start history: fall back to team pass volume, then league
    allrows["qb_att_pg"] = allrows.qb_att_pg.fillna(allrows.off_att_pg).fillna(33)
    # offensive-line starters missing (from snap counts; upcoming games get a value from the injury report, else 0)
    try:
        ol = pd.read_parquet(f"{DATA}/ol_continuity.parquet")[["game_id", "posteam", "ol_missing"]]
        allrows = allrows.merge(ol, on=["game_id", "posteam"], how="left")
    except Exception:
        allrows["ol_missing"] = np.nan
    allrows["ol_missing"] = allrows.ol_missing.fillna(allrows.get("ol_missing_pred", np.nan)).fillna(0.9)
    allrows["league_int"] = league_int
    return allrows


# ---------------------------------------------------------------- models
ATT_X = ["exp_margin", "total_line", "qb_att_pg", "off_att_pg", "home"]
INT_X = ["log_proj_att", "log_qb_risk", "log_qb_pd", "qb_adot", "qb_exp_low",
         "off_press_rate", "def_press_rate", "log_def_rate", "wind", "exp_margin"]


def prep(df):
    df = df.copy()
    df["qb_exp_low"] = (df.qb_exp < np.log1p(300)).astype(int)
    df["log_qb_rate"] = np.log(df.qb_int_rate)
    df["log_def_rate"] = np.log(df.def_int_rate)
    df["log_qb_risk"] = np.log(df.qb_iw_rate)
    df["log_qb_pd"] = np.log(df.qb_pd_rate)
    return df


def fit_attempts(train):
    tr = train.dropna(subset=ATT_X + ["att"])
    tr = tr[tr.att >= 10]  # drop in-game injury exits so they don't drag volume down
    m = sm.OLS(tr.att, sm.add_constant(tr[ATT_X])).fit()
    return m


def fit_ints(train, att_model):
    tr = train.dropna(subset=ATT_X + ["att", "ints"]).copy()
    tr = tr[tr.att >= 10]
    tr["proj_att"] = att_model.predict(sm.add_constant(tr[ATT_X], has_constant="add")).clip(15, 55)
    tr["log_proj_att"] = np.log(tr.proj_att)
    tr = tr.dropna(subset=INT_X)
    # exposure = ACTUAL attempts during training (so coefficients describe per-attempt
    # risk); at prediction time we swap in projected attempts.
    m = sm.GLM(tr.ints, sm.add_constant(tr[[c for c in INT_X if c != "log_proj_att"]]),
               family=sm.families.Poisson(), offset=np.log(tr.att)).fit()
    return m


def predict(df, att_model, int_model):
    df = df.copy()
    df["proj_att"] = att_model.predict(sm.add_constant(df[ATT_X], has_constant="add")).clip(15, 55)
    X = sm.add_constant(df[[c for c in INT_X if c != "log_proj_att"]], has_constant="add")
    df["int_per_att"] = np.exp(X.values @ int_model.params.values)
    df["exp_ints"] = df.int_per_att * df.proj_att
    df["p_int"] = 1 - np.exp(-df.exp_ints)
    df["log_proj_att"] = np.log(df.proj_att)
    return df


def american(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)
