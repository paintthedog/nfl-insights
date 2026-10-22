"""
Player stat projections: team volume -> player share -> efficiency, with outcome ranges.

  * TEAM VOLUME: pass attempts, carries from the Vegas total/spread and each team's recent volume (OLS).
  * SHARE: player's decayed share of team targets / carries (same shares as the TD model).
  * EFFICIENCY: catch rate, yards per target, yards per carry, first downs per touch, QB completion %,
    yards per attempt, sack rate -- decayed and shrunk hard toward position averages.
  * CALIBRATION: per stat and position group, a Poisson GLM on log(raw projection) and the opponent's
    (shrunk) yards-allowed ratio, fit only on past seasons.
  * TOUCHDOWNS come from the TD model (td_model.py); INTs from a shrunk QB rate.
  * RANGES: empirical actual/projected ratios from out-of-sample games, by stat, position and size of projection.
"""
import glob, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import td_model as T

TEAM_DECAY = 0.90
SHARE_DECAY = 0.80
EFF_DECAY = 0.95
# Extra discount on everything before a season boundary / team change (1.0 = no extra discount).
SEASON_CARRY = {"WR": 0.5, "RB": 1.0, "QB": 1.0, "TE": 1.0}   # player usage shares; chosen by carryover_test2.py
TEAM_CARRY = 0.3         # player usage shares after the player changes teams
EFF_SEASON_CARRY = 1.0   # player efficiency (yards per touch etc.)
TEAM_VOL_CARRY = 1.0     # team play volume (pass attempts / carries per game)
USE_OL = True            # offensive-line starters missing, on QB passing stats (ol_test.py decides)
USE_WEATHER = True       # wind / cold terms in volume + calibration (weather_proj_test.py decides)
WEATHER_STATS = {"pass_yds", "cmp", "rec_yds", "rec", "att", "tgt", "rec_1d", "pass_1d"}
QB_POS = "QB"
GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
QUANTS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "two_point_attempt", "qb_kneel", "qb_hit", "qb_dropback",
            "passer_player_id", "rusher_player_id", "receiver_player_id", "pass_attempt", "rush_attempt", "sack",
            "complete_pass", "passing_yards", "rushing_yards", "receiving_yards", "pass_touchdown",
            "rush_touchdown", "interception", "first_down_rush", "first_down_pass", "air_yards", "pass_oe", "wp"]

# Optional features (switched on by the backtest that proved them). Each entry: stat -> extra calibration columns.
# Chosen by feature_test.py / role_change_test.py (walk-forward 2021-2026):
#   recent snap share + snap-share DROPS help; snap-share jumps hurt (one-week spikes), so only drops are used.
#   WOPR helps receivers slightly. Pass rate over expected and air-yards share alone added nothing.
# qd: this game's QB quality minus the QBs behind the team's recent numbers (team_adj.py); helps receivers most after a
# QB change (rec yds MAE 17.32 -> 17.18 when |qd| > .05; tests/t_rcv_qb.py)
EXTRA = {**{s: ["l_snap", "snap_dn", "l_wopr", "qd"] for s in ["tgt", "rec", "rec_yds", "rec_1d"]},
         **{s: ["l_snap", "snap_dn"] for s in ["car", "rush_yds", "rush_1d"]}}
USE_PROE = False
# Opponent-defense settings (tested in defense_test.py)
USE_DEF = True
DEF_K_PASS, DEF_K_RUSH = 300, 200    # pseudo-plays of league-average pulled into each defense's yards-allowed rate
DEF_DECAY = 0.90                     # per-game memory for defenses
POS_DEF = False                      # receiving stats use yards-per-target allowed to that position (RB/WR/TE)
POS_DEF_K = 60
DEF_SEASON_CARRY = 1.0       # extra discount on a pass defense's previous seasons
DEF_RUSH_SEASON_CARRY = 0.25  # run defenses: last season counts 25% (def_season_test.py)

# shrinkage strengths (pseudo-events at the position average)
K = dict(catch=40, ypt=80, rec1d=40, ypc=120, rush1d=100, cmp=150, ypa=250, sack=200, pass1d=150, qbint=600)


def scoring(df, fmt):
    """Fantasy points from stat columns (means or actuals)."""
    g = lambda c: df[c] if c in df else 0
    base = g("pass_yds") / 25 + 4 * g("pass_td") - 2 * g("ints") + g("rush_yds") / 10 + 6 * g("rush_td") \
        + g("rec_yds") / 10 + 6 * g("rec_td")
    if fmt == "std":
        return base
    if fmt == "ppr":
        return base + g("rec")
    if fmt == "legends":   # Legends of the Fall: full PPR, first-down premium, -1 per sack
        return base + g("rec") - g("sacks") + 0.5 * g("pass_1d") + g("rush_1d") + g("rec_1d")
    raise ValueError(fmt)


# ---------------------------------------------------------------- tables
def load():
    pbp = pd.concat([pd.read_parquet(f, columns=PBP_COLS) for f in sorted(glob.glob("data/pbp_*.parquet"))])
    return pbp


def player_games(pbp):
    p = pbp[(pbp.two_point_attempt != 1) & pbp.posteam.notna()].copy()
    z = lambda c: p[c].fillna(0)
    p["cmp"] = z("complete_pass")
    dropback = p[p.passer_player_id.notna() & ((p.pass_attempt == 1) | (p.sack == 1))]
    qb = (dropback.assign(att=(dropback.sack != 1).astype(int),
                          cmp=dropback.complete_pass.fillna(0),
                          pass_yds=dropback.passing_yards.fillna(0),
                          pass_td=dropback.pass_touchdown.fillna(0),
                          ints=dropback.interception.fillna(0),
                          sacks=dropback.sack.fillna(0),
                          pass_1d=(dropback.first_down_pass.fillna(0) * dropback.complete_pass.fillna(0)))
          .groupby(["game_id", "posteam", "passer_player_id"], as_index=False)
          [["att", "cmp", "pass_yds", "pass_td", "ints", "sacks", "pass_1d"]].sum()
          .rename(columns={"passer_player_id": "pid"}))
    ru = p[p.rusher_player_id.notna() & (p.rush_attempt == 1) & (p.qb_kneel != 1)]
    rush = (ru.assign(car=1, rush_yds=ru.rushing_yards.fillna(0), rush_td=ru.rush_touchdown.fillna(0),
                      rush_1d=ru.first_down_rush.fillna(0))
            .groupby(["game_id", "posteam", "rusher_player_id"], as_index=False)[["car", "rush_yds", "rush_td", "rush_1d"]].sum()
            .rename(columns={"rusher_player_id": "pid"}))
    rc = p[p.receiver_player_id.notna() & (p.pass_attempt == 1) & (p.sack != 1)]
    recv = (rc.assign(tgt=1, rec=rc.complete_pass.fillna(0), rec_yds=rc.receiving_yards.fillna(0),
                      rec_td=rc.pass_touchdown.fillna(0), rec_1d=rc.first_down_pass.fillna(0) * rc.complete_pass.fillna(0),
                      ay=rc.air_yards.fillna(0))
            .groupby(["game_id", "posteam", "receiver_player_id"], as_index=False)[["tgt", "rec", "rec_yds", "rec_td", "rec_1d", "ay"]].sum()
            .rename(columns={"receiver_player_id": "pid"}))
    pg = qb.merge(rush, on=["game_id", "posteam", "pid"], how="outer").merge(recv, on=["game_id", "posteam", "pid"], how="outer").fillna(0)
    team = pg.groupby(["game_id", "posteam"], as_index=False).agg(tm_att=("att", "sum"), tm_car=("car", "sum"),
                                                                 tm_tgt=("tgt", "sum"), tm_sacks=("sacks", "sum"),
                                                                 tm_ay=("ay", "sum"))
    # pass rate over expected in neutral game states (win probability 20-80%)
    neu = p[p.pass_oe.notna() & p.wp.between(0.2, 0.8)]
    proe = neu.groupby(["game_id", "posteam"], as_index=False).pass_oe.mean().rename(columns={"pass_oe": "proe"})
    team = team.merge(proe, on=["game_id", "posteam"], how="left")
    pg = pg.merge(team, on=["game_id", "posteam"])
    # starting QB = most dropbacks for the team
    pg["db"] = pg.att + pg.sacks
    pg = pg.sort_values("db", ascending=False)
    pg["qb_start"] = (pg.db > 0) & ~pg.duplicated(["game_id", "posteam"]) & (pg.db >= 5)
    # defense allowed (for opponent adjustment)
    dfn = (p.assign(dpy=np.where((p.pass_attempt == 1) & (p.sack != 1), p.passing_yards.fillna(0), 0),
                    dpa=((p.pass_attempt == 1) & (p.sack != 1)).astype(int),
                    dry=np.where((p.rush_attempt == 1) & (p.qb_kneel != 1), p.rushing_yards.fillna(0), 0),
                    dra=((p.rush_attempt == 1) & (p.qb_kneel != 1)).astype(int))
           .groupby(["game_id", "defteam"], as_index=False)[["dpy", "dpa", "dry", "dra"]].sum())
    return pg, team, dfn


def decayed_sum(df, key, cols, decay, date_col="gameday"):
    """Decayed SUMS of cols over each key's prior rows. Adds ds_<col>."""
    df = df.sort_values([key, date_col]).copy()
    vals = {c: np.nan_to_num(df[c].to_numpy(float)) for c in cols}
    out = {c: np.zeros(len(df)) for c in cols}
    for _, idx in df.groupby(key).indices.items():
        acc = {c: 0.0 for c in cols}
        for i in idx:
            for c in cols:
                out[c][i] = acc[c]
                acc[c] = acc[c] * decay + vals[c][i]
    for c in cols:
        df[f"ds_{c}"] = out[c]
    return df


def decayed_carry(df, key, cols, decay, season_carry=1.0, team_carry=1.0, mode="mean", prefix="pr_", date_col="gameday"):
    """Decayed history of cols over each key's PRIOR rows, with an extra multiplier applied to
    everything accumulated before a season boundary (season_carry) or a team change (team_carry).
    mode="mean": NaN-aware weighted mean (adds <prefix><col>); mode="sum": decayed sums (adds ds_<col>)."""
    df = df.sort_values([key, date_col]).copy()
    vals = {c: df[c].to_numpy(float) for c in cols}
    seas = df["season"].to_numpy(); tms = df["posteam"].to_numpy() if "posteam" in df else np.full(len(df), "")
    sc_row = (df["pos"].map(season_carry).fillna(1.0).to_numpy() if isinstance(season_carry, dict)
              else np.full(len(df), float(season_carry)))
    out = {c: np.full(len(df), np.nan if mode == "mean" else 0.0) for c in cols}
    for _, idx in df.groupby(key).indices.items():
        num = {c: 0.0 for c in cols}; den = {c: 0.0 for c in cols}
        ps, pt = None, None
        for i in idx:
            mult = 1.0
            if ps is not None and seas[i] != ps: mult *= sc_row[i]
            if pt is not None and tms[i] != pt: mult *= team_carry
            for c in cols:
                num[c] *= mult; den[c] *= mult
                out[c][i] = (num[c] / den[c] if den[c] > 0 else np.nan) if mode == "mean" else num[c]
                v = vals[c][i]
                num[c] *= decay; den[c] *= decay
                if not np.isnan(v):
                    num[c] += v; den[c] += 1
            ps, pt = seas[i], tms[i]
    for c in cols:
        df[(prefix if mode == "mean" else "ds_") + c] = out[c]
    return df


def roll_forward(df, key, cols, decay, date_col="gameday"):
    """State after each key's last row (for projecting its next game)."""
    last = df.sort_values(date_col).groupby(key).tail(1).copy()
    for c in cols:
        last[f"ds_{c}"] = last[f"ds_{c}"] * decay + np.nan_to_num(last[c].astype(float))
    return last.set_index(key)


# ---------------------------------------------------------------- features
def build(pbp, sched, players, predict_rows=None):
    """Returns one row per (player, game) with raw projection components.
    predict_rows: DataFrame(game_id, posteam, pid, gameday, season, week) for unplayed games."""
    pg, team, dfn = player_games(pbp)
    ctx = pd.concat([sched.assign(team=sched.home_team, opp=sched.away_team, margin=sched.spread_line),
                     sched.assign(team=sched.away_team, opp=sched.home_team, margin=-sched.spread_line)])
    ctx["gameday"] = pd.to_datetime(ctx.gameday)
    outdoor = ~ctx.roof.isin(["dome", "closed"])
    ctx["wind_eff"] = np.where(outdoor, ctx.wind.fillna(7).clip(0, 40), 0.0)        # 7 mph = median outdoor game
    ctx["cold"] = np.where(outdoor, (ctx.temp.fillna(55) < 35).astype(int), 0)
    ctx = ctx[["game_id", "season", "week", "gameday", "team", "opp", "margin", "total_line", "wind_eff", "cold"]]
    ctx["implied"] = ctx.total_line / 2 + ctx.margin / 2
    pos = players.drop_duplicates("gsis_id").set_index("gsis_id").position

    pg = pg.merge(ctx.rename(columns={"team": "posteam"}), on=["game_id", "posteam"])
    pg["pos"] = pg.pid.map(pos).map(GROUP).fillna("OTH")
    try:
        _ol = pd.read_parquet("data/ol_continuity.parquet")[["game_id", "posteam", "ol_missing"]]
        pg = pg.merge(_ol, on=["game_id", "posteam"], how="left")
    except Exception:
        pg["ol_missing"] = np.nan
    pg["_pred"] = False
    _qd = pd.read_parquet("data/team_qd.parquet").rename(columns={"team": "posteam"})[["game_id", "posteam", "qd"]] if os.path.exists("data/team_qd.parquet") else None
    if predict_rows is not None:
        pr = predict_rows.merge(ctx.rename(columns={"team": "posteam"}), on=["game_id", "posteam", "gameday", "season", "week"], how="left")
        pr["pos"] = pr.pid.map(pos).map(GROUP).fillna("OTH")
        pr["_pred"] = True
        if "ol_missing" not in pr: pr["ol_missing"] = np.nan
        try:   # game-day forecasts for this week's outdoor games (forecasts_wk<week>.csv); others keep the median wind
            fc = pd.read_csv(f"forecasts_wk{int(pr.week.iloc[0])}.csv").set_index("team")
            home_of = ctx.drop_duplicates("game_id").set_index("game_id")
            # a forecast is keyed by HOME team; apply it to both teams in that game
            gm_home = sched.set_index("game_id").home_team
            pr_home = pr.game_id.map(gm_home)
            w = pr_home.map(fc.wind); t = pr_home.map(fc.temp)
            pr.loc[w.notna(), "wind_eff"] = w[w.notna()].clip(0, 40)
            pr.loc[t.notna(), "cold"] = (t[t.notna()] < 35).astype(int)
        except FileNotFoundError:
            pass
        pr["qb_start"] = pr.pos == "QB"
        pg = pd.concat([pg, pr], ignore_index=True)
    pg = pg.drop(columns=["qd"], errors="ignore")
    pg["qd"] = pg.merge(_qd, on=["game_id", "posteam"], how="left").qd.fillna(0.0).to_numpy() if _qd is not None else 0.0

    # ---- team volume priors (decayed means per game)
    tm = team.merge(ctx.rename(columns={"team": "posteam"}), on=["game_id", "posteam"])
    tm["one"] = 1.0
    tm["proe_n"] = tm.proe.notna().astype(float); tm["proe"] = tm.proe.fillna(0)
    tm = decayed_carry(tm.assign(season=tm.season), "posteam", ["tm_att", "tm_car", "tm_tgt", "tm_sacks", "one", "proe", "proe_n"],
                       TEAM_DECAY, season_carry=TEAM_VOL_CARRY, mode="sum")
    for c in ["tm_att", "tm_car", "tm_tgt", "tm_sacks"]:
        tm[f"pg_{c}"] = tm[f"ds_{c}"] / tm.ds_one.replace(0, np.nan)
    tm["pg_proe"] = (tm.ds_proe / (tm.ds_proe_n + 2)).fillna(0)     # shrink toward 0 (league-neutral)
    # defense allowed per pass att / carry, and defense faced volume
    dd = dfn.merge(ctx.rename(columns={"team": "defteam"}), on=["game_id", "defteam"])
    dd["one"] = 1.0
    dd = decayed_carry(dd, "defteam", ["dpy", "dpa", "one"], DEF_DECAY, season_carry=DEF_SEASON_CARRY, mode="sum")
    _r = decayed_carry(dd[["game_id", "defteam", "season", "gameday", "dry", "dra"]], "defteam", ["dry", "dra"], DEF_DECAY,
                       season_carry=DEF_RUSH_SEASON_CARRY, mode="sum")
    dd = dd.merge(_r[["game_id", "defteam", "ds_dry", "ds_dra"]], on=["game_id", "defteam"])
    lg_ypa = dfn.dpy.sum() / dfn.dpa.sum(); lg_ypc = dfn.dry.sum() / dfn.dra.sum()
    dd["def_pass_ratio"] = ((dd.ds_dpy + DEF_K_PASS * lg_ypa) / (dd.ds_dpa + DEF_K_PASS)) / lg_ypa
    dd["def_rush_ratio"] = ((dd.ds_dry + DEF_K_RUSH * lg_ypc) / (dd.ds_dra + DEF_K_RUSH)) / lg_ypc
    dd["def_att_pg"] = dd.ds_dpa / dd.ds_one.replace(0, np.nan)
    dd["def_car_pg"] = dd.ds_dra / dd.ds_one.replace(0, np.nan)

    tcols = ["pg_tm_att", "pg_tm_car", "pg_tm_tgt", "pg_tm_sacks", "pg_proe"]
    dcols = ["def_pass_ratio", "def_rush_ratio", "def_att_pg", "def_car_pg"]
    pg = pg.merge(tm[["game_id", "posteam"] + tcols], on=["game_id", "posteam"], how="left")
    pg = pg.merge(dd[["game_id", "defteam"] + dcols].rename(columns={"defteam": "opp"}), on=["game_id", "opp"], how="left")
    if predict_rows is not None:
        tl = roll_forward(tm, "posteam", ["tm_att", "tm_car", "tm_tgt", "tm_sacks", "one", "proe", "proe_n"], TEAM_DECAY)
        for c in ["tm_att", "tm_car", "tm_tgt", "tm_sacks"]:
            tl[f"pg_{c}"] = tl[f"ds_{c}"] / tl.ds_one
        tl["pg_proe"] = (tl.ds_proe / (tl.ds_proe_n + 2)).fillna(0)
        dl = roll_forward(dd, "defteam", ["dpy", "dpa", "dry", "dra", "one"], DEF_DECAY)
        dl["def_pass_ratio"] = ((dl.ds_dpy + DEF_K_PASS * lg_ypa) / (dl.ds_dpa + DEF_K_PASS)) / lg_ypa
        dl["def_rush_ratio"] = ((dl.ds_dry + DEF_K_RUSH * lg_ypc) / (dl.ds_dra + DEF_K_RUSH)) / lg_ypc
        dl["def_att_pg"] = dl.ds_dpa / dl.ds_one; dl["def_car_pg"] = dl.ds_dra / dl.ds_one
        m = pg._pred
        for c in tcols:
            pg.loc[m, c] = pg.loc[m, "posteam"].map(tl[c])
        for c in dcols:
            pg.loc[m, c] = pg.loc[m, "opp"].map(dl[c])

    # ---- player shares (decayed mean of per-game shares) and efficiency (decayed sums)
    pg["tgt_sh"] = np.where(pg.tm_tgt > 0, pg.tgt / pg.tm_tgt, np.nan)
    pg["car_sh"] = np.where(pg.tm_car > 0, pg.car / pg.tm_car, np.nan)
    pg["ay_sh"] = np.where(pg.tm_ay > 0, (pg.ay / pg.tm_ay).clip(-0.5, 1.5), np.nan)
    snaps = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_pct"]).drop_duplicates(["game_id", "gsis_id"])
    if snaps.offense_pct.max() > 1.5:
        snaps["offense_pct"] = snaps.offense_pct / 100
    pg = pg.merge(snaps.rename(columns={"gsis_id": "pid", "offense_pct": "snap"}), on=["game_id", "pid"], how="left")
    pg = T.decayed_mean(pg, "pid", ["tgt_sh", "car_sh", "ay_sh", "snap"], SHARE_DECAY)
    if SEASON_CARRY != 1.0 or TEAM_CARRY != 1.0 or isinstance(SEASON_CARRY, dict):
        n_keep = pg[["pid", "game_id", "pr_n"]]
        pg = decayed_carry(pg.drop(columns=["pr_tgt_sh", "pr_car_sh", "pr_ay_sh", "pr_snap"]), "pid",
                           ["tgt_sh", "car_sh", "ay_sh", "snap"], SHARE_DECAY, SEASON_CARRY, TEAM_CARRY)
    last = T.decayed_mean(pg[["pid", "gameday", "game_id", "snap"]].copy(), "pid", ["snap"], 0.001)
    pg = pg.merge(last[["pid", "game_id", "pr_snap"]].rename(columns={"pr_snap": "last_snap"}), on=["pid", "game_id"], how="left")
    pos_snap = pg.groupby("pos").snap.mean()
    fill = pg.pos.map(pos_snap).fillna(0.5)
    pg["l_snap"] = np.log(pg.pr_snap.fillna(fill).clip(0.02, 1))
    pg["snap_trend"] = (pg.last_snap.fillna(pg.pr_snap).fillna(fill) - pg.pr_snap.fillna(fill)).clip(-0.6, 0.6)
    pg["snap_dn"] = pg.snap_trend.clip(upper=0)
    pg["l_ay_sh"] = np.log(pg.pr_ay_sh.fillna(0).clip(0, 1) + 0.01)
    pg["l_wopr"] = np.log((1.5 * pg.pr_tgt_sh.fillna(0) + 0.7 * pg.pr_ay_sh.fillna(0).clip(0, 1)).clip(0, None) + 0.01)
    # receiving yards per target allowed by each defense to each position (RB / WR / TE)
    h = pg[~pg._pred & pg.pos.isin(["RB", "WR", "TE"])].groupby(["game_id", "opp", "pos", "gameday"], as_index=False)[["rec_yds", "tgt"]].sum()
    h["key"] = h.opp + "|" + h.pos
    h = decayed_sum(h, "key", ["rec_yds", "tgt"], DEF_DECAY)
    lgp = (h.groupby("pos").rec_yds.sum() / h.groupby("pos").tgt.sum())
    h["def_pos_ratio"] = ((h.ds_rec_yds + POS_DEF_K * h.pos.map(lgp)) / (h.ds_tgt + POS_DEF_K)) / h.pos.map(lgp)
    pg = pg.merge(h[["game_id", "opp", "pos", "def_pos_ratio"]], on=["game_id", "opp", "pos"], how="left")
    if predict_rows is not None:
        hl = roll_forward(h, "key", ["rec_yds", "tgt"], DEF_DECAY)
        hl["r"] = ((hl.ds_rec_yds + POS_DEF_K * hl.pos.map(lgp)) / (hl.ds_tgt + POS_DEF_K)) / hl.pos.map(lgp)
        m = pg._pred
        pg.loc[m, "def_pos_ratio"] = (pg.loc[m, "opp"] + "|" + pg.loc[m, "pos"]).map(hl.r)
    pg["def_pos_ratio"] = pg.def_pos_ratio.fillna(1.0)
    eff = ["tgt", "rec", "rec_yds", "rec_1d", "car", "rush_yds", "rush_1d", "att", "cmp", "pass_yds", "sacks", "pass_1d", "ints"]
    pg = decayed_carry(pg, "pid", eff, EFF_DECAY, season_carry=EFF_SEASON_CARRY, mode="sum")
    # QB attempts per start (decayed)
    # pressure rates entering each game: the opponent defense's sacks+hits per dropback (shrunk, decayed), and this
    # offense's rate allowed. Tested across QB stats and the team model (tests/t_pressure.py): helps SACKS only
    # (MAE 1.313 -> 1.306, coef +0.44); pass yards/completions and team margins already carry it through yardage ratios.
    _pr = pbp[pbp.qb_dropback == 1].copy() if "qb_hit" in pbp else None
    if _pr is not None:
        _pr["prs"] = ((_pr.sack.fillna(0) + _pr.qb_hit.fillna(0)) > 0).astype(float)
        _LG = _pr.prs.mean(); _K = 300
        def _prate(df, key, out):
            df = df.groupby(["game_id", key]).agg(db=("prs", "size"), pr=("prs", "sum")).reset_index()
            df = df.merge(sched[["game_id", "gameday"]], on="game_id"); df["gameday"] = pd.to_datetime(df.gameday); df = df.sort_values("gameday")
            rows = []; last = {}
            for t, gdf in df.groupby(key):
                cdb = cpr = 0.0
                for r in gdf.itertuples():
                    rows.append((r.game_id, t, (cpr + _K * _LG) / (cdb + _K)))
                    cdb = cdb * 0.95 + r.db; cpr = cpr * 0.95 + r.pr
                last[t] = (cpr + _K * _LG) / (cdb + _K)
            return pd.DataFrame(rows, columns=["game_id", key, out]), last
        _rd, _rd_last = _prate(_pr, "defteam", "opp_prs"); _ro, _ro_last = _prate(_pr, "posteam", "own_prs")
        pg = pg.merge(_rd.rename(columns={"defteam": "opp"}), on=["game_id", "opp"], how="left")
        pg = pg.merge(_ro, on=["game_id", "posteam"], how="left")
        m = pg._pred
        pg.loc[m, "opp_prs"] = pg.loc[m, "opp_prs"].fillna(pg.loc[m, "opp"].map(_rd_last))
        pg.loc[m, "own_prs"] = pg.loc[m, "own_prs"].fillna(pg.loc[m, "posteam"].map(_ro_last))
        pg["opp_prs"] = (pg.opp_prs.fillna(_LG) - _LG) * 10; pg["own_prs"] = (pg.own_prs.fillna(_LG) - _LG) * 10
    else:
        pg["opp_prs"] = 0.0; pg["own_prs"] = 0.0
    pg["start_att"] = np.where(pg.qb_start & ~pg._pred, pg.att, np.nan)
    pg["start_db"] = np.where(pg.qb_start & ~pg._pred, pg.att + pg.sacks, np.nan)
    pg = T.decayed_mean(pg, "pid", ["start_att", "start_db"], 0.85)
    return pg


def efficiency(pg, posmeans):
    """Shrunk efficiency rates. posmeans: dict of position-group averages from training data."""
    d = pg
    pm = lambda k: d.pos.map(posmeans[k]).fillna(np.nanmean(list(posmeans[k].values())))
    d["e_catch"] = (d.ds_rec + K["catch"] * pm("catch")) / (d.ds_tgt + K["catch"])
    d["e_ypt"] = (d.ds_rec_yds + K["ypt"] * pm("ypt")) / (d.ds_tgt + K["ypt"])
    d["e_rec1d"] = (d.ds_rec_1d + K["rec1d"] * pm("rec1d")) / (d.ds_rec + K["rec1d"])
    d["e_ypc"] = (d.ds_rush_yds + K["ypc"] * pm("ypc")) / (d.ds_car + K["ypc"])
    d["e_rush1d"] = (d.ds_rush_1d + K["rush1d"] * pm("rush1d")) / (d.ds_car + K["rush1d"])
    lg = posmeans["qb"]
    d["e_cmp"] = (d.ds_cmp + K["cmp"] * lg["cmp"]) / (d.ds_att + K["cmp"])
    d["e_ypa"] = (d.ds_pass_yds + K["ypa"] * lg["ypa"]) / (d.ds_att + K["ypa"])
    d["e_sack"] = (d.ds_sacks + K["sack"] * lg["sack"]) / (d.ds_att + d.ds_sacks + K["sack"])
    d["e_pass1d"] = (d.ds_pass_1d + K["pass1d"] * lg["pass1d"]) / (d.ds_cmp + K["pass1d"])
    d["e_int"] = (d.ds_ints + K["qbint"] * lg["int"]) / (d.ds_att + K["qbint"])
    return d


def position_means(pg):
    h = pg[~pg._pred]
    g = h.groupby("pos")[["rec", "tgt", "rec_yds", "rec_1d", "car", "rush_yds", "rush_1d"]].sum()
    pm = dict(catch=(g.rec / g.tgt).to_dict(), ypt=(g.rec_yds / g.tgt).to_dict(), rec1d=(g.rec_1d / g.rec).to_dict(),
              ypc=(g.rush_yds / g.car).to_dict(), rush1d=(g.rush_1d / g.car).to_dict())
    q = h[h.qb_start]
    pm["qb"] = dict(cmp=q.cmp.sum() / q.att.sum(), ypa=q.pass_yds.sum() / q.att.sum(),
                    sack=q.sacks.sum() / (q.att.sum() + q.sacks.sum()), pass1d=q.pass_1d.sum() / q.cmp.sum(),
                    int=q.ints.sum() / q.att.sum())
    return pm


# ---------------------------------------------------------------- volume + raw projections
VOL_X = ["total_line", "margin", "pg_tm_att", "pg_tm_car", "def_att_pg", "def_car_pg"]
vol_x = lambda: VOL_X + (["pg_proe"] if USE_PROE else []) + (["wind_eff", "cold"] if USE_WEATHER else [])


def fit_volume(team_rows):
    tr = team_rows.dropna(subset=vol_x() + ["tm_att", "tm_car"])
    X = sm.add_constant(tr[vol_x()])
    return sm.OLS(tr.tm_att, X).fit(), sm.OLS(tr.tm_car, X).fit()


def raw_projection(d, vol_att, vol_car):
    X = sm.add_constant(d[vol_x()].fillna(d[vol_x()].median()), has_constant="add")
    d["v_att"] = vol_att.predict(X).clip(18, 55)
    d["v_car"] = vol_car.predict(X).clip(12, 45)
    d["v_tgt"] = d.v_att * 0.985          # a few attempts have no intended receiver
    ts, cs = d.pr_tgt_sh.fillna(0), d.pr_car_sh.fillna(0)
    d["r_tgt"] = d.v_tgt * ts
    d["r_rec"] = d.r_tgt * d.e_catch
    d["r_rec_yds"] = d.r_tgt * d.e_ypt
    d["r_rec_1d"] = d.r_rec * d.e_rec1d
    d["r_car"] = d.v_car * cs
    d["r_rush_yds"] = d.r_car * d.e_ypc
    d["r_rush_1d"] = d.r_car * d.e_rush1d
    qb = d.pos == "QB"
    att = np.where(qb, d.v_att * 0.5 + d.pr_start_att.fillna(d.v_att) * 0.5, 0)
    d["r_att"] = att
    d["r_cmp"] = att * d.e_cmp
    d["r_pass_yds"] = att * d.e_ypa
    d["r_sacks"] = att * d.e_sack / (1 - d.e_sack)
    d["r_pass_1d"] = d.r_cmp * d.e_pass1d
    d["r_ints"] = att * d.e_int
    return d


# stat -> (raw column, opponent ratio column or None)
STATS = {"tgt": ("r_tgt", None), "rec": ("r_rec", None), "rec_yds": ("r_rec_yds", "def_pass_ratio"),
         "rec_1d": ("r_rec_1d", "def_pass_ratio"), "car": ("r_car", None), "rush_yds": ("r_rush_yds", "def_rush_ratio"),
         "rush_1d": ("r_rush_1d", "def_rush_ratio"), "att": ("r_att", None), "cmp": ("r_cmp", None),
         "pass_yds": ("r_pass_yds", "def_pass_ratio"), "sacks": ("r_sacks", None), "pass_1d": ("r_pass_1d", "def_pass_ratio"),
         "ints": ("r_ints", None)}
QB_ONLY = {"att", "cmp", "pass_yds", "sacks", "pass_1d", "ints"}


def def_col(st, grp, dcol):
    if not USE_DEF or not dcol:
        return None
    if POS_DEF and grp in ("RB", "WR", "TE") and st in ("rec_yds", "rec_1d"):
        return "def_pos_ratio"
    return dcol


def fit_calibration(tr, use_def=True):
    """Per (stat, position group): Poisson GLM  actual ~ log(raw) [+ log(def ratio)]."""
    models = {}
    for st, (raw, dcol) in STATS.items():
        for grp in (["QB"] if st in QB_ONLY else ["QB", "RB", "WR", "TE"]):
            rows = tr[(tr.pos == grp) & (tr[raw] > 0.05)]
            if grp == "QB":
                rows = rows[rows.qb_start]
            if len(rows) < 200:
                continue
            X = pd.DataFrame({"lraw": np.log(rows[raw])})
            dc = def_col(st, grp, dcol)
            if use_def and dc:
                X["ldef"] = np.log(rows[dc].fillna(1.0))
            if grp != "QB":
                for c in EXTRA.get(st, []):
                    X[c] = rows[c].fillna(0)
            if USE_WEATHER and st in WEATHER_STATS:
                X["wind_eff"] = rows.wind_eff.fillna(0); X["cold"] = rows.cold.fillna(0)
            if USE_OL and grp == "QB" and st in ("pass_yds", "sacks", "cmp", "ints", "pass_1d"):
                X["ol_missing"] = rows.ol_missing.fillna(0.9)
            if grp == "QB" and st == "sacks":
                X["opp_prs"] = rows.opp_prs.fillna(0); X["own_prs"] = rows.own_prs.fillna(0)
            models[(st, grp)] = sm.GLM(rows[st], sm.add_constant(X), family=sm.families.Poisson()).fit()
    return models


def apply_calibration(d, models, use_def=True):
    for st, (raw, dcol) in STATS.items():
        d[f"p_{st}"] = 0.0
        for grp in ["QB", "RB", "WR", "TE"]:
            m = models.get((st, grp))
            sel = (d.pos == grp) & (d[raw] > 0.05)
            if m is None or not sel.any():
                continue
            X = pd.DataFrame({"lraw": np.log(d.loc[sel, raw])})
            dc = def_col(st, grp, dcol)
            if use_def and dc:
                X["ldef"] = np.log(d.loc[sel, dc].fillna(1.0))
            if grp != "QB":
                for c in EXTRA.get(st, []):
                    X[c] = d.loc[sel, c].fillna(0)
            if USE_WEATHER and st in WEATHER_STATS:
                X["wind_eff"] = d.loc[sel, "wind_eff"].fillna(0); X["cold"] = d.loc[sel, "cold"].fillna(0)
            if USE_OL and grp == "QB" and st in ("pass_yds", "sacks", "cmp", "ints", "pass_1d"):
                X["ol_missing"] = d.loc[sel, "ol_missing"].fillna(0.9)
            if grp == "QB" and st == "sacks":
                X["opp_prs"] = d.loc[sel, "opp_prs"].fillna(0); X["own_prs"] = d.loc[sel, "own_prs"].fillna(0)
            d.loc[sel, f"p_{st}"] = m.predict(sm.add_constant(X, has_constant="add")).to_numpy()
    return d


def fantasy(d, prefix="p_"):
    cols = ["pass_yds", "pass_td", "ints", "sacks", "pass_1d", "rush_yds", "rush_td", "rush_1d",
            "rec", "rec_yds", "rec_td", "rec_1d"]
    s = pd.DataFrame({c: d[prefix + c] if prefix + c in d else 0 for c in cols})
    return {f: scoring(s, f) for f in ["legends", "ppr", "std"]}


# ---------------------------------------------------------------- outcome ranges
RANGE_STATS = ["rec", "rec_yds", "rush_yds", "car", "tgt", "pass_yds", "cmp", "att", "fp_legends", "fp_ppr", "fp_std"]


def ratio_table(oos):
    """Empirical actual/projected ratio quantiles by (stat, pos, projection tercile) from out-of-sample rows."""
    tab = {}
    for st in RANGE_STATS:
        pcol, acol = (f"p_{st}", st) if not st.startswith("fp_") else (f"p_{st}", f"a_{st}")
        for grp in ["QB", "RB", "WR", "TE"]:
            rows = oos[(oos.pos == grp) & (oos[pcol] > 0.5)]
            if grp == "QB":
                rows = rows[rows.qb_start]
            if len(rows) < 300:
                continue
            edges = np.quantile(rows[pcol], [0, 1 / 3, 2 / 3, 1])
            for b in range(3):
                r = rows[(rows[pcol] >= edges[b]) & (rows[pcol] <= edges[b + 1])]
                ratio = (r[acol] / r[pcol]).to_numpy()
                tab[(st, grp, b)] = dict(lo=float(edges[b]), hi=float(edges[b + 1]),
                                         q=np.quantile(ratio, QUANTS).tolist(), ratios=ratio)
    return tab


def quantiles_for(tab, st, grp, mu):
    keys = [k for k in tab if k[0] == st and k[1] == grp]
    if not keys or not mu or mu <= 0.5:
        return None
    b = 0 if mu < tab[(st, grp, 1)]["lo"] else (1 if mu < tab[(st, grp, 2)]["lo"] else 2)
    return [mu * x for x in tab[(st, grp, b)]["q"]]


def p_over(tab, st, grp, mu, line):
    keys = [k for k in tab if k[0] == st and k[1] == grp]
    if not keys or mu <= 0.5:
        return np.nan
    b = 0 if mu < tab[(st, grp, 1)]["lo"] else (1 if mu < tab[(st, grp, 2)]["lo"] else 2)
    return float(np.mean(mu * tab[(st, grp, b)]["ratios"] > line))
