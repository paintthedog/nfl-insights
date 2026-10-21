"""
Spread & total model.

Team ratings: decayed EPA/play on offense and defense, pass and rush, from play-by-play (prior games only).
Game model (OLS, walk-forward): margin = f(rating gaps, QB change, rest, home field, wind);
                                total  = f(rating sums, pace, pass rates, wind, roof).
Also: how the market would price it, so the output is "model number vs line" with cover probability.
"""
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

DECAY = 0.92          # per-game memory for team ratings (~8-game half-life)
SEASON_CARRY = 0.5    # last season counts half once the new one starts
K_RATE = 300          # pseudo-plays of league average pulled into each rating
MARGIN_SD = 13.3      # historical sd of (result - model margin); refined in backtest
TOTAL_SD = 13.1
OPP_ADJ = 0.0       # strength-of-schedule adjustment (0 = off, 1 = full): each game's output is judged against the opponent's rating going in
OPP_ITERS = 3

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "play_type", "epa",
            "pass", "rush", "qb_dropback", "passer_player_id", "sack", "pass_attempt", "wp", "two_point_attempt"]


def load():
    pbp = pd.concat([pd.read_parquet(f, columns=PBP_COLS) for f in sorted(glob.glob("data/pbp_*.parquet"))])
    sched = pd.read_parquet("data/sched.parquet")
    sched = sched[(sched.season >= 2016) & (sched.game_type == "REG")].copy()
    return pbp, sched


def team_games(pbp):
    p = pbp[(pbp.season_type == "REG") & pbp.posteam.notna() & pbp.epa.notna() & (pbp.two_point_attempt != 1)
            & pbp.play_type.isin(["pass", "run"])].copy()
    p = p.reset_index(drop=True)
    p["is_pass"] = (p.play_type == "pass").astype(int)
    p["pass_epa_v"] = p.epa * p.is_pass; p["rush_epa_v"] = p.epa * (1 - p.is_pass)
    p["neutral"] = p.wp.between(0.2, 0.8).astype(int); p["neutral_pass_v"] = p.is_pass * p.neutral
    g = p.groupby(["game_id", "posteam"]).agg(
        off_epa=("epa", "sum"), plays=("epa", "size"), pass_epa=("pass_epa_v", "sum"), pass_n=("is_pass", "sum"),
        rush_epa=("rush_epa_v", "sum"), neu_n=("neutral", "sum"), neu_pass=("neutral_pass_v", "sum"),
    ).reset_index()
    g["neutral_pass"] = np.where(g.neu_n > 0, g.neu_pass / g.neu_n.replace(0, np.nan), np.nan)
    g = g.drop(columns=["neu_n", "neu_pass"])
    g["rush_n"] = g.plays - g.pass_n
    # defense = what the opponent did on offense in this game
    d = g.rename(columns={"posteam": "defteam", "off_epa": "def_epa_allowed", "pass_epa": "def_pass_allowed",
                          "rush_epa": "def_rush_allowed", "plays": "def_plays", "pass_n": "def_pass_n", "rush_n": "def_rush_n"})
    d = d.drop(columns=["neutral_pass"])
    teams_in_game = g.groupby("game_id").posteam.apply(list)
    d["posteam"] = [[t for t in teams_in_game[gid] if t != dt][0] if len(teams_in_game[gid]) == 2 else None
                    for gid, dt in zip(d.game_id, d.defteam)]
    g = g.merge(d.drop(columns=["defteam"]), on=["game_id", "posteam"])
    # starting QB = most dropbacks
    q = p[p.passer_player_id.notna()].groupby(["game_id", "posteam", "passer_player_id"]).size().reset_index(name="db")
    q = q.sort_values("db", ascending=False).drop_duplicates(["game_id", "posteam"])
    g = g.merge(q[["game_id", "posteam", "passer_player_id"]].rename(columns={"passer_player_id": "qb"}), on=["game_id", "posteam"], how="left")
    return g


def decayed(df, key, cols, decay, carry, date_col="gameday"):
    """Decayed sums over PRIOR rows, extra discount at season boundaries. Adds ds_<col>."""
    df = df.sort_values([key, date_col]).copy()
    vals = {c: np.nan_to_num(df[c].to_numpy(float)) for c in cols}
    seas = df.season.to_numpy()
    out = {c: np.zeros(len(df)) for c in cols}
    for _, idx in df.groupby(key).indices.items():
        acc = {c: 0.0 for c in cols}; ps = None
        for i in idx:
            if ps is not None and seas[i] != ps:
                for c in cols: acc[c] *= carry
            for c in cols:
                out[c][i] = acc[c]; acc[c] = acc[c] * decay + vals[c][i]
            ps = seas[i]
    for c in cols: df["ds_" + c] = out[c]
    return df


def build(pbp, sched, predict_games=None):
    """One row per game (home perspective) with ratings from prior games only.
    predict_games: schedule rows for unplayed games (same columns)."""
    tg = team_games(pbp)
    s = sched.copy()
    if predict_games is not None:
        s = pd.concat([s, predict_games.assign(_pred=True)], ignore_index=True)
    s["_pred"] = s.get("_pred", False); s["_pred"] = s._pred.fillna(False).astype(bool)
    s["gameday"] = pd.to_datetime(s.gameday)
    # long form: one row per team-game, with stats (NaN for unplayed)
    long = pd.concat([s.assign(team=s.home_team, opp=s.away_team, home=1), s.assign(team=s.away_team, opp=s.home_team, home=0)])
    long = long[["game_id", "season", "week", "gameday", "team", "opp", "home", "_pred"]]
    long = long.merge(tg.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")
    long["games"] = np.where(long._pred, 0.0, 1.0)
    cols = ["off_epa", "plays", "pass_epa", "pass_n", "rush_epa", "rush_n", "def_epa_allowed", "def_plays",
            "def_pass_allowed", "def_pass_n", "def_rush_allowed", "def_rush_n", "games"]
    long["np_sum"] = long.neutral_pass; long["np_n"] = long.neutral_pass.notna().astype(float)
    lg_off = tg.off_epa.sum() / tg.plays.sum(); lg_pass = tg.pass_epa.sum() / tg.pass_n.sum(); lg_rush = tg.rush_epa.sum() / tg.rush_n.sum()
    def rate(L):
        L = decayed(L, "team", cols + ["np_sum", "np_n"], DECAY, SEASON_CARRY)
        L["r_off"] = (L.ds_off_epa + K_RATE * lg_off) / (L.ds_plays + K_RATE)
        L["r_pass"] = (L.ds_pass_epa + K_RATE * lg_pass) / (L.ds_pass_n + K_RATE)
        L["r_rush"] = (L.ds_rush_epa + K_RATE * lg_rush) / (L.ds_rush_n + K_RATE)
        L["r_def"] = (L.ds_def_epa_allowed + K_RATE * lg_off) / (L.ds_def_plays + K_RATE)        # lower = better defense
        L["r_dpass"] = (L.ds_def_pass_allowed + K_RATE * lg_pass) / (L.ds_def_pass_n + K_RATE)
        L["r_drush"] = (L.ds_def_rush_allowed + K_RATE * lg_rush) / (L.ds_def_rush_n + K_RATE)
        return L
    long = rate(long)
    if OPP_ADJ:
        # strength of schedule: credit each past game's output for who it came against, using the opponent's rating
        # GOING INTO that game (prior games only, so nothing leaks), then re-rate; repeat so the opponents' own ratings settle
        raw = {c: long[c].copy() for c in ("off_epa", "pass_epa", "rush_epa", "def_epa_allowed", "def_pass_allowed", "def_rush_allowed")}
        for _ in range(OPP_ITERS):
            o = long[["game_id", "team", "r_off", "r_pass", "r_rush", "r_def", "r_dpass", "r_drush"]].rename(columns={"team": "opp"})
            o = o.rename(columns={c: "o_" + c for c in o.columns if c.startswith("r_")})
            m = long[["game_id", "opp"]].merge(o, on=["game_id", "opp"], how="left").fillna({"o_r_off": lg_off, "o_r_pass": lg_pass, "o_r_rush": lg_rush, "o_r_def": lg_off, "o_r_dpass": lg_pass, "o_r_drush": lg_rush})
            m.index = long.index
            long["off_epa"] = raw["off_epa"] - OPP_ADJ * long.plays * (m.o_r_def - lg_off)
            long["pass_epa"] = raw["pass_epa"] - OPP_ADJ * long.pass_n * (m.o_r_dpass - lg_pass)
            long["rush_epa"] = raw["rush_epa"] - OPP_ADJ * long.rush_n * (m.o_r_drush - lg_rush)
            long["def_epa_allowed"] = raw["def_epa_allowed"] - OPP_ADJ * long.def_plays * (m.o_r_off - lg_off)
            long["def_pass_allowed"] = raw["def_pass_allowed"] - OPP_ADJ * long.def_pass_n * (m.o_r_pass - lg_pass)
            long["def_rush_allowed"] = raw["def_rush_allowed"] - OPP_ADJ * long.def_rush_n * (m.o_r_rush - lg_rush)
            long = rate(long.drop(columns=[c for c in long.columns if c.startswith("ds_") or c.startswith("r_")]))
    long["pace"] = (long.ds_plays + long.ds_def_plays) / (2 * long.ds_games.replace(0, np.nan))        # plays per game, both sides
    long["pass_rate"] = long.ds_np_sum / long.ds_np_n.replace(0, np.nan)
    long["n_games"] = long.ds_games
    # QB continuity: is this week's starter the same as last game's? (for upcoming games, use the schedule's listed QB)
    long = long.sort_values(["team", "gameday"])
    long["prev_qb"] = long.groupby("team").qb.shift()
    long["prev_qb"] = long.groupby("team").prev_qb.ffill()
    long["prev_qb_games"] = long.groupby("team").qb.transform(lambda s: s.shift().groupby((s.shift() != s.shift().shift()).cumsum()).cumcount() + 1)
    sched_qb = pd.concat([s[["game_id", "home_team", "home_qb_id"]].rename(columns={"home_team": "team", "home_qb_id": "listed_qb"}),
                         s[["game_id", "away_team", "away_qb_id"]].rename(columns={"away_team": "team", "away_qb_id": "listed_qb"})])
    long = long.merge(sched_qb, on=["game_id", "team"], how="left")
    long["this_qb"] = np.where(long._pred, long.listed_qb, long.qb)
    long["qb_change"] = ((long.this_qb != long.prev_qb) & long.prev_qb.notna() & long.this_qb.notna()).astype(int)

    keep = ["game_id", "team", "r_off", "r_pass", "r_rush", "r_def", "r_dpass", "r_drush", "pace", "pass_rate", "n_games", "qb_change", "this_qb"]
    h = long[long.home == 1][keep].rename(columns={c: "h_" + c for c in keep if c not in ("game_id",)})
    a = long[long.home == 0][keep].rename(columns={c: "a_" + c for c in keep if c not in ("game_id",)})
    g = s.merge(h, on="game_id").merge(a, on="game_id")
    for side, rest in [("h", "home_rest"), ("a", "away_rest")]:
        g[f"{side}_short"] = (g[rest] <= 5).astype(int); g[f"{side}_bye"] = (g[rest] >= 13).astype(int)
    outdoor = ~g.roof.isin(["dome", "closed"])
    g["wind_eff"] = np.where(outdoor, g.wind.fillna(8).clip(0, 40), 0)
    g["dome"] = (~outdoor).astype(int)
    g["cold"] = np.where(outdoor, (g.temp.fillna(55) < 32).astype(int), 0)
    # features
    g["off_gap"] = g.h_r_off - g.a_r_off; g["def_gap"] = g.a_r_def - g.h_r_def      # positive = home better
    g["pass_gap"] = g.h_r_pass - g.a_r_pass; g["rush_gap"] = g.h_r_rush - g.a_r_rush
    g["dpass_gap"] = g.a_r_dpass - g.h_r_dpass; g["drush_gap"] = g.a_r_drush - g.h_r_drush
    g["qb_gap"] = g.a_qb_change - g.h_qb_change
    g["rest_gap"] = (g.home_rest - g.away_rest).clip(-7, 7)
    g["off_sum"] = g.h_r_off + g.a_r_off; g["def_sum"] = g.h_r_def + g.a_r_def
    g["pace_sum"] = g.h_pace + g.a_pace; g["pass_sum"] = g.h_pass_rate.fillna(0.55) + g.a_pass_rate.fillna(0.55)
    g["qb_any"] = g.h_qb_change + g.a_qb_change
    g["min_games"] = np.minimum(g.h_n_games, g.a_n_games)
    import team_adj
    g = team_adj.add(g, pbp, s, tg)          # quarterback quality and missing defensive regulars
    return g


MARGIN_X = ["off_gap", "def_gap", "qb_gap", "rest_gap", "h_short", "a_short", "wind_eff", "qd_gap", "dmiss_gap"]
TOTAL_X = ["off_sum", "def_sum", "pace_sum", "pass_sum", "wind_eff", "dome", "cold", "qb_any", "qd_sum", "dmiss_sum"]


def fit(train):
    tr = train[train.result.notna() & (train.min_games >= 3)]
    mm = sm.OLS(tr.result, sm.add_constant(tr[MARGIN_X])).fit()
    mt = sm.OLS(tr.total, sm.add_constant(tr[TOTAL_X])).fit()
    return mm, mt


def predict(g, mm, mt):
    g = g.copy()
    g["m_margin"] = mm.predict(sm.add_constant(g[MARGIN_X].fillna(0), has_constant="add"))
    g["m_total"] = mt.predict(sm.add_constant(g[TOTAL_X].fillna(g[TOTAL_X].median()), has_constant="add"))
    return g


def cover_prob(model_margin, spread, sd=MARGIN_SD):
    """P(home covers): home margin > spread_line (nflverse: positive = home favored)."""
    from scipy.stats import norm
    return 1 - norm.cdf((spread - model_margin) / sd)


def over_prob(model_total, total_line, sd=TOTAL_SD):
    from scipy.stats import norm
    return 1 - norm.cdf((total_line - model_total) / sd)
