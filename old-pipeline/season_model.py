"""Season-long projections shared by the SEASON and AWARDS sections.
Teams: three ratings per team (what the betting market has priced them at, how they have actually played, and last season), combined by a
regression fitted on 2016-2025 to predict each remaining game; the remaining schedule is then simulated.
Players: rest-of-season production per remaining team game from this season's rate, recent workload, last season, draft slot and games
missed, fitted on 2017-2025 and tested by leaving each season out."""
import json, glob, os, numpy as np, pandas as pd, warnings
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
SEASON, FIRST = 2026, 2016
PL = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
S = pd.read_parquet("data/sched.parquet"); S = S[(S.game_type == "REG") & (S.season >= FIRST - 1)].copy()
if os.path.exists("coach_fix.json"):
    for yr, fx in json.load(open("coach_fix.json")).items():
        for tm, c in fx.items():
            S.loc[(S.season == int(yr)) & (S.home_team == tm), "home_coach"] = c; S.loc[(S.season == int(yr)) & (S.away_team == tm), "away_coach"] = c
THRU = int(S[(S.season == SEASON) & S.result.notna()].week.max())
C = ["season", "week", "season_type", "game_id", "posteam", "defteam", "play_type", "two_point_attempt", "passer_player_id", "pass_attempt", "sack", "passing_yards", "pass_touchdown", "interception", "qb_epa", "qb_dropback",
     "rusher_player_id", "rushing_yards", "rush_touchdown", "receiver_player_id", "receiving_yards", "complete_pass", "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
     "tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id", "qb_hit_1_player_id", "qb_hit_2_player_id", "interception_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id",
     "forced_fumble_player_1_player_id", "forced_fumble_player_2_player_id", "solo_tackle_1_player_id", "solo_tackle_2_player_id", "assist_tackle_1_player_id", "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id",
     "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id", "td_player_id", "td_team", "touchdown"]
P = pd.concat([pd.read_parquet(f, columns=C) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
P = P[(P.season_type == "REG") & (P.two_point_attempt != 1)]
OSTAT = ["db", "pass_yds", "pass_td", "ints", "epa", "car", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"]; DSTAT = ["sacks", "tfl", "qbh", "dint", "pd", "ff", "tkl", "dtd"]
DIV = {"BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East", "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North", "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
       "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West", "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East", "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
       "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South", "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West"}

def tally(df, col, name, w=1.0):
    d = df[df[col].notna()]; return d.groupby(["season", col, "defteam"]).size().mul(w).rename(name).rename_axis(["season", "pid", "team"])
def build(maxw, minw=0):
    """player-season and team-season totals from games in weeks minw..maxw"""
    p = P[(P.week <= maxw) & (P.week >= minw)]
    pa = p[(p.qb_dropback == 1) & p.passer_player_id.notna()].groupby(["season", "passer_player_id", "posteam"]).agg(db=("qb_dropback", "sum"), pass_yds=("passing_yards", "sum"), pass_td=("pass_touchdown", "sum"),
            ints=("interception", "sum"), epa=("qb_epa", "sum")).rename_axis(["season", "pid", "team"])
    ru = p[p.rusher_player_id.notna()].groupby(["season", "rusher_player_id", "posteam"]).agg(car=("rushing_yards", "size"), rush_yds=("rushing_yards", "sum"), rush_td=("rush_touchdown", "sum")).rename_axis(["season", "pid", "team"])
    rc = p[p.receiver_player_id.notna() & (p.play_type == "pass")].groupby(["season", "receiver_player_id", "posteam"]).agg(tgt=("receiving_yards", "size"), rec=("complete_pass", "sum"), rec_yds=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum")).rename_axis(["season", "pid", "team"])
    off = pa.join(ru, how="outer").join(rc, how="outer").fillna(0).reset_index()
    off = off.groupby(["season", "pid"]).agg({**{c: "sum" for c in OSTAT}, "team": "last"}).reset_index()
    gp = pd.concat([p[["season", "game_id", c]].rename(columns={c: "pid"}) for c in ("passer_player_id", "rusher_player_id", "receiver_player_id")]).dropna().drop_duplicates().groupby(["season", "pid"]).size().rename("gp")
    off = off.merge(gp, left_on=["season", "pid"], right_index=True, how="left").fillna({"gp": 0})
    d = [tally(p, "sack_player_id", "sacks"), tally(p, "half_sack_1_player_id", "sacks", .5), tally(p, "half_sack_2_player_id", "sacks", .5),
         tally(p, "tackle_for_loss_1_player_id", "tfl"), tally(p, "tackle_for_loss_2_player_id", "tfl"), tally(p, "qb_hit_1_player_id", "qbh"), tally(p, "qb_hit_2_player_id", "qbh"),
         tally(p, "interception_player_id", "dint"), tally(p, "pass_defense_1_player_id", "pd"), tally(p, "pass_defense_2_player_id", "pd"),
         tally(p, "forced_fumble_player_1_player_id", "ff"), tally(p, "forced_fumble_player_2_player_id", "ff")] + \
        [tally(p, f"solo_tackle_{i}_player_id", "tkl") for i in (1, 2)] + [tally(p, f"assist_tackle_{i}_player_id", "tkl") for i in (1, 2, 3, 4)] + [tally(p, f"tackle_with_assist_{i}_player_id", "tkl") for i in (1, 2)]
    dtd = p[(p.touchdown == 1) & p.td_player_id.notna() & (p.td_team == p.defteam)]
    d.append(dtd.groupby(["season", "td_player_id", "defteam"]).size().rename("dtd").rename_axis(["season", "pid", "team"]))
    de = pd.concat(d, axis=1).T.groupby(level=0).sum().T.fillna(0).reset_index()
    de = de.groupby(["season", "pid"]).agg({**{c: "sum" for c in DSTAT}, "team": "last"}).reset_index()
    de = de[~de.pid.map(PL.position_group).isin(["QB", "RB", "WR", "TE", "OL", "SPEC"])]
    s = S[(S.week <= maxw) & (S.week >= minw) & S.result.notna()]
    tg = pd.concat([s.assign(team=s.home_team, m=s.result, coach=s.home_coach), s.assign(team=s.away_team, m=-s.result, coach=s.away_coach)])
    tm = tg.groupby(["season", "team"]).agg(tg=("m", "size"), w=("m", lambda m: (m > 0).sum() + .5 * (m == 0).sum()), pdiff=("m", "sum"), coach=("coach", "last")).reset_index()
    tm["wp"] = tm.w / tm.tg; tm["mpg"] = tm.pdiff / tm.tg
    return off, de, tm
FULL = build(99)
fin_tm = FULL[2].set_index(["season", "team"]); prev_wp = {(s + 1, t): v for (s, t), v in fin_tm.wp.items()}; prev_coach = {(s + 1, t): v for (s, t), v in fin_tm.coach.items()}; prev_mpg = {(s + 1, t): v for (s, t), v in fin_tm.mpg.items()}
TOT_G = {**{(s, t): g for (s, t), g in fin_tm.tg.items()}, **{(SEASON, t): 17 for t in DIV}}

# ------------------------------------------------------------------ teams
def ls_rating(g, y, lam):
    """ratings that best explain y = home - away + home edge, shrunk toward zero"""
    teams = sorted(set(g.home_team) | set(g.away_team)); ix = {t: i for i, t in enumerate(teams)}; n = len(teams)
    A = np.zeros((len(g), n + 1)); A[np.arange(len(g)), [ix[t] for t in g.home_team]] = 1; A[np.arange(len(g)), [ix[t] for t in g.away_team]] = -1; A[:, n] = 1
    R = np.eye(n + 1) * lam; R[n, n] = 0; sol = np.linalg.solve(A.T @ A + R, A.T @ np.asarray(y, float)); return dict(zip(teams, sol[:n]))
_tf = {}
def team_feats(season, W):
    if (season, W) not in _tf:
        g = S[S.season == season]; lines = g[(g.week <= W + 1) & g.spread_line.notna()]; played = g[(g.week <= W) & g.result.notna()]
        mk = ls_rating(lines, lines.spread_line, 0.5) if len(lines) else {}; pf = ls_rating(played, played.result.clip(-24, 24), 3.0) if len(played) else {}
        _tf[(season, W)] = {t: (mk.get(t, 0.0), pf.get(t, 0.0), prev_mpg.get((season, t), 0.0)) for t in set(g.home_team) | set(g.away_team)}
    return _tf[(season, W)]
def game_rows(season, W):
    f = team_feats(season, W); g = S[(S.season == season) & (S.week > W)].copy()
    for i, k in enumerate(("mk", "pf", "pr")): g["d_" + k] = [f[h][i] - f[a][i] for h, a in zip(g.home_team, g.away_team)]
    return g
_gm = {}
def game_model(W, exclude=None):
    """predict the home margin of games after week W from the three rating gaps; the intercept is home-field advantage"""
    if (W, exclude) not in _gm:
        tr = pd.concat([game_rows(s, W) for s in range(FIRST, SEASON) if s != exclude]); tr = tr[tr.result.notna()]
        m = LinearRegression().fit(tr[["d_mk", "d_pf", "d_pr"]], tr.result); sd = float(np.std(tr.result - m.predict(tr[["d_mk", "d_pf", "d_pr"]])))
        _gm[(W, exclude)] = (m, sd)
    return _gm[(W, exclude)]
def remaining(season, W, exclude=None):
    m, sd = game_model(W, exclude); g = game_rows(season, W); g["pred"] = m.predict(g[["d_mk", "d_pf", "d_pr"]]) if len(g) else []; g["p_home"] = norm.cdf(g.pred / sd); return g, sd, m
def simulate(season, W, n=20000, tau=2.0, exclude=None, seed=7):
    """play out the remaining schedule n times. tau = how far a team's true strength may sit from our rating (points)"""
    g, sd, _ = remaining(season, W, exclude); teams = sorted(DIV if season == SEASON else set(S[S.season == season].home_team)); ix = {t: i for i, t in enumerate(teams)}; rng = np.random.default_rng(seed)
    done = S[(S.season == season) & (S.week <= W) & S.result.notna()]; w0 = np.zeros(len(teams))
    for h, a, r in zip(done.home_team, done.away_team, done.result): w0[ix[h]] += (r > 0) + .5 * (r == 0); w0[ix[a]] += (r < 0) + .5 * (r == 0)
    eff = rng.normal(0, tau, (n, len(teams))); wins = np.tile(w0, (n, 1)); hi = np.array([ix[t] for t in g.home_team], int); ai = np.array([ix[t] for t in g.away_team], int)
    if len(g):
        mar = g.pred.values[None, :] + eff[:, hi] - eff[:, ai] + rng.normal(0, np.sqrt(max(sd ** 2 - 2 * tau ** 2, 100.0)), (n, len(g))); hw = mar > 0
        for j in range(len(g)): wins[:, hi[j]] += hw[:, j]; wins[:, ai[j]] += ~hw[:, j]
    simulate.eff = eff      # per-sim true-strength draws, for anyone playing out the postseason
    return teams, wins, g, sd
def exp_wins(season, W):
    """expected final wins and point margin for every team, without simulating"""
    g, sd, _ = remaining(season, W); done = S[(S.season == season) & (S.week <= W) & S.result.notna()]; out = {}
    for h, a, r in zip(done.home_team, done.away_team, done.result):
        for t, m in ((h, r), (a, -r)): o = out.setdefault(t, [0, 0.0, 0.0]); o[0] += 1; o[1] += (m > 0) + .5 * (m == 0); o[2] += m
    for h, a, p, pr in zip(g.home_team, g.away_team, g.p_home, g.pred):
        for t, pw, m in ((h, p, pr), (a, 1 - p, -pr)): o = out.setdefault(t, [0, 0.0, 0.0]); o[0] += 1; o[1] += pw; o[2] += m
    return out
def team_table(W):
    """team table with projected final wins and margin, same columns build() gives"""
    _, _, tm = build(W); tm = tm.copy(); tm["ytd_w"] = tm.w; tm["ytd_tg"] = tm.tg; cache = {}
    for i, r in tm.iterrows():
        if r.season < FIRST: continue
        e = cache.setdefault(r.season, exp_wins(r.season, W)).get(r.team)
        if e: tm.loc[i, ["tg", "w", "pdiff"]] = e
    tm["wp"] = tm.w / tm.tg; tm["mpg"] = tm.pdiff / tm.tg; return tm

# ------------------------------------------------------------------ players
_of = {}
def off_frame(W):
    """one row per offensive player-season with everything known after week W, and what he did the rest of the way"""
    if W in _of: return _of[W]
    off, _, tm = build(W); rec, _, _ = build(W, max(W - 1, 1)); tmi = tm.set_index(["season", "team"]); fi = FULL[0].set_index(["season", "pid"]); d = off.copy()
    d["tg"] = [tmi.tg.get((s, t), W) for s, t in zip(d.season, d.team)]; d["rem"] = [TOT_G.get((s, t), 17) for s, t in zip(d.season, d.team)] - d.tg
    d["grp"] = d.pid.map(PL.position_group); d["qb"] = (d.grp == "QB"); d["rookie"] = (d.pid.map(PL.rookie_season) == d.season).astype(float); d["lpick"] = np.log(d.pid.map(PL.draft_pick).fillna(260))
    d["gshare"] = d.gp / d.tg.clip(lower=1); opp = lambda x: x.db + x.car + x.tgt
    d["y_opp"] = opp(d) / d.tg.clip(lower=1); ri = rec.set_index(["season", "pid"]); d["r_opp"] = [opp(ri.loc[k]) / min(W, 2) if k in ri.index else 0.0 for k in zip(d.season, d.pid)]
    key = list(zip(d.season, d.pid)); pk = [(s - 1, p) for s, p in key]; has = np.array([k in fi.index for k in pk]); d["np"] = (~has).astype(float)
    d["p_opp"] = [opp(fi.loc[k]) / 17 if h else 0.0 for k, h in zip(pk, has)]; d["p_gp"] = [fi.gp[k] / 17 if h else 0.0 for k, h in zip(pk, has)]
    mk = [team_feats(s, W).get(t, (0, 0, 0))[0] if s >= FIRST else 0.0 for s, t in zip(d.season, d.team)]; d["tm_mk"] = mk
    for k in OSTAT:
        d["y_" + k] = d[k] / d.tg.clip(lower=1); d["p_" + k] = [fi[k][kk] / 17 if h else 0.0 for kk, h in zip(pk, has)]
        d["t_" + k] = (np.array([fi[k].get(kk, np.nan) for kk in key], float) - d[k]) / d.rem.clip(lower=1)
    _of[W] = d[d.season >= FIRST].reset_index(drop=True); return _of[W]
def feats(k): return ["y_" + k, "p_" + k, "np", "y_opp", "r_opp", "p_opp", "p_gp", "gshare", "rookie", "lpick", "tm_mk"]
def _fitpred(tr, te, k):
    X = feats(k); return LinearRegression().fit(tr[X], tr["t_" + k]).predict(te[X])
def off_oof(W):
    """leave-one-season-out predictions of each stat's rest-of-season rate (o_<stat>), fitted separately by position"""
    d = off_frame(W).copy(); d["g4"] = np.where(d.grp.isin(["QB", "RB", "WR", "TE"]), d.grp, "WR"); hist = d[(d.season < SEASON) & (d.season > FIRST)]
    for k in OSTAT:
        d["o_" + k] = np.nan
        for g in ("QB", "RB", "WR", "TE"):
            for yr in sorted(hist.season.unique()):
                tr = hist[(hist.g4 == g) & (hist.season != yr) & hist["t_" + k].notna()]; te = d[(d.g4 == g) & (d.season == yr)]
                if len(tr) > 80 and len(te): d.loc[te.index, "o_" + k] = _fitpred(tr, te, k)
    return d
def project(W):
    """projected season totals for every player and team after week W (what has happened plus the predicted rest of the season)"""
    d = off_frame(W).copy(); d["g4"] = np.where(d.grp.isin(["QB", "RB", "WR", "TE"]), d.grp, "WR")
    for k in OSTAT:
        d["r_" + k] = d["y_" + k]
        for g in ("QB", "RB", "WR", "TE"):
            tr = d[(d.g4 == g) & (d.season < SEASON) & (d.season > FIRST) & d["t_" + k].notna()]; te = d[d.g4 == g]
            if len(tr) > 80 and len(te): d.loc[te.index, "r_" + k] = _fitpred(tr, te, k)
        if k in ("db", "pass_yds", "pass_td", "ints", "epa"): d.loc[(d.g4 != "QB") & (d.db == 0), "r_" + k] = 0.0     # non-passers stay at zero
        d["ytd_" + k] = d[k]; d[k] = d[k] + d["r_" + k].clip(lower=None if k == "epa" else 0) * d.rem
    off = d[["season", "pid", "team", "gp", "rem"] + OSTAT + ["ytd_" + k for k in OSTAT] + ["r_" + k for k in OSTAT]]
    _, de, tm0 = build(W); tmi = tm0.set_index(["season", "team"]); fi = FULL[1].set_index(["season", "pid"]); e = de.copy()
    e["tg"] = [tmi.tg.get((s, t), W) for s, t in zip(e.season, e.team)]; e["rem"] = [TOT_G.get((s, t), 17) for s, t in zip(e.season, e.team)] - e.tg
    for k in DSTAT:
        e["y"] = e[k] / e.tg.clip(lower=1); pv = np.array([fi[k].get((s - 1, p), np.nan) for s, p in zip(e.season, e.pid)], float) / 17; e["np"] = np.isnan(pv).astype(float); e["p"] = np.nan_to_num(pv)
        e["t"] = (np.array([fi[k].get((s, p), np.nan) for s, p in zip(e.season, e.pid)], float) - e[k]) / e.rem.clip(lower=1)
        tr = e[(e.season < SEASON) & (e.season >= FIRST + 1) & e.t.notna()]; e["ytd_" + k] = e[k]; e[k] = e[k] + np.clip(LinearRegression().fit(tr[["y", "p", "np"]], tr.t).predict(e[["y", "p", "np"]]), 0, None) * e.rem
    return off, e[["season", "pid", "team"] + DSTAT + ["ytd_" + k for k in DSTAT]], team_table(W)
