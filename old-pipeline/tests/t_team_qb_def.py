"""Do a quarterback-quality term and missing defensive starters improve the team spread/total model? Walk-forward 2019-2026."""
import sys; sys.path.insert(0, "/home/claude")
import numpy as np, pandas as pd, statsmodels.api as sm
import lines_model as L
pbp, sched = L.load(); g = L.build(pbp, sched)
# ---- quarterback quality: each passer's EPA per dropback before the game, shrunk toward a backup-level prior
p = pbp[(pbp.season_type == "REG") & (pbp.qb_dropback == 1) & pbp.passer_player_id.notna() & pbp.epa.notna()]
pg = p.groupby(["game_id", "passer_player_id"]).agg(epa=("epa", "sum"), db=("epa", "size")).reset_index()
pg = pg.merge(sched[["game_id", "gameday"]], on="game_id"); pg["gameday"] = pd.to_datetime(pg.gameday); pg = pg.sort_values("gameday")
pg["c_epa"] = pg.groupby("passer_player_id").epa.cumsum() - pg.epa; pg["c_db"] = pg.groupby("passer_player_id").db.cumsum() - pg.db
K, PRIOR = 350, -0.06
hist = {}   # passer -> list of (gameday, cum_epa_before, cum_db_before) to look up value at any date
for r in pg.itertuples(): hist.setdefault(r.passer_player_id, []).append((r.gameday, r.c_epa + r.epa, r.c_db + r.db))
def q_at(pid, day):
    """value of passer as of just before `day`"""
    if not isinstance(pid, str) or pid not in hist: return PRIOR
    e, n = 0.0, 0.0
    for d, ce, cn in hist[pid]:
        if d < day: e, n = ce, cn
        else: break
    return (e + K * PRIOR) / (n + K)
tg = L.team_games(pbp)[["game_id", "posteam", "qb"]].rename(columns={"posteam": "team"})
s = sched.copy(); s["gameday"] = pd.to_datetime(s.gameday)
long = pd.concat([s.assign(team=s.home_team), s.assign(team=s.away_team)])[["game_id", "season", "gameday", "team"]].merge(tg, on=["game_id", "team"], how="left").sort_values(["team", "gameday"])
rows = []
for tm, d in long.groupby("team"):
    past = []   # (season, qb) of earlier games
    for r in d.itertuples():
        # rated QB: decayed mix of the starters behind the team's rating, valued as of now
        w, num = 0.0, 0.0
        for i, (ss, qq) in enumerate(reversed(past)):
            wt = L.DECAY ** i * (L.SEASON_CARRY if ss < r.season else 1.0); w += wt; num += wt * q_at(qq, r.gameday)
        q_rated = num / w if w else PRIOR
        rows.append(dict(game_id=r.game_id, team=tm, q_this=q_at(r.qb, r.gameday), q_rated=q_rated)); past.append((r.season, r.qb))
Q = pd.DataFrame(rows); Q["q_delta"] = Q.q_this - Q.q_rated
# ---- missing defensive regulars: snap-weighted share of last-4-game regulars (50%+ of snaps) who did not play
sd = pd.read_parquet("data/snaps_def.parquet"); sd = sd[sd.game_type == "REG"]; sd = sd.merge(s[["game_id", "gameday"]], on="game_id"); sd["gameday"] = pd.to_datetime(sd.gameday)
dm = []
for (ss, tm), d in sd.groupby(["season", "team"]):
    games = d.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()
    for i, gid in enumerate(games):
        if i == 0: dm.append(dict(game_id=gid, team=tm, d_miss=0.0)); continue
        prev = d[d.game_id.isin(games[max(0, i - 4):i])]; reg = prev.groupby("pfr_player_id").defense_pct.mean(); reg = reg[reg >= .5]
        now = set(d[(d.game_id == gid) & (d.defense_snaps > 0)].pfr_player_id)
        dm.append(dict(game_id=gid, team=tm, d_miss=float(reg[~reg.index.isin(now)].sum())))
DM = pd.DataFrame(dm)
for side, tcol in (("h", "home_team"), ("a", "away_team")):
    g = g.merge(Q.rename(columns={"team": tcol, "q_delta": f"{side}_qd", "q_this": f"{side}_qthis"})[["game_id", tcol, f"{side}_qd", f"{side}_qthis"]], on=["game_id", tcol], how="left")
    g = g.merge(DM.rename(columns={"team": tcol, "d_miss": f"{side}_dmiss"}), on=["game_id", tcol], how="left")
g["qd_gap"] = g.h_qd - g.a_qd; g["qd_sum"] = g.h_qd + g.a_qd
g["dmiss_gap"] = g.a_dmiss.fillna(0) - g.h_dmiss.fillna(0); g["dmiss_sum"] = g.h_dmiss.fillna(0) + g.a_dmiss.fillna(0)
g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)]
def run(MX, TX):
    out = []
    for yr in range(2019, 2027):
        tr, te = g[g.season < yr], g[g.season == yr].copy()
        mm = sm.OLS(tr.result, sm.add_constant(tr[MX])).fit(); mt = sm.OLS(tr.total, sm.add_constant(tr[TX])).fit()
        te["mm"] = mm.predict(sm.add_constant(te[MX], has_constant="add")); te["mt"] = mt.predict(sm.add_constant(te[TX], has_constant="add")); out.append(te)
    o = pd.concat(out); ch = o[(o.h_qd.abs() > .03) | (o.a_qd.abs() > .03)]
    return (round(np.mean(np.abs(o.result - o.mm)), 3), round(np.mean(np.abs(o.total - o.mt)), 3), round(np.mean(np.abs(ch.result - ch.mm)), 3), len(ch), mm.params.round(2).to_dict())
MX, TX = L.MARGIN_X, L.TOTAL_X
for lab, mx, tx in (("current", MX, TX), ("+ QB quality", MX + ["qd_gap"], TX + ["qd_sum"]), ("+ missing defenders", MX + ["dmiss_gap"], TX + ["dmiss_sum"]),
                    ("+ both", MX + ["qd_gap", "dmiss_gap"], TX + ["qd_sum", "dmiss_sum"])):
    r = run(mx, tx); print(f"{lab:22s} margin MAE {r[0]}  total MAE {r[1]}  | games with a QB-quality shift: margin MAE {r[2]} (n={r[3]})")
    print("   ", {k: v for k, v in r[4].items() if k in ("qd_gap", "dmiss_gap", "qb_gap", "off_gap", "def_gap")})
print("closing-line margin MAE for reference:", round(np.mean(np.abs(g[g.season >= 2019].result - g[g.season >= 2019].spread_line)), 3))
g[["game_id", "h_qd", "a_qd", "h_dmiss", "a_dmiss"]].to_pickle("/home/claude/tests/team_feats.pkl")
