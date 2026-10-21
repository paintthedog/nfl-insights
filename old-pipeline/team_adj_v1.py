"""Two adjustments to the team spread/total model, each tested on 2019-2026 (tests/t_team_qb_def.py):
  quarterback quality  this game's starter's EPA per dropback (shrunk toward a backup level) minus the decayed mix of the
                       starters behind the team's rating. Fitted weight: about 25 points of margin per 1.0 EPA/dropback.
  missing defenders    snap-weighted count of defensive regulars (50%+ of snaps over the last four games) who are out.
                       Fitted weight: about 0.8 points of margin per missing regular.
For upcoming games the starter mix comes from this week's projections (each QB's chance to play) and missing defenders from
the injury report (game status when posted, practice participation before that)."""
import os, numpy as np, pandas as pd
import injuries as J
K, PRIOR = 350, -0.06
DECAY, CARRY = 0.92, 0.5
_PRAC_OUT = {"DNP": 1 - J.PRAC_MID["DNP"], "Limited": 1 - J.PRAC_MID["Limited"], "Full": 1 - J.PRAC_MID["Full"]}

def _qb_hist(pbp, sched):
    p = pbp[(pbp.season_type == "REG") & (pbp.qb_dropback == 1) & pbp.passer_player_id.notna() & pbp.epa.notna()]
    pg = p.groupby(["game_id", "passer_player_id"]).agg(epa=("epa", "sum"), db=("epa", "size")).reset_index()
    pg = pg.merge(sched[["game_id", "gameday"]], on="game_id"); pg["gameday"] = pd.to_datetime(pg.gameday); pg = pg.sort_values("gameday")
    pg["ce"] = pg.groupby("passer_player_id").epa.cumsum(); pg["cn"] = pg.groupby("passer_player_id").db.cumsum()
    return {pid: (d.gameday.values, d.ce.values, d.cn.values) for pid, d in pg.groupby("passer_player_id")}

def add(g, pbp, sched_all, tg):
    """g: game rows from lines_model.build (home perspective, incl. _pred rows); tg: team_games output (has qb per played game)"""
    H = _qb_hist(pbp, sched_all)
    def q_at(pid, day):
        if not isinstance(pid, str) or pid not in H: return PRIOR
        days, ce, cn = H[pid]; i = np.searchsorted(days, np.datetime64(day)) - 1
        e, n = (ce[i], cn[i]) if i >= 0 else (0.0, 0.0); return (e + K * PRIOR) / (n + K)
    s = g[["game_id", "season", "week", "gameday", "home_team", "away_team", "_pred"]].rename(columns={"_pred": "pred"}).copy(); s["gameday"] = pd.to_datetime(s.gameday)
    long = pd.concat([s.assign(team=s.home_team), s.assign(team=s.away_team)])[["game_id", "season", "week", "gameday", "team", "pred"]]
    long = long.merge(tg[["game_id", "posteam", "qb"]].rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left").sort_values(["team", "gameday"])
    # expected starter value for upcoming games: this week's QB chances to play, the rest of the time a backup-level QB
    exp_q = {}
    for (yr, wk) in long[long.pred][["season", "week"]].drop_duplicates().itertuples(index=False):
        f = f"td_props_{yr}_wk{wk}.csv"
        if os.path.exists(f):
            t = pd.read_csv(f); t = t[t.pos == "QB"]
            for tm, d in t.groupby("posteam"):
                p = d.p_play.fillna(1.0).clip(0, 1); tot = min(p.sum(), 1.0); p = p / max(p.sum(), 1e-9) * tot
                exp_q[(yr, wk, tm)] = lambda day, d=d, p=p, tot=tot: float(sum(pi * q_at(pid, day) for pid, pi in zip(d.pid, p)) + (1 - tot) * PRIOR)
    rows = []
    for tm, d in long.groupby("team"):
        past = []
        for r in d.itertuples():
            w = num = 0.0
            for i, (ss, qq) in enumerate(reversed(past[-40:])):
                wt = DECAY ** i * (CARRY if ss < r.season else 1.0); w += wt; num += wt * q_at(qq, r.gameday)
            qr = num / w if w else PRIOR
            qt = exp_q[(r.season, r.week, tm)](r.gameday) if r.pred and (r.season, r.week, tm) in exp_q else q_at(r.qb, r.gameday)
            rows.append((r.game_id, tm, qt - qr, qt))
            if not r.pred and isinstance(r.qb, str): past.append((r.season, r.qb))
    Q = pd.DataFrame(rows, columns=["game_id", "team", "qd", "qthis"])
    # missing defensive regulars
    DM = []
    if os.path.exists("data/snaps_def.parquet"):
        sd = pd.read_parquet("data/snaps_def.parquet"); sd = sd[sd.game_type == "REG"].merge(s[["game_id", "gameday"]], on="game_id"); sd["gameday"] = pd.to_datetime(sd.gameday)
        pfr2g = pd.read_parquet("data/players.parquet")[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
        inj = pd.read_parquet("data/injuries_all.parquet") if os.path.exists("data/injuries_all.parquet") else None
        for (yr, tm), d in sd.groupby(["season", "team"]):
            games = d.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()
            for i, gid in enumerate(games):
                if i == 0: DM.append((gid, tm, 0.0)); continue
                prev = d[d.game_id.isin(games[max(0, i - 4):i])]; reg = prev.groupby("pfr_player_id").defense_pct.mean(); reg = reg[reg >= .5]
                now = set(d[(d.game_id == gid) & (d.defense_snaps > 0)].pfr_player_id); DM.append((gid, tm, float(reg[~reg.index.isin(now)].sum())))
        for r in long[long.pred].itertuples():
            d = sd[(sd.season == r.season) & (sd.team == r.team)]
            if d.empty: DM.append((r.game_id, r.team, 0.0)); continue
            last = d.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()[-4:]
            reg = d[d.game_id.isin(last)].groupby("pfr_player_id").defense_pct.mean(); reg = reg[reg >= .5]
            miss = 0.0
            if inj is not None:
                w = inj[(inj.season == r.season) & (inj.week == r.week) & (inj.team == r.team)].set_index("gsis_id")
                for pfr, share in reg.items():
                    gid_ = pfr2g.get(pfr)
                    if gid_ is None or gid_ not in w.index: continue
                    x = w.loc[gid_]; x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
                    if x.report_status in ("Out", "Doubtful"): po = 1.0 if x.report_status == "Out" else 0.97
                    elif x.report_status == "Questionable": po = 1 - J.p_play(x.practice_status)
                    elif "not injury related" in str(x.practice_primary_injury).lower(): po = 0.0
                    else: po = _PRAC_OUT.get(J.PRACTICE.get(x.practice_status), 0.0)
                    miss += share * po
            DM.append((r.game_id, r.team, miss))
    DM = pd.DataFrame(DM, columns=["game_id", "team", "dmiss"]).drop_duplicates(["game_id", "team"], keep="last")
    for side, col in (("h", "home_team"), ("a", "away_team")):
        g = g.merge(Q.rename(columns={"team": col, "qd": f"{side}_qd", "qthis": f"{side}_qthis"}), on=["game_id", col], how="left")
        g = g.merge(DM.rename(columns={"team": col, "dmiss": f"{side}_dmiss"}), on=["game_id", col], how="left")
    for c in ("h_qd", "a_qd", "h_dmiss", "a_dmiss"): g[c] = g[c].fillna(0.0)
    g["qd_gap"] = g.h_qd - g.a_qd; g["qd_sum"] = g.h_qd + g.a_qd
    g["dmiss_gap"] = g.a_dmiss - g.h_dmiss; g["dmiss_sum"] = g.h_dmiss + g.a_dmiss
    return g
