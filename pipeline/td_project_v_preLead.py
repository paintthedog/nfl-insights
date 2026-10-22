"""Project anytime-TD probabilities for the upcoming week."""
import numpy as np, pandas as pd
import td_model as T, injuries as J

SEASON, WEEK = 2026, int(__import__("os").environ.get("NFL_WEEK", 5))
# Only these QBs are projected (starters, same list as the INT model); backup QBs are dropped.
import json, os
QB_OVERRIDES = json.load(open("qb_overrides.json")).get(str(WEEK), {}) if os.path.exists("qb_overrides.json") else {}
MIN_USAGE = 0.04          # decayed carry share + target share needed to be listed
QUESTIONABLE = []

pbp, sched, players = T.load()
pg, team, ctx = T.build_tables(pbp, sched, players)

ros = pd.read_parquet("data/rosters_2026.parquet")
# each player's most recent roster row (bye-week teams are missing from that week's file, and the current week's file
# may not exist yet); a game-day inactive tag only counts if it is from THIS week
ros = ros[ros.week <= WEEK].sort_values("week").drop_duplicates("gsis_id", keep="last")
CARRY_ROS = not (ros.week == WEEK).any()
out_codes = ros.status.isin(["RES", "CUT", "RET", "DEV", "EXE"]) | ((ros.status == "INA") & (ros.week == WEEK))      # the "I01" description lags a week (it marks LAST game's inactives), so it is not used
unavailable = set(ros[out_codes].gsis_id)
# official injury report: Out / Doubtful are removed from the pool; Questionable is flagged
inj = pd.read_parquet("data/injuries_2026.parquet")
_inj_all = inj[inj.season == SEASON]
inj = _inj_all[_inj_all.week == WEEK]
# Game statuses arrive team by team (Thursday teams first). A team with statuses this week uses them; every other team
# carries its last report forward at the measured rate those players return the next game (or the one after, for a look-ahead week).
_teams_now = set(inj[inj.report_status.notna()].team)
CARRY_INJ = len(_teams_now) == 0
q_play, q_lbl = {}, {}
_cur = inj[inj.team.isin(_teams_now)]
unavailable |= set(_cur[_cur.report_status.isin(["Out", "Doubtful"])].gsis_id.dropna())
QUESTIONABLE = list(set(QUESTIONABLE) | set(_cur[_cur.report_status == "Questionable"].full_name.dropna()))
q_play.update({r.gsis_id: J.p_play(r.practice_status) for r in _cur[_cur.report_status == "Questionable"].itertuples() if isinstance(r.gsis_id, str)})
listed = set(_cur[~_cur.report_status.isin(["Out", "Doubtful"])].gsis_id.dropna())
# Teams with this week's practice report but no game statuses yet (midweek): practice participation decides, and last
# week's statuses and Monday's news are superseded. A player missing from his team's report is practicing normally.
_teams_prac = set(inj[inj.practice_status.notna()].team) - _teams_now
for r in inj[inj.team.isin(_teams_prac)].itertuples():
    if not isinstance(r.gsis_id, str) or "not injury related" in str(r.practice_primary_injury).lower(): continue
    _p = J.p_mid(r.practice_status)
    if _p is None: continue
    q_play[r.gsis_id] = _p; q_lbl[r.gsis_id] = J.PRAC_LBL[J.PRACTICE[r.practice_status]] + " this week" + (f" ({str(r.practice_primary_injury).lower()})" if isinstance(r.practice_primary_injury, str) else "")
listed |= set(inj[inj.team.isin(_teams_prac)].gsis_id.dropna())
_prev_all = _inj_all[(_inj_all.week < WEEK) & ~_inj_all.team.isin(_teams_now | _teams_prac)]
if len(_prev_all):
    _lastwk = _prev_all.groupby("team").week.transform("max"); prev = _prev_all[_prev_all.week == _lastwk]
    _sn = pd.read_parquet("data/snaps.parquet"); _sn = _sn[(_sn.season == SEASON) & (_sn.offense_pct > 0)]
    _played = set(zip(_sn.week, _sn.gsis_id))
    _done = int(sched[(sched.season == SEASON) & sched.result.notna()].week.max())       # last completed week
    _table, _ahead = (J.CARRY, "last") if WEEK - _done <= 1 else (J.CARRY2, "wk %d" % _done)
    for r in prev.itertuples():
        if not isinstance(r.gsis_id, str): continue
        k = "Out" if r.report_status == "Out" else "Doubtful" if r.report_status == "Doubtful" else "Q_sat" if (r.report_status == "Questionable" and (r.week, r.gsis_id) not in _played) else None
        if k:
            q_play[r.gsis_id] = _table[k]; q_lbl[r.gsis_id] = J.CARRY_LBL[k].replace("last wk", "wk %d" % int(r.week))
    listed |= set(prev.gsis_id.dropna())
# hand-entered injury news (news_wk<week>.json), used until official statuses exist for that team; chances are the measured
# rates for the nearest report category, never a guess per player
if os.path.exists(f"news_wk{WEEK}.json"):
    import re as _re
    _nm = lambda s: _re.sub(r"[^a-z]", "", _re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower()))
    _rk = {(r.team, _nm(r.full_name)): r.gsis_id for r in ros.itertuples() if isinstance(r.gsis_id, str)}
    for _n, _t, _p, _note in json.load(open(f"news_wk{WEEK}.json"))["players"]:
        _pid = _rk.get((_t, _nm(_n)))
        if _pid is None: print("news: could not match", _n, _t); continue
        if _t in _teams_now or (_t in _teams_prac and _p != 0): continue   # this week's report has superseded the news (season-ending news still applies)
        if _p == 0: unavailable.add(_pid)
        else: q_play[_pid] = _p; q_lbl[_pid] = _note
roster_team = ros[~out_codes].drop_duplicates("gsis_id").set_index("gsis_id").team

# starting QBs: each team's latest starter (most attempts) unless overridden
import int_model as M
ipbp, isched = M.load(); iqb, _, ictx = M.build_tables(ipbp, isched)
st = iqb[iqb.is_starter].merge(ictx[["game_id", "gameday"]].drop_duplicates(), on="game_id")
starters = st.sort_values("gameday").groupby("posteam").tail(1).set_index("posteam").passer_player_id.to_dict()
starters.update(QB_OVERRIDES)
starter_ids = set(starters.values())
# an uncertain starting quarterback: he and his backup share the job in the simulation (exactly one of them plays each run)
QB_BACKUP = {}
if os.path.exists(f"news_wk{WEEK}.json"):
    for _t, _u in json.load(open(f"news_wk{WEEK}.json")).get("qb_uncertain", {}).items():
        _b = _rk.get((_t, _nm(_u["backup"])))
        if _b is None or _t not in starters or _t in _teams_now: print("qb news: skipped", _t); continue
        _pq = _u["p"]
        if _t in _teams_prac:                               # this week's practice report decides; a starter not on it is healthy
            if starters[_t] not in q_play: print("qb news: cleared by practice report", _t); continue
            _pq = q_play[starters[_t]]
        else: q_lbl[starters[_t]] = _u["note"]
        QB_BACKUP[_t] = _b; starter_ids.add(_b)
        q_play[starters[_t]] = _pq
        q_play[_b] = 1 - _pq; q_lbl[_b] = "starts if " + players.drop_duplicates("gsis_id").set_index("gsis_id").display_name.get(starters[_t], "the starter").split()[-1] + " sits"

games = sched[(sched.season == SEASON) & (sched.week == WEEK) & sched.result.isna()]   # unplayed games only
# ---- news overrides (news_overrides_wk<W>.json): official Friday designations and beat reporting arrive hours before
# the nflverse injuries feed; curated by hand at refresh time, names resolved to ids, they override the practice-based
# chances above. "out" removes the player from the pool entirely.
NEWS_NEXT_MAN = {}
try:
    _nw = json.load(open(f"news_overrides_wk{WEEK}.json"))
    _nm = players.drop_duplicates("gsis_id"); _nm = _nm.assign(k=_nm.display_name.str.lower().str.replace(r"\b(jr|sr|ii|iii|iv)\b\.?", "", regex=True).str.replace(r"[^a-z ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
    import re as _re
    _active_now = set(pg[pg.season == SEASON].pid) | set(roster_team.index)
    def _find(n):
        k = _re.sub(r"\s+", " ", _re.sub(r"[^a-z ]", "", _re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n.lower()))).strip()
        c = _nm[_nm.k == k].gsis_id.tolist()
        if not c: return None
        live = [p for p in c if p in _active_now]
        return (live or c)[-1]
    for n in _nw.get("out", []):
        _p = _find(n)
        if _p is None: print("news: no id for", n); continue
        unavailable.add(_p); q_play.pop(_p, None)
    for n, pr in _nw.get("p_play", {}).items():
        _p = _find(n)
        if _p is None: print("news: no id for", n); continue
        q_play[_p] = pr; q_lbl[_p] = f"news ({_nw['asof'].split('T')[0]})"
        if pr >= 0.97: q_play.pop(_p, None); q_lbl.pop(_p, None)
    for tm, n in _nw.get("qb_start", {}).items():
        _p = _find(n)
        if _p is not None: starters[tm] = _p; starter_ids.add(_p); QB_BACKUP.pop(tm, None); q_play.pop(_p, None); q_lbl.pop(_p, None)
    # uncertain starter: {"BAL": {"starter": "Lamar Jackson", "backup": "Tyler Huntley", "p": 0.35, "note": "..."}} - exactly one plays each run
    for tm, u in _nw.get("qb_uncertain", {}).items():
        _s, _b = _find(u["starter"]), _find(u["backup"])
        if _s is None or _b is None: print("qb_uncertain: no id for", u); continue
        starters[tm] = _s; QB_BACKUP[tm] = _b; starter_ids |= {_s, _b}
        q_play[_s] = u["p"]; q_lbl[_s] = u.get("note", f"news ({_nw['asof'].split('T')[0]})")
        q_play[_b] = 1 - u["p"]; q_lbl[_b] = f"starts if {u['starter'].split()[-1]} sits"
        print(f"qb_uncertain: {tm} {u['starter']} {u['p']:.0%} / {u['backup']}")
    NEWS_NEXT_MAN = {tm: _find(n) for tm, n in _nw.get("next_man", {}).items()}
    print("news overrides applied:", _nw["asof"], "| out:", len(_nw.get("out", [])), "| p_play:", len(_nw.get("p_play", {})))
except FileNotFoundError:
    pass
recent = pg[pg.season == SEASON].sort_values("gameday")
_POS = pg.drop_duplicates("pid", keep="last").set_index("pid").pos.to_dict()
rows = []
for _, g in games.iterrows():
    for tm in (g.away_team, g.home_team):
        tg = recent[recent.posteam == tm]
        order = tg.game_id.drop_duplicates().tolist(); last2 = order[-2:]
        pool = set(tg[tg.game_id.isin(last2)].pid) | {starters[tm]} | ({QB_BACKUP[tm]} if tm in QB_BACKUP else set())
        # only this week's quarterback(s): a replaced starter's recent dropbacks would otherwise give the team two passing lines
        _qbs_ok = {starters[tm]} | ({QB_BACKUP[tm]} if tm in QB_BACKUP else set())
        pool = {p for p in pool if _POS.get(p) != "QB" or p in _qbs_ok}
        # returning players: on the active roster, not ruled out, played for this team earlier this season but not in the
        # last two games - or haven't played yet this season and show up on this week's report as something other than Out
        back = set(tg.pid) - pool
        prev = pg[(pg.season == SEASON - 1) & (pg.posteam == tm)]
        back |= {p for p in set(prev.pid) & listed if roster_team.get(p) == tm} - set(tg.pid)
        back = {p for p in back if _POS.get(p) != "QB" or p in _qbs_ok}
        last_seen = tg.groupby("pid").game_id.last().map({gid: i for i, gid in enumerate(order)})
        for pid in pool | {p for p in back if p in roster_team.index}:
            if pid in unavailable or roster_team.get(pid, tm) != tm:
                continue
            missed = len(order) - 1 - last_seen[pid] if pid in last_seen.index else len(order)
            rows.append(dict(game_id=g.game_id, posteam=tm, pid=pid, gameday=pd.to_datetime(g.gameday),
                             season=SEASON, week=WEEK, missed=int(missed)))
pred = pd.DataFrame(rows)
pos = players.drop_duplicates("gsis_id").set_index("gsis_id")
pred["pos"] = pred.pid.map(pos.position).fillna("UNK")
pred["name"] = pred.pid.map(pos.display_name)
pred = pred[(pred.pos != "QB") | pred.pid.isin(starter_ids)]

d = T.features(pg, team, ctx, predict_rows=pred)

# ---- next man up: when a team's lead back (40%+ forward carry share) is Out or under 70% to play, the depth back with no
# recent usage is invisible to the pool. Add the top healthy roster RB, seeded at the share such backs actually took when a
# lead back sat: 12% of carries on average over 271 cases 2016-2025 (tests/t_rb_inherit.py). Proportional redistribution
# among known backs was confirmed there; an inheritance bonus for the top known back tested worse and is not used.
SEED = dict(pr_car_sh=0.12, pr_i10_sh=0.12, pr_i5_sh=0.12, pr_tgt_sh=0.04, pr_rz_sh=0.04, pr_ez_sh=0.02, rtd_sh_s=0.12, ptd_sh_s=0.04)
_dp = d[d._pred]
_ros_full = pd.read_parquet("data/rosters_2026.parquet")
_ros_full = _ros_full[_ros_full.week <= WEEK].sort_values("week").drop_duplicates("gsis_id", keep="last")
_rb_cand = _ros_full[(_ros_full.status == "ACT") & (_ros_full.position == "RB")]
seeded_notes = {}
_add = []
for _tm in pred.posteam.unique():
    _rbs = _dp[(_dp.posteam == _tm) & _dp.pos.isin(["RB", "FB"]) & _dp.pr_car_sh.notna()]
    absent = None
    if len(_rbs):
        _ld = _rbs.loc[_rbs.pr_car_sh.idxmax()]
        if _ld.pr_car_sh >= 0.40 and q_play.get(_ld.pid, 1.0) < 0.70: absent = _ld.pid
    if absent is None:   # lead ruled out entirely: top share this season among team RBs no longer in the pool
        _h = d[(~d._pred) & (d.posteam == _tm) & (d.season == SEASON) & d.pos.isin(["RB", "FB"])]
        if len(_h):
            _lastrow = _h.sort_values("gameday").groupby("pid").tail(1)
            _gone = _lastrow[~_lastrow.pid.isin(set(_dp[_dp.posteam == _tm].pid)) & (_lastrow.pr_car_sh >= 0.40)]
            if len(_gone): absent = _gone.sort_values("pr_car_sh").iloc[-1].pid
    if absent is None: continue
    # only when the KNOWN depth behind him is thin: with a real backup already in the pool, the committee absorbs the
    # work and an added body just dilutes (newcomers averaged 5% of carries with 25%+ of known depth remaining)
    _known_rest = _dp[(_dp.posteam == _tm) & _dp.pos.isin(["RB", "FB"]) & (_dp.pid != absent)].pr_car_sh.fillna(0).sum()
    if _known_rest >= 0.25: continue
    _dtm = _dp[_dp.posteam == _tm]
    _inpool = set(_dtm[((_dtm.pr_car_sh.fillna(0) + _dtm.pr_tgt_sh.fillna(0)) >= MIN_USAGE) | (_dtm.pos == "QB")].pid)
    _c = _rb_cand[(_rb_cand.team == _tm) & ~_rb_cand.gsis_id.isin(_inpool) & ~_rb_cand.gsis_id.isin(unavailable)]
    _c = _c[~_c.gsis_id.map(q_play).fillna(1.0).lt(0.5)]
    if not len(_c): continue
    if NEWS_NEXT_MAN.get(_tm) is not None:
        _nmrow = _ros_full[_ros_full.gsis_id == NEWS_NEXT_MAN[_tm]]      # trust the news even when the roster file lags a trade
        if len(_nmrow): _c = _nmrow.iloc[[0]].copy(); _c["team"] = _tm
    try:   # order candidates by the team's actual depth chart when we have it; draft pedigree is the fallback
        _dc = pd.read_parquet("data/depth_charts.parquet")
        _rk = _dc[(_dc.team == _tm) & (_dc.pos_abb == "RB")].sort_values("dt").drop_duplicates("gsis_id", keep="last").set_index("gsis_id").pos_rank
        _c = _c.assign(_dcr=_c.gsis_id.map(_rk)).sort_values(["_dcr", "draft_number", "entry_year"], ascending=[True, True, False], na_position="last")
    except Exception:
        _c = _c.sort_values(["draft_number", "entry_year"], ascending=[True, False], na_position="last")
    _new = _c.iloc[0]
    _m_ex = d._pred & (d.pid == _new.gsis_id) & (d.posteam == _tm)
    if _m_ex.any():      # already has a (sub-threshold) row: raise his role to the seed in place
        for _k, _v in SEED.items(): d.loc[_m_ex, _k] = np.maximum(d.loc[_m_ex, _k].fillna(0), _v)
        for _a, _b in [("l_car_sh", "pr_car_sh"), ("l_i10_sh", "pr_i10_sh"), ("l_i5_sh", "pr_i5_sh"), ("l_rtd_sh", "rtd_sh_s"),
                       ("l_tgt_sh", "pr_tgt_sh"), ("l_rz_sh", "pr_rz_sh"), ("l_ez_sh", "pr_ez_sh"), ("l_ptd_sh", "ptd_sh_s")]:
            d.loc[_m_ex, _a] = np.log(d.loc[_m_ex, _b] + T.EPS)
        _tmpl = None
    else:
        _tmpl = (_rbs if len(_rbs) else _dtm).iloc[[0]].copy()
        _tmpl["pid"] = _new.gsis_id; _tmpl["pos"] = "RB"; _tmpl["pos_RB"] = 1; _tmpl["is_qb"] = 0; _tmpl["over_xtd"] = 0.0
        for _k, _v in SEED.items(): _tmpl[_k] = _v
        for _a, _b in [("l_car_sh", "pr_car_sh"), ("l_i10_sh", "pr_i10_sh"), ("l_i5_sh", "pr_i5_sh"), ("l_rtd_sh", "rtd_sh_s"),
                       ("l_tgt_sh", "pr_tgt_sh"), ("l_rz_sh", "pr_rz_sh"), ("l_ez_sh", "pr_ez_sh"), ("l_ptd_sh", "ptd_sh_s")]:
            _tmpl[_a] = np.log(SEED[_b] + T.EPS)
        for _k in ("pr_car", "pr_tgt"): _tmpl[_k] = 0.0
    _lead_nm = pos.display_name.get(absent, "the starter").split()[-1]
    seeded_notes[_new.gsis_id] = f"next back up · added at a 12% carry share while {_lead_nm} is out or in doubt"
    if _tmpl is not None: _add.append(_tmpl)
    print("next man up:", _tm, _new.full_name, "behind", pos.display_name.get(absent, absent))
if _add:
    d = pd.concat([d] + _add, ignore_index=True)
hist = d[~d._pred]
tmm = T.fit_team(team)
mr, mp = T.fit_players(hist, tmm)
f = T.predict(d[d._pred].copy(), tmm, mr, mp)
f["usage"] = f.pr_car_sh.fillna(0) + f.pr_tgt_sh.fillna(0)
f = f[(f.usage >= MIN_USAGE) | (f.pos == "QB")]
f["missed"] = f.pid.map(pred.drop_duplicates("pid").set_index("pid").missed).fillna(0)
f.loc[f.pos == "QB", "missed"] = 0
f["back_mult"] = f.missed.map(J.back_mult)
f["p_play"] = f.pid.map(q_play).fillna(1.0)
avail = f.p_play * f.back_mult            # expected share of his usual role: sits some of the time, eased back in after an absence
f["lam"] = f.lam * avail; f["lam_rush"] = f.lam_rush * avail; f["lam_rec"] = f.lam_rec * avail
# Players on a team must add up to that team's expected TDs. Rescale each team's pool
# (this also hands an injured player's share to his teammates, proportionally).
f["team_scale"] = f.exp_td / f.groupby("posteam").lam.transform("sum")
f["lam"] = f.lam * f.team_scale
f["lam_rush"] = f.lam_rush * f.team_scale * f.qb_mult
f["lam_rec"] = f.lam_rec * f.team_scale * f.qb_mult
# lam is now his expected TDs counting the games he sits; when he does play his rate is lam / p_play
for c in ["lam_rush", "lam_rec"]: f[c] = f[c] / f.p_play
f["p_td"] = 1 - np.exp(-f.lam / f.p_play)      # shown "if he plays" (books void the bet when a player is inactive)
f["name"] = f.pid.map(pos.display_name)
f["fair_odds"] = T.american(f.p_td)
f["flag"] = np.where(f.p_play < 1, f.pid.map(q_lbl).fillna("Q") + " · " + (f.p_play * 100).round().astype(int).astype(str) + "% to play · numbers assume he plays",
              np.where(f.name.isin(QUESTIONABLE), "questionable", np.where(f.missed >= 2, "back after " + f.missed.astype(int).astype(str) + " missed", "")))
for _pid, _note in seeded_notes.items(): f.loc[f.pid == _pid, "flag"] = _note
f["seeded"] = f.pid.isin(seeded_notes).astype(int)
opp = ctx.set_index(["game_id", "team"]).opp
f["opp"] = [opp.get((g, t)) for g, t in zip(f.game_id, f.posteam)]
cols = ["name", "pid", "game_id", "rush_share", "td_std", "xtd_std", "over_xtd", "pos", "posteam", "opp", "implied", "exp_td", "pr_car", "pr_tgt", "pr_car_sh", "pr_i10_sh",
        "pr_tgt_sh", "pr_rz_sh", "lam_rush", "lam_rec", "p_td", "fair_odds", "flag", "p_play", "missed", "back_mult", "seeded"]
res = f[cols].sort_values("p_td", ascending=False).reset_index(drop=True)
res.to_csv(f"td_props_{SEASON}_wk{WEEK}.csv", index=False)
print(f"{len(res)} players projected; team exp TDs range {f.exp_td.min():.2f}-{f.exp_td.max():.2f}")
print(res.head(30).round({"implied": 1, "exp_td": 2, "pr_car": 1, "pr_tgt": 1, "pr_car_sh": 2, "pr_i10_sh": 2,
                          "pr_tgt_sh": 2, "pr_rz_sh": 2, "lam_rush": 2, "lam_rec": 2, "p_td": 3, "fair_odds": 0})
      [["name", "pos", "posteam", "opp", "implied", "pr_car", "pr_tgt", "p_td", "fair_odds", "flag"]].to_string())
# sanity: expected TDs per team = sum of listed player lambdas vs team model
chk = f.groupby("posteam").agg(sum_lam=("lam", "sum"), team_exp=("exp_td", "first"))
print("\nteams most rescaled (scale < 0.85 or > 1.15 means the pool was thin or crowded):")
sc = f.groupby("posteam").team_scale.first().sort_values()
print(sc[(sc < 0.85) | (sc > 1.15)].round(2).to_dict())
