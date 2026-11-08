"""Staleness audit for week W: everything the model takes from outside feeds, checked against the newest source we have.
Run after the daily rebuild; any line it prints needs a look (or an override) before publishing."""
import os, sys, json, re, datetime as dt, pandas as pd, numpy as np
W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
site = json.load(open(f"site/data_wk{W}.json")); P = pd.DataFrame(site["proj"]["players"])
P["pts"] = [(r.get("mean") or {}).get("fp_ppr") if isinstance(r.get("mean"), dict) else None for r in site["proj"]["players"]]
P["pts"] = P.pts.fillna(0); P["p_play"] = P.p_play.fillna(1)
live = P[(P.p_play >= 0.5) & (P.pts >= 4)]
issues = []
def flag(kind, msg): issues.append((kind, msg))
# 1. schedule feed's listed QB vs last starter vs override
s = pd.read_parquet("data/sched.parquet"); sw = s[(s.season == 2026) & (s.week == W)]
pbp = pd.read_parquet("data/pbp_2026.parquet", columns=["game_id", "week", "posteam", "passer_player_name", "qb_dropback"])
last = pbp[pbp.qb_dropback.eq(1)].groupby(["posteam", "week", "passer_player_name"]).size().reset_index(name="n").sort_values("n").drop_duplicates(["posteam", "week"], keep="last").sort_values("week").drop_duplicates("posteam", keep="last").set_index("posteam").passer_player_name
try: ov = json.load(open(f"news_overrides_wk{W}.json"))
except FileNotFoundError: ov = {}; flag("OVERRIDES", f"news_overrides_wk{W}.json missing")
qs = ov.get("qb_start", {})
for r in sw.itertuples():
    for side in ("home", "away"):
        tm, lq = getattr(r, f"{side}_team"), getattr(r, f"{side}_qb_name"); prev = last.get(tm)
        # Nov 8 (the McKee miss): an override that disagrees with the feed is flagged every run until the news settles it;
        # keep it only with a dated source naming that QB (then list his name under "confirmed"), else delete it and let the feed decide
        if tm in qs and isinstance(lq, str) and qs[tm].split()[-1].lower() not in lq.lower() and lq.split()[-1].lower() not in qs[tm].lower():
            _d = json.load(open(f"news_dates_wk{W}.json")) if os.path.exists(f"news_dates_wk{W}.json") else {}
            _when = next((v for k, v in _d.items() if k.startswith(f"qb_start|{tm}|")), "?")
            if _when == "?" or (dt.date.today() - dt.date.fromisoformat(_when)).days > 2:
                flag("QB-CONFLICT", f"{tm}: qb_start says {qs[tm]} (dated {_when}) but the schedule feed lists {lq} - search the news; keep only with a dated source")
        if tm in qs or tm in ov.get("qb_uncertain", {}) or (isinstance(lq, str) and lq in ov.get("p_play", {})): continue   # confirmed starter, or the listed QB's chance is set explicitly
        if prev and isinstance(lq, str) and prev.split(".")[-1].lower() not in lq.lower():
            flag("QB", f"{tm}: feed lists {lq}, last starter {prev} — confirm and set qb_start")
# 1b. one passing line per team: quarterbacks' chances to play must add up to ~1 (two full-time QBs doubles every pass catcher)
qq = P[P.pos == "QB"].groupby("team").p_play.agg(["sum", "count"])
for tm, x in qq[(qq["sum"] > 1.05) | ((qq["count"] > 1) & (qq["sum"] < 0.95))].iterrows(): flag("QB", f"{tm}: {int(x['count'])} quarterbacks projected with chances summing to {x['sum']:.2f} (should be 1.00)")
# 2. weekly roster status: projected players on reserve/cut/inactive lists, or on a different team
ro = pd.read_parquet("data/rosters_2026.parquet"); rw = ro.sort_values("week").drop_duplicates("gsis_id", keep="last")   # latest listing per player (bye teams skip a week)
rw = rw.assign(n=rw.full_name.map(norm)).drop_duplicates("gsis_id")
st = rw.set_index("gsis_id")
news_on = {norm(n) for n in ov.get("back", [])} | {norm(n) for n, p in (ov.get("p_play") or {}).items() if p > 0}
for r in live.itertuples():
    if r.pid in st.index:
        x = st.loc[r.pid]
        if x.status in ("RES", "EXE") and norm(r.name) in news_on: continue   # activated by the news ahead of the roster file (Oct 22)
        if x.status in ("RES", "CUT", "INA", "RET", "EXE", "TRD"): flag("ROSTER", f"{r.name} ({r.team}) projected {r.pts:.1f} pts but roster status {x.status} ({x.status_description_abbr})")
        elif x.team != r.team: flag("TEAM", f"{r.name} projected for {r.team} but roster lists {x.team}")
    else: flag("ROSTER", f"{r.name} ({r.team}) not on any 2026 roster")
# 2b. trades: players whose latest roster team differs from their team in recent games, with real usage, missing from the new team's projections
lastteam = pd.read_parquet("data/pbp_2026.parquet", columns=["week", "posteam", "receiver_player_id", "rusher_player_id"])
use = pd.concat([lastteam[["week", "posteam", "receiver_player_id"]].rename(columns={"receiver_player_id": "pid"}), lastteam[["week", "posteam", "rusher_player_id"]].rename(columns={"rusher_player_id": "pid"})]).dropna()
cnt = use.groupby(["pid", "posteam"]).size().reset_index(name="touches").sort_values("touches").drop_duplicates("pid", keep="last")
projected = set(P.pid)
for r in cnt[cnt.touches >= 8].itertuples():
    _playing = set(sw.home_team) | set(sw.away_team)       # a new team on bye this week has no projections to be missing from
    if r.pid in st.index and st.loc[r.pid].team != r.posteam and st.loc[r.pid].status == "ACT" and r.pid not in projected and st.loc[r.pid].team in _playing:
        flag("TRADE", f"{st.loc[r.pid].full_name} moved {r.posteam} -> {st.loc[r.pid].team} ({r.touches} targets+carries with {r.posteam}) and is not in the {st.loc[r.pid].team} projections")
# 3. official injury report (this week) vs projection
inj = pd.read_parquet("data/injuries_2026.parquet"); iw = inj[inj.week == W]
if len(iw):
    ii = iw.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    for r in live.itertuples():
        if r.pid in ii.index:
            x = ii.loc[r.pid]; rs, ps = str(x.report_status), str(x.practice_status)
            if rs in ("Out", "Doubtful") and r.p_play > 0.3: flag("INJURY", f"{r.name} ({r.team}) report '{rs}' but p_play {r.p_play:.2f}")
            elif "Did Not" in ps and r.p_play >= 0.9 and "resting" not in str(x.practice_primary_injury).lower(): flag("INJURY", f"{r.name} ({r.team}) practice '{ps}' but projected as healthy (p_play {r.p_play:.2f})")
else: flag("INJURY", f"no week {W} injury report in the feed yet (normal before Wednesday)")
# 4. last week's absences not re-addressed this week
try:
    prev = json.load(open(f"news_overrides_wk{W-1}.json")); cur = set(ov.get("out", [])) | set(ov.get("p_play", {})) | set(ov.get("back", []))   # 'back' = confirmed returned
    for nm in prev.get("out", []):
        m = live[live.name.map(norm) == norm(nm)]
        if nm not in cur and len(m): flag("CARRYOVER", f"{nm} was out last week, now projected {m.pts.iloc[0]:.1f} pts with no Week {W} status — confirm he's back")
    for nm in prev.get("ol_out", []):
        if nm not in ov.get("ol_out", []) and nm not in ov.get("back", []): flag("CARRYOVER", f"OL {nm} was out last week, not listed this week — confirm")
except FileNotFoundError: pass
# 4a. (Nov 8, McCaffrey) a returning lead back needs lead_back: otherwise the share-out splits carries with his fill-in.
#     Returning = led his team in carries over his last 3 games played, missed the team's latest game, projected to play now.
try:
    _pb = pd.read_parquet("data/pbp_2026.parquet", columns=["week", "posteam", "rusher_player_id", "rusher_player_name"]).dropna(subset=["rusher_player_id"])
    _lb = ov.get("lead_back", {}) or {}
    for tm in set(sw.home_team) | set(sw.away_team):
        _t = _pb[_pb.posteam == tm]
        if _t.empty or tm in _lb or tm in (ov.get("committee") or []): continue   # "committee": [team codes] = news says shared backfield
        _wks = sorted(_t.week.unique()); _lastwk = _wks[-1]
        _c = _t.groupby(["rusher_player_id", "week"]).size().reset_index(name="c")
        for pid, d in _c.groupby("rusher_player_id"):
            d = d.sort_values("week"); last3 = d.tail(3)
            if d.week.max() >= _lastwk or len(last3) < 2: continue          # played the latest game, or too little history
            tot = _c[_c.week.isin(last3.week)].groupby("week").c.sum()
            if (last3.set_index("week").c / tot).mean() < 0.45: continue      # was not the lead back
            m = P[(P.pid == pid) & (P.p_play >= 0.5)]
            if len(m): flag("LEAD", f"{m.name.iloc[0]} ({tm}) returns after missing week {int(_lastwk)} (was the lead back) and {tm} has no lead_back - set it from the news (or list {tm} under 'committee' if the news says shared)")
except Exception as e: flag("LEAD", f"returning-back check skipped: {e}")
# 4c. (Nov 8, Joe Alt / Ben Brown) an ol_out lineman who is active on the roster and absent from this week's injury report
try:
    _ros = ro.sort_values("week").drop_duplicates("gsis_id", keep="last")
    for nm in ov.get("ol_out", []):
        x = _ros[_ros.full_name.map(norm) == norm(nm)]
        if len(x) and x.status.iloc[0] == "ACT" and x.team.iloc[0] in set(iw.team) and not (iw.gsis_id == x.gsis_id.iloc[0]).any():
            flag("OL", f"{nm} ({x.team.iloc[0]}) is in ol_out but active on the roster and not on the {x.team.iloc[0]} injury report - verify or remove")
except Exception as e: flag("OL", f"ol_out check skipped: {e}")
# 4b. per-entry staleness (Oct 22): chances, lead backs and uncertain QBs set more than 3 days ago
try:
    from auto_overrides import stale_lines
    for m in stale_lines(W): flag("STALE", m)
except Exception as e: flag("STALE", f"stale check skipped: {e}")
# 5. feed freshness
now = pd.Timestamp.now(tz="UTC")
asof = ov.get("asof")
if asof:
    a = pd.Timestamp(asof.replace(" ET", "")).tz_localize("America/New_York")
    if (now - a) > pd.Timedelta(hours=48): flag("FRESHNESS", f"overrides asof {asof} is older than 48h")
dc = pd.read_parquet("data/depth_charts.parquet").dt.max()
if (now - pd.Timestamp(dc)) > pd.Timedelta(hours=36): flag("FRESHNESS", f"depth charts last updated {dc}")
lh = pd.read_csv("data/line_history.csv"); cap = pd.to_datetime(lh.captured_at.max(), utc=True)
if (now - cap) > pd.Timedelta(hours=36): flag("FRESHNESS", f"spread/total feed last captured {cap}")
done_wk = s[(s.season == 2026) & s.result.notna()].week.max()
if pbp.week.max() < done_wk: flag("FRESHNESS", f"play-by-play only through week {pbp.week.max()}, games finished through week {done_wk}")
# DFS salaries: Clinton attaches DK's salary export early in the week; until then the site's value columns are blank
if not os.path.exists(f"dk_salaries_wk{W}.csv"):
    flag("DFS", f"no dk_salaries_wk{W}.csv - ask Clinton to attach DraftKings' salary CSV for the Sunday main slate")
# 6. weather (Oct 31, Clinton: "no forecast data so close to the games" is a flaw). Every unplayed OUTDOOR game kicking off within
#    72 hours needs a row in forecasts_wk{W}.csv, refreshed today (MySportsWeather primary, Covers cross-check; see UPDATE_CHECKLIST step 4).
import os
_now_et = pd.Timestamp.now(tz="America/New_York")
_ko = lambda r: pd.Timestamp(f"{r.gameday} {r.gametime}").tz_localize("America/New_York")
_soon = [r for r in sw.itertuples() if pd.isna(r.result) and str(r.roof) in ("outdoors", "open") and pd.Timedelta(0) < _ko(r) - _now_et < pd.Timedelta(hours=72)]
if _soon:
    try: _fc = pd.read_csv(f"forecasts_wk{W}.csv")
    except FileNotFoundError: _fc = None
    if _fc is None:
        flag("WEATHER", f"{len(_soon)} outdoor game(s) kick off within 72h and forecasts_wk{W}.csv is missing: fetch MySportsWeather + Covers")
    else:
        _miss = [r.home_team for r in _soon if r.home_team not in set(_fc.team)]
        if _miss: flag("WEATHER", f"no forecast row for outdoor game(s) at {', '.join(_miss)} (kickoff within 72h)")
        _today = [_now_et.strftime(f"%b {_now_et.day}"), (_now_et - pd.Timedelta(days=1)).strftime("%b ") + str((_now_et - pd.Timedelta(days=1)).day)]
        if not _fc.source.astype(str).str.contains("|".join(_today)).any():
            flag("WEATHER", f"forecasts_wk{W}.csv was not refreshed today or yesterday (no '{_today[0]}' in its source notes): refetch")
# ROLE CHECK (Oct 17): DraftKings prices carry role news fast (Allgeier, R. White, McLaughlin, the Drew Lock miss). When DK implies a much
# bigger (or smaller) role than the model, it is usually the model that is stale: look at the news, adjust out/p_play/next_man, never the numbers.
try:
    _sd = json.load(open(f"site/data_wk{W}.json")); _td = _sd["td"] if isinstance(_sd["td"], list) else _sd["td"].get("rows", [])
    for r in _td:
        if r.get("dk") and r["dk"] <= 6.0:
            dkp = 1 / r["dk"] / 1.07; gap = r["p"] - dkp
            if gap <= -0.20: flag("ROLE", f"{r['name']} ({r['team']}) TD: model {r['p']:.0%} vs DK {dkp:.0%} - DK sees a bigger role (starter? injury ahead of him?)")
            elif gap >= 0.15 and r["dk"] <= 4.0: flag("ROLE", f"{r['name']} ({r['team']}) TD: model {r['p']:.0%} vs DK {dkp:.0%} - DK sees a smaller role (limited? benched?)")
    _pl = {p["name"]: p for p in _sd["proj"]["players"]}
    _u = pd.read_csv("props_log_ou.csv"); _u = _u[(_u.week == W) & (_u.market == "player_receptions")].drop_duplicates("player", keep="last")
    for r in _u.itertuples():
        p = _pl.get(r.player)
        if p and abs(r.point - p["mean"].get("rec", 0)) >= 2.0:
            flag("ROLE", f"{r.player} ({p['team']}) catches: model {p['mean'].get('rec', 0):.1f} vs DK line {r.point} - check role/targets news")
except Exception as e:
    flag("ROLE", f"role check skipped: {e}")
print(f"Staleness audit, week {W}: {len(issues)} item(s)")
for k_, m in issues: print(f"  [{k_}] {m}")
