"""Build the NFL Insights page from this week's model outputs.
Usage: python build_site.py <week>
Reads int_props_2026_wk<week>.csv, td_props_2026_wk<week>.csv, proj_2026_wk<week>.json, lines_2026_wk<week>.json.
Writes site/paydirt.html.
"""
import json, os, sys, datetime as dt
import numpy as np, pandas as pd

SEASON = 2026
week = int(sys.argv[1]) if len(sys.argv) > 1 else 5

players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"]).drop_duplicates("gsis_id")
full = players.set_index("gsis_id").display_name

# ---------------- INT board
ip = pd.read_csv(f"int_props_{SEASON}_wk{week}.csv")
notes = {}
if os.path.exists(f"starter_notes_wk{week}.json"):
    notes = json.load(open(f"starter_notes_wk{week}.json"))
# season to date: interceptions and how many of his games had at least one (a game = 10+ pass attempts)
_ip = pd.read_parquet(f"data/pbp_{SEASON}.parquet", columns=["game_id", "season_type", "passer_player_id", "pass_attempt", "interception", "sack", "two_point_attempt"])
_ip = _ip[(_ip.season_type == "REG") & (_ip.pass_attempt == 1) & (_ip.sack != 1) & (_ip.two_point_attempt != 1) & _ip.passer_player_id.notna()]
_ig = _ip.groupby(["passer_player_id", "game_id"]).agg(att=("pass_attempt", "sum"), ints=("interception", "sum")).reset_index()
_ig = _ig[_ig.att >= 10]
season_int = _ig.groupby("passer_player_id").agg(gp=("game_id", "nunique"), gint=("ints", lambda x: int((x > 0).sum())), gint2=("ints", lambda x: int((x > 1).sum())), ints=("ints", "sum"), att=("att", "sum"))
int_rows = [dict(
    full=full.get(r.passer_player_id, r["name"]), team=r.posteam, matchup=r.matchup,
    day=pd.to_datetime(r.gameday).strftime("%a %b %-d"), p=float(r.p_int), exp_ints=float(r.exp_ints),
    gp=int(season_int.gp.get(r.passer_player_id, 0)), gint=int(season_int.gint.get(r.passer_player_id, 0)), gint2=int(season_int.gint2.get(r.passer_player_id, 0)),
    sints=int(season_int.ints.get(r.passer_player_id, 0)), satt=int(season_int.att.get(r.passer_player_id, 0)),
    iw=float(r.qb_iw_pct), spread=float(-r.exp_margin), note=notes.get(r.posteam, "")) for _, r in ip.iterrows()]

# ---------------- TD board
tp = pd.read_csv(f"td_props_{SEASON}_wk{week}.csv")
# season-to-date scoring: rushing + receiving TDs per game played (games = offensive snaps > 0, or any touch)
_cols = ["game_id", "season", "season_type", "rusher_player_id", "receiver_player_id", "rush_touchdown", "pass_touchdown",
         "rush_attempt", "pass_attempt", "two_point_attempt"]
_p = pd.read_parquet(f"data/pbp_{SEASON}.parquet", columns=_cols)
_p = _p[(_p.season_type == "REG") & (_p.two_point_attempt != 1)]
_r = _p[_p.rush_touchdown == 1][["game_id", "rusher_player_id"]].rename(columns={"rusher_player_id": "pid"})
_c = _p[(_p.pass_touchdown == 1) & _p.receiver_player_id.notna()][["game_id", "receiver_player_id"]].rename(columns={"receiver_player_id": "pid"})
_td = pd.concat([_r, _c]).dropna().groupby(["pid", "game_id"]).size().rename("tds").reset_index()
_touch = pd.concat([_p[["game_id", "rusher_player_id"]].rename(columns={"rusher_player_id": "pid"}),
                    _p[["game_id", "receiver_player_id"]].rename(columns={"receiver_player_id": "pid"})]).dropna()
_sn = pd.read_parquet("data/snaps.parquet", columns=["game_id", "season", "gsis_id", "offense_snaps"])
_sn = _sn[(_sn.season == SEASON) & (_sn.offense_snaps > 0)].rename(columns={"gsis_id": "pid"})[["game_id", "pid"]]
_gp = pd.concat([_touch, _sn]).drop_duplicates()
_gp = _gp.merge(_td, on=["pid", "game_id"], how="left").fillna({"tds": 0})
season_td = _gp.groupby("pid").agg(gp=("game_id", "nunique"), gtd=("tds", lambda s: int((s > 0).sum())), gtd2=("tds", lambda s: int((s > 1).sum())), tds=("tds", "sum"))

# opponent TDs allowed per game to the player's position (season to date; rank 1 = most allowed)
try:
    _dv = pd.read_parquet("data/td_dvp.parquet").set_index(["team", "pos"])
except FileNotFoundError:
    _dv = None
def _otd(opp, pos):
    if _dv is None or (opp, pos) not in _dv.index: return (None, None)
    v = _dv.loc[(opp, pos)]
    return (round(float(v.tdpg), 2), int(v["rank"]))

td_rows = [dict(name=r["name"], pos=r.pos, team=r.posteam, opp=r.opp, implied=float(r.implied), p=float(r.p_td),
                otdg=_otd(r.opp, r.pos)[0], otdrk=_otd(r.opp, r.pos)[1],
                gp=int(season_td.gp.get(r.pid, 0)), gtd=int(season_td.gtd.get(r.pid, 0)), gtd2=int(season_td.gtd2.get(r.pid, 0)), stds=int(season_td.tds.get(r.pid, 0)),
                xtd=None if pd.isna(r.xtd_std) else round(float(r.xtd_std), 1), due=None if pd.isna(r.xtd_std) else round(float(r.td_std - r.xtd_std), 1),
                fair=float(r.fair_odds), car=float(np.nan_to_num(r.pr_car)), tgt=float(np.nan_to_num(r.pr_tgt)),
                i10=float(np.nan_to_num(r.pr_i10_sh)), rz=float(np.nan_to_num(r.pr_rz_sh)),
                flag=r.flag if isinstance(r.flag, str) else "") for _, r in tp.iterrows()]

how = """
<div>
  <h2>Reading the charts</h2>
  <p>Each bar is how far that input moves the number when it's swapped for an average value. Bars add to 100%; hover for the move in points.</p>
</div>
<div>
  <h2>Accuracy</h2>
  <table class="mini"><thead><tr><th>Fantasy-point miss vs.</th><th class="r">Season avg</th><th class="r">Last 3</th></tr></thead><tbody>
    <tr><td>QB</td><td class="r num">−12%</td><td class="r num">−14%</td></tr>
    <tr><td>RB</td><td class="r num">−4%</td><td class="r num">−7%</td></tr>
    <tr><td>WR</td><td class="r num">−5%</td><td class="r num">−8%</td></tr>
    <tr><td>TE</td><td class="r num">−5%</td><td class="r num">−10%</td></tr></tbody></table>
  <p style="margin-top:8px">How much smaller the model's miss is than a player's season or last-three average, on 2021–26 games it hadn't seen. Anytime TD: 8% better than the average guess; interceptions: 2%.</p>
</div>
<div>
  <h2>Ranges</h2>
  <p>Every game is simulated 6,000 times, with teammates sharing one set of team volume so they rise and fall together. <b>Floor</b> and <b>Ceil</b> are the 10th and 90th percentiles (held-out games land inside 78–84% of the time, against 80%). <b>Boom</b> and <b>Bust</b> are the chances of a big or bad game; hover the column for the cutoffs.</p>
</div>
<div>
  <h2>Injuries</h2>
  <p>Out, Doubtful, IR and inactive players are removed and their volume goes to teammates. Questionable players sit in a share of the simulations based on their last practice (did not practice 48% to play, limited 72%, full 85%); their own numbers assume they play, and teammates' numbers already account for the chance they don't.</p>
  <details class="more-detail"><summary>More detail</summary>
  <p><b>Midweek</b>, before game statuses: did not practice 18% to play, limited 66%, full 88%. <b>Early week</b>, before any report: last week's Out players return 31% of the time, Doubtful 45%. <b>Back from injury</b>: eased in at 77–87% of usual volume for one game. <b>Next man up</b>: if a lead back is out and no backup has 25%+ of carries, the top healthy back is added at a 12% carry share. <b>Quarterback changes</b> move receivers' volume and efficiency, not who gets the targets. Listed players get 97% of team volume; the rest goes to depth players.</p>
  </details>
</div>
<div>
  <h2>Prices and DraftKings</h2>
  <p>DraftKings lines appear in small orange type under the projections, and anytime-TD prices next to the model's fair odds, once a game's prices are pulled — a snapshot, not live. Fair odds have no house margin; DK's do. <b>DK pts</b> uses DraftKings scoring (with 3-point yardage bonuses); <b>Value</b> is DK points per $1,000 of salary, <b>Ceil value</b> the same at the 90th percentile.</p>
</div>
<div>
  <h2>Terms</h2>
  <p><b>Snap %</b>: share of offensive snaps (arrows = 10+ point change last game). <b>WOPR</b>: 1.5 × target share + 0.7 × air-yards share. <b>PPR</b>: 1 per catch, 0.1 per yard, 6 per TD, 0.04 per passing yard, 4 per passing TD, −2 per INT; <b>Std</b> without catches. <b>Quality starts / Duds</b>: games at or above a startable score / at or below a droppable one — description only; they add nothing to the projection.</p>
</div>"""

# ---------------- Spreads and totals section (numbers from the walk-forward backtest, data/lines_bt_summary.json)
try:
    _bt = json.load(open("data/lines_bt_summary.json")); _s, _t = _bt["spreads"], _bt["totals"]
    _rec = (f"<p><b>Record.</b> On {_bt['games']:,} past games, DK's closing line has been slightly more accurate than the model on both margins and totals. When they differ by 3+ points, "
            f"the model's spread side covers {_s['w']/_s['n']:.1%} and its total side {_t['w']/_t['n']:.1%}, against 52.4% to break even — so game numbers here are context, not picks. "
            f"Game totals and moneylines are tracked as paper bets on the Record tab.</p>")
except FileNotFoundError:
    _rec = ""
how_games = f"""<div>
  <h2>Spreads, totals and team totals</h2>
  <p>Two regressions on every game since 2016 — one for the margin, one for the total — built from team ratings plus quarterback, weather, rest and injury adjustments (see the Spread and Total charts). Their main job is setting each team's expected points, which drive every touchdown chance and player's volume on the site. Margins typically land within 13 points of the model's number, totals within 13.3.</p>
  {_rec}
</div>
"""
_k = how.index("</div>") + len("</div>\n")
how = how[:_k] + how_games + how[_k:]

# ---------------- Team totals (market-implied vs model) for the LINES tab
_ln = json.load(open(f'lines_{SEASON}_wk{week}.json'))
_tt = tp.groupby("posteam")[["exp_td", "rush_share"]].first()
team_totals = []
for g in _ln["games"]:
    for tm, opp, sign, name, home in [(g["home"], g["away"], 1, g["home_name"], True), (g["away"], g["home"], -1, g["away_name"], False)]:
        mkt = None if g["spread"] is None or g["total"] is None else (g["total"] + sign * g["spread"]) / 2
        mdl = (g["m_total"] + sign * g["m_margin"]) / 2
        team_totals.append(dict(team=tm, name=name, opp=opp, home=home, day=g["day"], mkt=None if mkt is None else round(mkt, 1), model=round(mdl, 1),
                                exp_td=round(float(_tt.exp_td.get(tm, float("nan"))), 2), rush_share=round(float(_tt.rush_share.get(tm, float("nan"))), 2)))
_ln["team_totals"] = team_totals

# ---------------- Defense vs position
from dvp import dvp
dv = dvp([SEASON - 1, SEASON], pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]))
sch = pd.read_parquet("data/sched.parquet")
wk = sch[(sch.season == SEASON) & (sch.week == week)]
faces = {**dict(zip(wk.home_team, wk.away_team)), **dict(zip(wk.away_team, wk.home_team))}
def dvrows(season):
    t = dv[dv.season == season]
    return [dict(def_=r.defteam, opp=faces.get(r.defteam, ""), pos=r.pos, games=int(r.games),
                 **{k: round(float(r[k]), 2) for k in ["fp", "lg_fp", "td", "pass_att", "pass_yds", "pass_td", "ints",
                                                       "car", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"]},
                 fp_rank=int(r.fp_rank)) for _, r in t.iterrows()]
fix = lambda rows: [{("def" if k == "def_" else k): v for k, v in r.items()} for r in rows]
dvp_data = dict(cur=fix(dvrows(SEASON)), prev=fix(dvrows(SEASON - 1)),
                cur_label=f"{SEASON} season", prev_label=f"{SEASON - 1} season")

# ---------------- Projections + Clinton's Legends of the Fall roster (as of Week 1)
proj = json.load(open(f"proj_{SEASON}_wk{week}.json"))
ROSTER = [("Joe Burrow", "CIN"), ("C.J. Stroud", "HOU"), ("Bryce Young", "CAR"), ("Aaron Rodgers", "PIT"), ("Ty Simpson", "LA"),
          ("Breece Hall", "NYJ"), ("Jeremiah Love", "ARI"), ("Bhayshul Tuten", "JAX"), ("Blake Corum", "LA"), ("Tyjae Spears", "TEN"),
          ("Kenny Gainwell", "TB"), ("Zach Charbonnet", "SEA"), ("Woody Marks", "HOU"), ("Demond Claiborne", "MIN"),
          ("Puka Nacua", "LA"), ("Malik Nabers", "NYG"), ("Tee Higgins", "CIN"), ("Jaylen Waddle", "DEN"), ("Josh Downs", "IND"),
          ("Quentin Johnston", "LAC"), ("Romeo Doubs", "NE"), ("Tank Dell", "HOU"), ("Malik Washington", "MIA"),
          ("Travis Hunter", "JAX"), ("Antonio Williams", "WAS"), ("Sam LaPorta", "DET"), ("George Kittle", "SF"),
          ("Mark Andrews", "BAL"), ("Dalton Kincaid", "BUF")]
roster = [dict(name=n, team=t) for n, t in ROSTER]
# opponent's 2026 rank in PPR points allowed to each position (1 = gives up the most)
_dv = dv[dv.season == SEASON].set_index(["defteam", "pos"])
for p in proj["players"]:
    k = (p["opp"], p["pos"])
    p["opp_rank"] = int(_dv.fp_rank.get(k)) if k in _dv.index else None

# DraftKings main-slate salaries (dk_salaries_wk<week>.csv: name,salary) -> price and points per $1,000
try:
    _dk = pd.read_csv(f"dk_salaries_wk{week}.csv").set_index("name").salary.to_dict()
except FileNotFoundError:
    _dk = {}
for p in proj["players"]:
    sal = _dk.get(p["name"]); d = p.get("dk")
    pp = p.get("p_play", 1) or 1
    p["dk_sal"] = int(sal) if sal else None
    p["dk_pts"] = round(d["pts"], 1) if d else None        # already averaged over the games he sits
    p["dk_val"] = round(d["pts"] / (sal / 1000), 2) if d and sal else None
    p["dk_cval"] = round(d["ceil"] / (sal / 1000), 2) if d and sal else None
proj["dk_n"] = sum(1 for p in proj["players"] if p["dk_sal"])

# team stat lines for the GAME PROJ tab: sums of the player projections (the QB's passing line is the team's; the listed
# rushers are 97% of the team, so they are scaled back up), with a Questionable player counted at his chance of playing
_pt = {}
for p in proj["players"]:
    t = _pt.setdefault(p["team"], dict(att=0, cmp=0, pass_yds=0, pass_td=0, ints=0, sacks=0, car=0, rush_yds=0, qb=""))
    m, pp = p["mean"], (p.get("p_play", 1) or 1)
    if p["pos"] == "QB":
        for k in ["att", "cmp", "pass_yds", "pass_td", "ints", "sacks"]: t[k] += m[k] * pp      # two QBs when the starter is uncertain
        t["car"] += m["car"] * pp; t["rush_yds"] += m["rush_yds"] * pp
        if pp >= t.get("qb_pp", 0): t["qb"], t["qb_pp"] = p["name"] + ("" if pp >= 1 else f" ({round(pp * 100)}%)"), pp
    else:
        t["car"] += m["car"] * pp / 0.97; t["rush_yds"] += m["rush_yds"] * pp / 0.97
for tt in _ln["team_totals"]:
    s = _pt.get(tt["team"])
    if not s: continue
    etd = tt["exp_td"] if tt["exp_td"] == tt["exp_td"] else None
    tt.update(qb=s["qb"], pass_att=round(s["att"], 1), cmp=round(s["cmp"], 1), pass_yds=round(s["pass_yds"]), ints=round(s["ints"], 2), sacks=round(s["sacks"], 1),
              rush_att=round(s["car"], 1), rush_yds=round(s["rush_yds"]), plays=round(s["att"] + s["sacks"] + s["car"], 1), yds=round(s["pass_yds"] + s["rush_yds"]),
              rush_td=None if etd is None else round(etd * tt["rush_share"], 2), pass_td=None if etd is None else round(etd * (1 - tt["rush_share"]), 2))
    if etd is None: tt["exp_td"] = None; tt["rush_share"] = None

# sportsbook prices pulled from the odds service (props_log.csv = anytime TD, props_log_ou.csv = over/unders); latest pull per book
import re as _re
_norm = lambda s: _re.sub(r"\s+", " ", _re.sub(r"[^a-z ]", " ", _re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
_STAT = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds", "player_receptions": "rec"}
_lines, _tdpx, _asof = {}, {}, None
if os.path.exists("props_log_ou.csv"):
    _o = pd.read_csv("props_log_ou.csv"); _o = _o[(_o.season == SEASON) & (_o.week == week)].sort_values("pulled_at").drop_duplicates(["market", "book", "player"], keep="last")
    for r in _o.itertuples():
        _lines.setdefault(_norm(r.player), {}).setdefault(_STAT[r.market], {})[r.book] = dict(pt=float(r.point), o=float(r.over), u=float(r.under))
    if len(_o): _asof = _o.pulled_at.max()
if os.path.exists("props_log.csv"):
    _t = pd.read_csv("props_log.csv"); _t = _t[(_t.season == SEASON) & (_t.week == week)].sort_values("pulled_at").drop_duplicates(["market", "book", "player"], keep="last")
    for r in _t.itertuples(): _tdpx.setdefault(_norm(r.player), {})[r.book] = float(r.price)
    if len(_t): _asof = max(_asof or "", _t.pulled_at.max())
for p in proj["players"]:
    k = _norm(p["name"]); p["lines"] = _lines.get(k); p["tdpx"] = _tdpx.get(k)
for r in td_rows:
    px = _tdpx.get(_norm(r["name"])) or {}
    r["dk"], r["fd"] = px.get("draftkings"), px.get("fanduel")
proj["prices_asof"] = _asof; proj["n_priced"] = sum(1 for p in proj["players"] if p["lines"] or p["tdpx"])

# bet board: only the bet types that have earned a place (see record.py verdicts). Yardage gaps are listed as information, not bets.
_QS = proj.get("quants")
def _p_over(arr, line):
    arr = np.asarray(arr, float)
    if line < arr[0]: return 1 - _QS[0] * (line / max(arr[0], 1e-9))
    for i in range(1, len(arr)):
        if line <= arr[i]:
            t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (_QS[i - 1] + t * (_QS[i] - _QS[i - 1]))
    return max(0, (1 - _QS[-1]) * .5)
# shown probabilities are blended with DraftKings in log-odds (tests/t_prop_calib.py, Oct 16): on weeks 4-6 props DK beat the model
# on Brier score and the best blend gave the model 0-20% weight. W_PROP = 0.2 is deliberately at the generous end while the sample is
# two weeks; refit from props_log*.csv as weeks accumulate. Selection still uses the raw model-vs-DK disagreement (it is information).
W_PROP = 0.2
def _blend(pm, pk):
    pm, pk = min(max(pm, 0.01), 0.99), min(max(pk, 0.01), 0.99); lo = lambda p: np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-(W_PROP * lo(pm) + (1 - W_PROP) * lo(pk)))))
board = dict(value=[], fades=[], rec=[], info=[])
for r in td_rows:
    if r.get("dk"):
        pb = _blend(r["p"], 1 / r["dk"]); ev = pb * r["dk"] - 1
        base = dict(name=r["name"], pos=r["pos"], team=r["team"], opp=r["opp"], p=round(pb, 3), p_model=round(r["p"], 3), fair=round(1 / pb, 2), dk=r["dk"], ev=round(ev, 3), flag=r.get("flag", ""))
        if 2.5 <= r["dk"] <= 3.9: board["value"].append(base)            # the price range Clinton bets: show every player, model view alongside
        elif 1.4 <= r["dk"] < 2.5 and ev <= -0.10: board["fades"].append(base)
for p in proj["players"]:
    for st, L_ in (p.get("lines") or {}).items():
        dk = L_.get("draftkings"); q = (p.get("q") or {}).get(st)
        if not dk or not q or not _QS: continue
        po = _p_over(q, dk["pt"]); mo = (1 / dk["o"]) / (1 / dk["o"] + 1 / dk["u"])
        side = "Over" if po > mo else "Under"; pb = _blend(po, mo); mp = pb if side == "Over" else 1 - pb; price = dk["o"] if side == "Over" else dk["u"]
        row = dict(name=p["name"], pos=p["pos"], team=p["team"], opp=p["opp"], stat=st, line=dk["pt"], model=round(p["mean"].get(st, 0), 1),
                   side=side, p=round(mp, 3), p_model=round(po if side == "Over" else 1 - po, 3), price=price, ev=round(mp * price - 1, 3), flag=p.get("flag", ""))
        if st == "rec":
            if abs(po - mo) >= 0.05: board["rec"].append(row)
        elif abs(p["mean"].get(st, 0) - dk["pt"]) >= (12 if st == "pass_yds" else 6): board["info"].append(row)
for k in board: board[k].sort(key=lambda x: -x["ev"] if k != "fades" else x["ev"])

# season against-the-spread and team-total records (closing lines; team total = (total ± spread)/2)
_s = sch[(sch.season == SEASON) & (sch.week < week) & (sch.game_type == "REG") & sch.result.notna() & sch.spread_line.notna() & sch.total_line.notna()]
ats_rec = {}
for r in _s.itertuples():
    for team, sgn, pts in ((r.home_team, 1, r.home_score), (r.away_team, -1, r.away_score)):
        d = ats_rec.setdefault(team, dict(cw=0, cl=0, cp=0, ow=0, ol=0, op=0))
        m = sgn * (r.result - r.spread_line); d["cw" if m > 0 else "cl" if m < 0 else "cp"] += 1
        tt = (r.total_line + sgn * r.spread_line) / 2; o = pts - tt; d["ow" if o > 0 else "ol" if o < 0 else "op"] += 1

# soft news: curated per-week notes (play-caller changes, retirements, roster churn) shown on the game cards
try:
    _nw = json.load(open(f"news_overrides_wk{week}.json"))
    from auto_overrides import fresh_notes
    news_notes = fresh_notes(week, _nw.get("team_notes", {})); news_asof = _nw.get("asof", "")   # notes older than 4 days are hidden (Oct 22)
except FileNotFoundError:
    news_notes = {}; news_asof = ""

# forecasts for the page (home-team keyed); dome games marked
try:
    _fc = pd.read_csv(f"forecasts_wk{week}.csv").set_index("team")
except FileNotFoundError:
    _fc = pd.DataFrame(columns=["wind", "temp", "precip", "source"])
_roof = sch[(sch.season == SEASON) & (sch.week == week)].set_index("home_team").roof
weather = {t: (dict(dome=True, retract=str(_roof.get(t)) == "closed") if str(_roof.get(t)) in ("dome", "closed") else
               (dict(wind=int(_fc.wind[t]), temp=int(_fc.temp[t]), precip=int(_fc.precip[t]), source=str(_fc.source[t])) if t in _fc.index else dict(unverified=True)))
           for t in _roof.index}

# line movement history for this week's games
try:
    _lh = pd.read_csv("data/line_history.csv"); _lh = _lh[_lh.game_id.str.startswith(f"{SEASON}_{week:02d}")].sort_values("captured_at")
    line_hist = {g: [dict(at=r.captured_at, spread=float(r.spread_line), total=float(r.total_line)) for _, r in x.iterrows()] for g, x in _lh.groupby("game_id")}
except FileNotFoundError:
    line_hist = {}

data = dict(season=SEASON, week=week,
            built=dt.datetime.now().strftime("%b %-d, %Y"),
            footer="Not betting advice.",
            int=int_rows, td=td_rows, dvp=dvp_data, lines=_ln, weather=weather, line_hist=line_hist, proj=proj, how=how,
            news_notes=news_notes, news_asof=news_asof, board=board, ats_rec=ats_rec, weights=json.load(open('weights.json')))
_pb = pd.read_parquet(f"data/pbp_{SEASON}.parquet", columns=["game_id", "week"])
data["games_in"] = int(_pb.game_id.nunique()); data["through"] = int(_pb.week.max())
data["n_games"] = len(_ln["games"])
os.makedirs("site", exist_ok=True)
json.dump(data, open(f"site/data_wk{week}.json", "w"), allow_nan=False)
print(f"site/data_wk{week}.json: {len(_ln['games'])} games, {len(int_rows)} QBs, {len(td_rows)} TD players")
