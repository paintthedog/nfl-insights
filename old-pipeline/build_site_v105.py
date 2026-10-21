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
  <h2>Reading the weights</h2>
  <p>Each bar shows how much that input separates one player from another this week. We swap the input for an average value, rerun the model, and see how far each player's number moves. Bars add to 100%. Hover for the average move.</p>
  <p>It's not the same as what predicts best: Vegas lines barely separate teammates, but they set every team's baseline.</p>
</div>
<div>
  <h2>Accuracy</h2>
  <table class="mini"><thead><tr><th>Fantasy-point error vs.</th><th class="r">Season avg</th><th class="r">Last 3</th></tr></thead><tbody>
    <tr><td>QB</td><td class="r num">−12%</td><td class="r num">−14%</td></tr>
    <tr><td>RB</td><td class="r num">−4%</td><td class="r num">−7%</td></tr>
    <tr><td>WR</td><td class="r num">−5%</td><td class="r num">−8%</td></tr>
    <tr><td>TE</td><td class="r num">−5%</td><td class="r num">−10%</td></tr></tbody></table>
  <p style="margin-top:8px">Average miss on PPR points against simply using the player's season average or last three games, every game 2021–26 predicted by a model that hadn't seen it. Anytime TD: 8% better than guessing the average. Interceptions: 2% better. Rerun October 8.</p>
</div>
<div>
  <h2>Where the ranges come from</h2>
  <p>Every game is played out 6,000 times. In each run the two teams draw their pass attempts and carries together (a team that throws more runs less), each team's targets and carries are split among its players around their projected shares with extra noise for the role itself, yards come from heavy-tailed draws with a hot-or-cold day multiplier, the team's touchdown count is drawn and handed to players by their touchdown rates and that run's usage, and a few percent of player-games are cut short. A quarterback's passing line is the sum of what his receivers did in that run, so teammates rise and fall together.</p>
  <p><b>PPR</b> and <b>Std</b> are the average of those runs, so a touchdown is counted once and interceptions are in. <b>Floor</b> and <b>Ceiling</b> are the 10th and 90th percentiles; held-out games land inside that band 79% (QB), 78% (RB), 82% (WR) and 84% (TE) of the time against an 80% target. <b>Boom</b> is the chance of a 25+ point game for a QB, 20+ for an RB or WR, 15+ for a TE; <b>Bust</b> is 12 or fewer for a QB, 6 or fewer for an RB or WR, 4 or fewer for a TE. The simulated averages were at least as accurate as the direct projections on 2025–26 games.</p>
</div>
<div>
  <h2>Injuries</h2>
  <p><b>Out, Doubtful, injured reserve and game-day inactives</b> are removed, and their volume goes to teammates. <b>Questionable</b> players sit in a share of the simulated games set by their last practice: did not practise 48% to play, limited 72%, full 85% (2,027 cases since 2019). Their own numbers are shown as if they play, with a note icon giving the chance they do; teammates' numbers already include the runs where they sit. Those who do play show no usage drop worth modelling.</p>
  <p><b>Quality starts and duds</b> (Projections tab) are observed, not projected: a quality start is a PPR game at or above what the last startable player at the position scores in a typical week (QB 18.4, RB 11.1, WR 12.3, TE 10.1, from 2021–25 weekly cutoffs); a dud is at or below the last rosterable rank. Tested and kept out of the model: once you know a player's average, his past quality-start rate predicts nothing extra. Use them as context for how a projection has been earned — steadily or in spikes.</p>
  <p><b>Next man up</b>: when a lead back is out or under 70% to play and no listed backup has 25%+ of the carries, the top healthy running back on the roster is added at a 12% carry share, the share such backs have actually taken (271 cases, 2016–25). An inheritance bonus for the known backup tested worse than plain redistribution and is not used.</p>
  <p><b>First game back</b>: players returning from missed games are listed again and eased in, at 87% of their usual volume after one missed game, 85% after two or three, 77% after four or more. By the second game back they are at normal volume.</p>
  <p><b>Midweek</b>, once a team posts its first practice report but before game statuses, a player's chance to play comes from his latest practice: did not practise 18%, limited 66%, full 88% (how often players with that practice line played that week, 2019–25). "Not injury related – resting" counts as healthy. A team's practice report replaces last week's report and Monday news for that team.</p>
  <p><b>Early in the week</b>, before any practice report, last week's report is carried forward at the rate those players actually return the next game: 31% if he was Out, 45% if Doubtful, 52% if Questionable and sat. Those flags read "out last wk" or "sat last wk". Injuries from the most recent games don't show up until the team's first report.</p>
  <p><b>Quarterback changes</b>: when this week's likely starter is better or worse than the quarterbacks behind a team's recent numbers, receivers' targets, catches and yards move with the gap (each quarterback is rated on shrunk EPA per dropback). Who gets the targets is left alone: leaning on games with the same quarterback made share predictions worse, not better.</p>
  <p>The listed players on a team are scaled to 97% of its targets and carries; the rest goes to players who aren't listed. Tested and left out: handing an absent player's volume to specific teammates by position. It was no more accurate than spreading it across the team.</p>
</div>
<div>
  <h2>DraftKings value</h2>
  <p><b>DK pts</b> is the simulated average under DraftKings classic scoring: full PPR, 4 per passing TD, −1 per interception, and the 3-point bonuses for 300 passing, 100 rushing or 100 receiving yards, counted in the runs where they happen. Fumbles are not modelled. <b>Value</b> is DK points per $1,000 of salary; <b>Ceil value</b> uses the 90th-percentile score instead of the average, which is the number that matters in tournaments. A dash means no price was found for that player.</p>
</div>
<div>
  <h2>Sportsbook prices</h2>
  <p>Where a game's prices have been pulled, Projections shows the book's line in small orange type under the model's passing yards, rushing yards, receiving yards and receptions (DraftKings; hover for the over/under prices). ATD shows each book's anytime-touchdown price next to the model's fair odds. Prices are a snapshot from the time shown at the top of the tab, not live, and a game has none until it has been pulled. A book's price includes its margin; the model's fair odds do not.</p>
</div>
<div>
  <h2>Other tabs</h2>
  <p>Season and Awards each carry their own accuracy notes, tested on past seasons the same way. Trends marks the few situations with any edge left as thin edges; the rest are reference.</p>
</div>
<div>
  <h2>Terms</h2>
  <p><b>Snap %</b>: share of offensive snaps; arrows mark a 10+ point change last game. <b>WOPR</b>: 1.5 × target share + 0.7 × air-yards share.
     Fair odds are decimal, no house margin. <b>PPR</b>: 1 per catch, 0.1 per rush/rec yard, 6 per rush/rec TD, 0.04 per passing yard, 4 per passing TD, −2 per INT. <b>Std</b>: same without catch points.</p>
</div>"""

# ---------------- Spreads and totals section (numbers from the walk-forward backtest, data/lines_bt_summary.json)
try:
    _bt = json.load(open("data/lines_bt_summary.json")); _s, _t = _bt["spreads"], _bt["totals"]
    _rec = (f"<p><b>How it has done.</b> Tested on {_bt['games']:,} games ({_bt['seasons']}), each predicted only from games before it. "
            f"Spreads: the closing line misses the final margin by {_s['mae_line']} points on average, the model by {_s['mae_model']}. When the two differ by 3+ points the model's side has covered "
            f"{_s['w']}–{_s['n']-_s['w']} ({_s['w']/_s['n']:.1%}) — no edge, and the best blend gives the model zero weight. "
            f"Totals: the line misses by {_t['mae_line']}, the model by {_t['mae_model']}; at 3+ point gaps the model's side is {_t['w']}–{_t['n']-_t['w']} ({_t['w']/_t['n']:.1%}), "
            f"just over the 52.4% break-even and unstable from year to year. So the game numbers on the cards are context, not picks. Game totals (3+ point gaps) and moneylines (4+ point gaps in win chance) "
            f"are tracked as paper bets on the Record tab.</p>")
except FileNotFoundError:
    _rec = ""
how_games = f"""<div>
  <h2>Spreads, totals and team totals</h2>
  <p>The game model is two regressions fit on every game since 2016: one for the home team's final margin, one for total points. Its most important job isn't picking sides — it sets each team's expected points, which drive every touchdown chance and every player's volume on the site.</p>
  <p><b>What moves it.</b> The Spread and Total charts above show how much each input typically moves a game's number. A few anchors: a 0.10 edge in offensive EPA per play is worth about 6 points of margin; a backup quarterback costs about 2.6 points per 0.10 EPA per dropback; each defensive regular out costs about 0.8; wind takes about 0.3 points off the total per mile an hour, and sub-freezing temperatures about 3.6.</p>
  <p><b>From numbers to chances.</b> Real margins land within about 13.0 points of the model's number two-thirds of the time, and totals within about 13.3. Those spreads turn the model's margin and total into the cover, over and win chances shown on each game card. Each team's own total is the model total split by the margin, with a typical miss of about 9.3 points.</p>
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
board = dict(value=[], fades=[], rec=[], info=[])
for r in td_rows:
    if r.get("dk"):
        ev = r["p"] * r["dk"] - 1
        base = dict(name=r["name"], pos=r["pos"], team=r["team"], opp=r["opp"], p=round(r["p"], 3), fair=round(1 / r["p"], 2), dk=r["dk"], ev=round(ev, 3), flag=r.get("flag", ""))
        if 2.5 <= r["dk"] <= 3.9: board["value"].append(base)            # the price range Clinton bets: show every player, model view alongside
        elif 1.4 <= r["dk"] < 2.5 and ev <= -0.10: board["fades"].append(base)
for p in proj["players"]:
    for st, L_ in (p.get("lines") or {}).items():
        dk = L_.get("draftkings"); q = (p.get("q") or {}).get(st)
        if not dk or not q or not _QS: continue
        po = _p_over(q, dk["pt"]); mo = (1 / dk["o"]) / (1 / dk["o"] + 1 / dk["u"])
        side = "Over" if po > mo else "Under"; mp = po if side == "Over" else 1 - po; price = dk["o"] if side == "Over" else dk["u"]
        row = dict(name=p["name"], pos=p["pos"], team=p["team"], opp=p["opp"], stat=st, line=dk["pt"], model=round(p["mean"].get(st, 0), 1),
                   side=side, p=round(mp, 3), price=price, ev=round(mp * price - 1, 3), flag=p.get("flag", ""))
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
    news_notes = _nw.get("team_notes", {}); news_asof = _nw.get("asof", "")
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
