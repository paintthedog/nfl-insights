"""Trend tables for the TRENDS tab. Everything is computed from the schedule (scores, closing lines, moneylines, coaches,
quarterbacks, referees, rest, weather) and play-by-play already in ./data. Writes trends.json.
A table = dict(title, note, cols=[...], rows=[[cell, ...]]); a cell is text, a number, or a record dict(w, l, p, pct)."""
import json, glob, numpy as np, pandas as pd
SEASON, FIRST = 2026, 2016
BE = 52.4
S = pd.read_parquet("data/sched.parquet")
S = S[(S.game_type == "REG") & (S.season >= FIRST)].copy()
S["surface"] = S.surface.fillna("").str.strip()
# the schedule file carries last year's coach forward for some teams; corrections live in coach_fix.json
import os
if os.path.exists("coach_fix.json"):
    for yr, fx in json.load(open("coach_fix.json")).items():
        for tm, c in fx.items():
            S.loc[(S.season == int(yr)) & (S.home_team == tm), "home_coach"] = c; S.loc[(S.season == int(yr)) & (S.away_team == tm), "away_coach"] = c
G = S[S.result.notna() & S.spread_line.notna() & S.total_line.notna()].copy()
G["sp"] = G.spread_line.abs(); G["fav_m"] = np.where(G.spread_line >= 0, G.result, -G.result)
G["fav_cover"] = np.sign(G.fav_m - G.sp); G["over"] = np.sign(G.total - G.total_line); G["home_cover"] = np.sign(G.result - G.spread_line)
G["hour"] = G.gametime.fillna("13:00").str[:2].astype(int)
H = G[G.season < SEASON]          # the ten completed seasons
YRS = f"{FIRST}–{SEASON - 1}"

def rec(sign, be=BE):
    sign = pd.Series(sign); w, l, p = int((sign > 0).sum()), int((sign < 0).sum()), int((sign == 0).sum())
    return dict(w=w, l=l, p=p, pct=round(w / (w + l) * 100, 1) if w + l else None, be=be)
def su(sign): return rec(sign, None)
def r1(x, d=1): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)
def pctf(x): return None if x is None or np.isnan(x) else f"{x * 100:.1f}%"

# ---------- one row per team per game
def long(df):
    a = df.assign(team=df.home_team, opp=df.away_team, home=True, pf=df.home_score, pa=df.away_score, line=df.spread_line, ml=df.home_moneyline, coach=df.home_coach, qb=df.home_qb_name, rest=df.home_rest)
    b = df.assign(team=df.away_team, opp=df.home_team, home=False, pf=df.away_score, pa=df.home_score, line=-df.spread_line, ml=df.away_moneyline, coach=df.away_coach, qb=df.away_qb_name, rest=df.away_rest)
    t = pd.concat([a, b]); t["margin"] = t.pf - t.pa; t["cover"] = np.sign(t.margin - t.line); t["win"] = np.sign(t.margin)
    return t.sort_values(["team", "season", "week"])
T = long(G)
g = T.groupby(["team", "season"])
T["prev_margin"] = g.margin.shift(1); T["prev_home"] = g.home.shift(1); T["prev2_home"] = g.home.shift(2); T["prev_wd"] = g.weekday.shift(1)
T["prev_line"] = g.line.shift(1); T["prev_win"] = g.win.shift(1)
T["meet"] = T.groupby(["team", "season", "opp"]).cumcount() + 1
WEST = {"LA", "LAC", "SF", "SEA", "LV", "ARI", "OAK", "SD"}
def three(mask, col="cover", per_game=False):
    out = []
    for yrs in ([y for y in range(FIRST, SEASON)], [SEASON - 1], [SEASON]):
        d = T[mask & T.season.isin(yrs)]
        if per_game: d = d.drop_duplicates("game_id")
        out.append(rec(d[col]))
    return out
out = {}

# ================= 1. LUCK
P26 = pd.read_parquet(f"data/pbp_{SEASON}.parquet", columns=["game_id", "season_type", "posteam", "defteam", "fumble", "fumble_lost", "interception", "field_goal_attempt", "field_goal_result", "play_type",
      "down", "ydstogo", "yardline_100", "wp", "two_point_attempt", "extra_point_attempt", "fixed_drive", "fixed_drive_result", "qtr", "total_home_score", "total_away_score", "home_team", "away_team"])
P26 = P26[P26.season_type == "REG"]
cur = long(S[(S.season == SEASON) & S.result.notna()].assign(spread_line=lambda x: x.spread_line.fillna(0)))
to_g = P26[P26.posteam.notna()].groupby("posteam").agg(giv=("interception", "sum"), fl=("fumble_lost", "sum"), fum=("fumble", "sum"))
to_t = P26[P26.defteam.notna()].groupby("defteam").agg(tk_i=("interception", "sum"), tk_f=("fumble_lost", "sum"), ofum=("fumble", "sum"))
fg = P26[P26.field_goal_attempt == 1].groupby("defteam").field_goal_result.agg(att="size", made=lambda x: (x == "made").sum())
rows = []
for tm, d in cur.groupby("team"):
    n, w, l, t_ = len(d), int((d.margin > 0).sum()), int((d.margin < 0).sum()), int((d.margin == 0).sum())
    pf, pa = d.pf.sum(), d.pa.sum(); pyth = pf ** 2.37 / (pf ** 2.37 + pa ** 2.37) * n
    cg = d[d.margin.abs() <= 8]
    tg_, tt = to_g.loc[tm] if tm in to_g.index else None, to_t.loc[tm] if tm in to_t.index else None
    tom = (tt.tk_i + tt.tk_f) - (tg_.giv + tg_.fl)
    allf = tg_.fum + tt.ofum; recd = (tg_.fum - tg_.fl) + tt.tk_f
    f = fg.loc[tm] if tm in fg.index else None
    rows.append([tm, f"{w}-{l}" + (f"-{t_}" if t_ else ""), int(pf - pa), r1(pyth), r1(w + 0.5 * t_ - pyth, 1), f"{int((cg.margin > 0).sum())}-{int((cg.margin < 0).sum())}", int(tom),
                 pctf(recd / allf) if allf else None, f"{int(f.made)}/{int(f.att)}" if f is not None else "—"])
rows.sort(key=lambda r: -(r[4] or 0))
luck_now = dict(title=f"{SEASON} luck by team", note="Sorted luckiest first. <b>Expected wins</b> come from points scored and allowed. <b>Wins over expected</b>, a lopsided <b>one-score record</b>, a big <b>turnover margin</b> and recovering well over half of all <b>fumbles</b> are the things that tend to drift back. Opponent FG = field goals opponents made against them.",
                cols=["Team", "Record", "Point diff", "Expected wins", "Wins over expected", "One-score games", "Turnover margin", "Fumbles recovered", "Opp. field goals"], rows=rows)
# does it reverse? first nine weeks -> rest of season, 2016-2025
A = long(H); fh = A[A.week <= 9].groupby(["team", "season"]).agg(n=("win", "size"), w=("win", lambda x: (x > 0).sum() + 0.5 * (x == 0).sum()), pf=("pf", "sum"), pa=("pa", "sum"),
       cgw=("margin", lambda m: ((m > 0) & (m.abs() <= 8)).sum()), cgn=("margin", lambda m: (m.abs() <= 8).sum()))
fh["luck"] = fh.w - fh.pf ** 2.37 / (fh.pf ** 2.37 + fh.pa ** 2.37) * fh.n; fh["wp1"] = fh.w / fh.n
sh = A[A.week >= 10].merge(fh[["luck", "wp1"]], left_on=["team", "season"], right_index=True)
q = fh.luck.quantile([.2, .8]); rows = []
for lab, m in [("Luckiest fifth (won 1+ more than their points say)", sh.luck >= q[.8]), ("Middle", (sh.luck > q[.2]) & (sh.luck < q[.8])), ("Unluckiest fifth", sh.luck <= q[.2])]:
    d = sh[m]; rows.append([lab, int(d.drop_duplicates(["team", "season"]).shape[0]), pctf(d.wp1.mean()), pctf(((d.win > 0).sum() + 0.5 * (d.win == 0).sum()) / len(d)), rec(d.cover), r1(d.margin.mean() - d.line.mean(), 2)])
s2 = A[A.week >= 10].groupby(["team", "season"]).agg(cgw=("margin", lambda m: ((m > 0) & (m.abs() <= 8)).sum()), cgn=("margin", lambda m: (m.abs() <= 8).sum()))
j = fh.join(s2, rsuffix="2"); j = j[(j.cgn >= 3) & (j.cgn2 >= 3)]
c1 = np.corrcoef(j.cgw / j.cgn, j.cgw2 / j.cgn2)[0, 1]
luck_hist = dict(title=f"Does luck reverse? Weeks 1–9 against the rest of the season, {YRS}", note=f"Teams are grouped by wins over expected through Week 9. A team's one-score record in the first half of a season has a correlation of {c1:+.2f} with its one-score record in the second half, which is to say none. Break-even against the spread is {BE}%.",
                 cols=["Group after Week 9", "Team-seasons", "Win % weeks 1–9", "Win % weeks 10+", "Against the spread, weeks 10+", "Avg margin vs spread"], rows=rows)
out["luck"] = dict(label="Luck", tables=[luck_now, luck_hist])

# ================= 2. TEASERS AND MONEYLINES
def leg(m): return rec(np.sign(m), 73.9)
rows = []
for lab, lo, hi in [("1 to 2.5", 1, 2.5), ("3", 3, 3), ("3.5 to 6.5", 3.5, 6.5), ("7", 7, 7), ("7.5 to 8.5", 7.5, 8.5), ("9 to 10", 9, 10), ("10.5 or more", 10.5, 99)]:
    d = H[H.sp.between(lo, hi)]; rows.append([lab, len(d), leg(d.fav_m - (d.sp - 6)), leg((d.sp + 6) - d.fav_m), rec(d.fav_cover), rec(-d.fav_cover)])
teas = dict(title=f"Six-point teaser legs by spread, {YRS}", note="A leg is one side moved six points in your favor. A two-team teaser at 1.83 (−120) needs each leg to win about <b>73.9%</b>; at 1.91 (−110), 72.4%. The textbook play is underdogs of 1–2.5 teased up through 3 and 7, and that one has cleared the bar. Favorites of 7.5–8.5 teased down have not.",
            cols=["Spread", "Games", "Favorite teased down 6", "Underdog teased up 6", "Favorite, no teaser", "Underdog, no teaser"], rows=rows)
rows = []
for lab, lo, hi in [("Under 41", 0, 40.5), ("41 to 44.5", 41, 44.5), ("45 to 48.5", 45, 48.5), ("49 or more", 49, 99)]:
    d = H[H.total_line.between(lo, hi)]; rows.append([lab, len(d), leg(d.total - (d.total_line - 6)), leg((d.total_line + 6) - d.total), rec(d.over)])
teas_t = dict(title="Six-point teaser legs on totals", note="Totals teasers have no key numbers working for them, and it shows.", cols=["Total", "Games", "Over teased down 6", "Under teased up 6", "Over, no teaser"], rows=rows)
def ml_profit(ml, won): return np.where(won > 0, np.where(ml > 0, ml / 100, 100 / ml.abs()), np.where(won == 0, 0, -1.0))
TH = T[(T.season < SEASON) & T.ml.notna()].copy(); TH["pl"] = ml_profit(TH.ml, TH.win); TH["dog"] = TH.line < 0; TH["sp"] = TH.line.abs(); rows = []
for lab, lo, hi in [("1 to 2.5", 1, 2.5), ("3", 3, 3), ("3.5 to 6.5", 3.5, 6.5), ("7 to 9.5", 7, 9.5), ("10 to 13.5", 10, 13.5), ("14 or more", 14, 99)]:
    d, f = TH[TH.dog & TH.sp.between(lo, hi)], TH[~TH.dog & TH.sp.between(lo, hi)]
    rows.append([lab, len(d), pctf((d.win > 0).mean()), r1(1 + (d.ml[d.ml > 0] / 100).mean() if (d.ml > 0).any() else np.nan, 2), f"{d.pl.mean() * 100:+.1f}%", f"{f.pl.mean() * 100:+.1f}%"])
mlt = dict(title=f"Moneylines by spread size, {YRS}", note="Return per unit staked on every underdog, and on every favorite, to win outright at closing prices. A flat bettor loses the book's margin (about 4%) with no edge, so anything better than about −4% is beating the average.",
           cols=["Spread", "Games", "Underdog wins", "Avg underdog price", "Underdog return", "Favorite return"], rows=rows)
out["teasers"] = dict(label="Teasers & moneylines", tables=[teas, teas_t, mlt])

# ================= 3. KEY NUMBERS
am = H.result.abs(); vc = am.value_counts(normalize=True); rows = [[int(k), pctf(v), pctf((am <= k).mean())] for k, v in vc.sort_values(ascending=False).head(12).items()]
kn = dict(title=f"Most common winning margins, {YRS}", note=f"{len(H)} games. Half a point matters most at 3, then 7, 6 and 10: that is how often the game lands exactly there.", cols=["Margin", "Share of games", "Decided by this or less"], rows=rows)
rows = [[lab, int(n), pctf(k)] for lab, n, k in [(f"{x}", (H.sp == x).sum(), ((H.sp == x) & (H.fav_m == x)).sum() / max((H.sp == x).sum(), 1)) for x in [3, 7, 6, 10, 4, 14]]]
kp = dict(title="Push rate when the spread sits on a key number", note="How often the favorite won by exactly the spread.", cols=["Spread", "Games", "Pushes"], rows=rows)
tv = H.total.value_counts(normalize=True).sort_values(ascending=False).head(12); rows = [[int(k), pctf(v)] for k, v in tv.items()]
kt = dict(title="Most common final totals", note="Totals cluster far less than margins. No number comes close to 3 or 7.", cols=["Total points", "Share of games"], rows=rows)
out["keys"] = dict(label="Key numbers", tables=[kn, kp, kt])

# ================= 4. COACHES AND QUARTERBACKS
act = S[(S.season == SEASON)]; coaches = sorted(set(act.home_coach.dropna()) | set(act.away_coach.dropna()))
team_of = {**dict(zip(act.home_coach, act.home_team)), **dict(zip(act.away_coach, act.away_team))}
rows = []
for c in coaches:
    d = T[T.coach == c]
    if len(d) < 8: continue
    rows.append([c, team_of.get(c, ""), len(d), su(d.win), rec(d.cover), rec(d[d.line > 0].cover), rec(d[d.line < 0].cover), rec(d[d.rest >= 13].cover), rec(d.over)])
rows.sort(key=lambda r: -(r[4]["pct"] or 0))
co = dict(title=f"Current head coaches against the spread, {FIRST}–{SEASON}", note="Coaches with at least eight games since 2016, all teams they have coached. Small samples swing wildly: a coach needs roughly 150 games before 55% means anything.",
          cols=["Coach", "Team", "Games", "Straight up", "Against the spread", "As favorite", "As underdog", "Off a bye", "Overs"], rows=rows)
lastq = T[T.season == SEASON].sort_values("week").groupby("team").tail(1).set_index("team").qb.to_dict(); rows = []
for tm, qn in sorted(lastq.items()):
    d = T[T.qb == qn]
    if len(d) < 4: rows.append([qn, tm, len(d), su(d.win), rec(d.cover), rec(d[d.line > 0].cover), rec(d[d.line < 0].cover), rec(d.over)]); continue
    rows.append([qn, tm, len(d), su(d.win), rec(d.cover), rec(d[d.line > 0].cover), rec(d[d.line < 0].cover), rec(d.over)])
rows.sort(key=lambda r: -(r[4]["pct"] or 0) if r[2] >= 20 else 1)
qbt = dict(title=f"Each team's most recent starting quarterback, {FIRST}–{SEASON}", note="Career starts since 2016. Quarterbacks with 20+ starts are sorted by record against the spread; the rest follow.",
           cols=["Quarterback", "Team", "Starts", "Straight up", "Against the spread", "As favorite", "As underdog", "Overs"], rows=rows)
T["first_qb"] = T.groupby("qb").cumcount() < 3
b = T[T.first_qb & (T.season >= FIRST + 1)]
bk = dict(title="A quarterback's first three career starts (since 2017)", note="Includes highly drafted rookies and emergency backups alike.", cols=["Situation", "Games", "Straight up", "Against the spread", "Overs"],
          rows=[["First three starts", len(b), su(b.win), rec(b.cover), rec(b.over)], ["... as an underdog of 3+", int((b.line <= -3).sum()), su(b[b.line <= -3].win), rec(b[b.line <= -3].cover), rec(b[b.line <= -3].over)]])
out["people"] = dict(label="Coaches & QBs", tables=[co, qbt, bk])

# ================= 5. REFEREES
pen = pd.concat([pd.read_parquet(f, columns=["game_id", "penalty", "penalty_yards"]) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
pg_ = pen[pen.penalty == 1].groupby("game_id").agg(pens=("penalty", "sum"), yds=("penalty_yards", "sum"))
R = G.merge(pg_, left_on="game_id", right_index=True, how="left"); lg_p, lg_y = R.pens.mean(), R.yds.mean()
this_wk = S[(S.season == SEASON) & S.result.isna() & S.referee.notna()]
nxt = {r.referee: f"{r.away_team} at {r.home_team} (wk {int(r.week)})" for r in this_wk.sort_values("week", ascending=False).itertuples()}
rows = []
for ref in sorted(set(S[S.season == SEASON].referee.dropna())):
    d = R[R.referee == ref]
    if len(d) < 16: continue
    rows.append([ref, len(d), rec(d.over), r1((d.total - d.total_line).mean()), rec(d.home_cover), rec(d.fav_cover), r1(d.pens.mean()), r1(d.yds.mean()), nxt.get(ref, "")])
rows.sort(key=lambda r: -(r[2]["pct"] or 0))
rf = dict(title=f"Referees working in {SEASON}: results in their games, {FIRST}–{SEASON}", note=f"League average is {lg_p:.1f} accepted penalties and {lg_y:.0f} penalty yards a game. A referee works about 16 games a season, so even ten years is a small sample; sorted by share of overs. Upcoming assignments show when the league has posted them.",
          cols=["Referee", "Games", "Overs", "Avg points vs total", "Home team vs spread", "Favorite vs spread", "Penalties / game", "Penalty yards / game", "Next game"], rows=rows)
R["pq"] = pd.qcut(R.groupby("referee").pens.transform("mean"), 4, labels=["Fewest flags", "Below average", "Above average", "Most flags"])
rows = [[str(k), len(d), r1(d.pens.mean()), rec(d.over), r1((d.total - d.total_line).mean(), 2), rec(d.home_cover)] for k, d in R.groupby("pq", observed=True)]
rq = dict(title="Do flag-happy crews change totals?", note="Referees grouped by their career penalties per game.", cols=["Referee group", "Games", "Penalties / game", "Overs", "Avg points vs total", "Home team vs spread"], rows=rows)
if not any(r[-1] for r in rf["rows"]): rf["cols"].pop(); rf["rows"] = [r[:-1] for r in rf["rows"]]
out["refs"] = dict(label="Referees", tables=[rf, rq])

# ================= 6. SCHEDULE SPOTS
def spot_list(T):
    return [("After winning by 17+", T.prev_margin >= 17), ("After losing by 17+", T.prev_margin <= -17), ("After a loss as a favorite", (T.prev_win < 0) & (T.prev_line > 0)), ("After a win as an underdog", (T.prev_win > 0) & (T.prev_line < 0)),
         ("Second straight road game", ~T.home & (T.prev_home == False)), ("Third straight road game", ~T.home & (T.prev_home == False) & (T.prev2_home == False)),
         ("Home after two straight on the road", T.home & (T.prev_home == False) & (T.prev2_home == False)),
         ("West Coast team, early kickoff on the road", T.team.isin(WEST) & ~T.home & (T.hour <= 13) & ~T.opp.isin(WEST)), ("Game after Monday night", T.prev_wd == "Monday"),
         ("Second meeting with a division rival", (T.div_game == 1) & (T.meet == 2)), ("Thursday road team", ~T.home & (T.weekday == "Thursday") & (T.rest <= 5)), ("Off a bye, on the road", ~T.home & (T.rest >= 13))]
spots = spot_list(T)
rows = []
for lab, m in spots:
    a, o = three(m), three(m, "over", per_game=True); rows.append([lab, a[0], o[0], a[1], a[2]])
out["spots"] = dict(label="Schedule spots", tables=[dict(title="Situational records", note=f"Records are for the team in the spot. The ten-year column is the test; break-even is {BE}%. Nothing here is strong enough to bet blind.",
                    cols=["Spot", f"Against the spread {YRS}", f"Overs {YRS}", f"{SEASON - 1} vs spread", f"{SEASON} vs spread"], rows=rows)])

# ================= 7. HOME FIELD
rows = [[int(y), len(d), pctf((d.result > 0).mean()), r1(d.result.mean(), 2), r1(d.spread_line.mean(), 2), rec(d.home_cover)] for y, d in G.groupby("season")]
hs = dict(title="Home-field advantage by season", note="Average home margin was near three points a decade ago. The market's average home spread shows what it now prices in.", cols=["Season", "Games", "Home team wins", "Avg home margin", "Avg home spread", "Home vs spread"], rows=rows)
L5 = T[T.season >= SEASON - 5]; rows = []
for tm, d in L5.groupby("team"):
    hm, aw = d[d.home], d[~d.home]; rows.append([tm, su(hm.win), r1(hm.margin.mean()), r1(aw.margin.mean()), r1((hm.margin.mean() - aw.margin.mean()) / 2), rec(hm.cover), rec(aw.cover)])
rows.sort(key=lambda r: -(r[4] or 0))
ht = dict(title=f"Home-field advantage by team, {SEASON - 5}–{SEASON}", note="Home edge = half the gap between a team's average margin at home and on the road, which removes how good the team is. About 45 home games each, so the extremes are partly noise.",
          cols=["Team", "Home record", "Avg margin at home", "Avg margin away", "Home edge (points)", "Home vs spread", "Away vs spread"], rows=rows)
out["home"] = dict(label="Home field", tables=[hs, ht])

# ================= 8. CONDITIONS
H2 = H.copy(); H2["turf"] = np.where(H2.surface.str.contains("grass"), "Grass", np.where(H2.surface == "", None, "Artificial turf"))
rows = [[k, len(d), rec(d.over), r1(d.total.mean()), r1(d.total_line.mean()), rec(d.fav_cover)] for k, d in H2.dropna(subset=["turf"]).groupby("turf")]
sf = dict(title=f"Playing surface, {YRS}", note="Turf games score more, and the total already knows it.", cols=["Surface", "Games", "Overs", "Avg points", "Avg total", "Favorite vs spread"], rows=rows)
od = H2[H2.roof.isin(["outdoors", "open"]) & H2.temp.notna()]; rows = []
for lab, lo, hi in [("Below 32°F", -50, 31), ("32 to 49°F", 32, 49), ("50 to 69°F", 50, 69), ("70 to 84°F", 70, 84), ("85°F and up", 85, 150)]:
    d = od[od.temp.between(lo, hi)]; rows.append([lab, len(d), rec(d.over), r1(d.total.mean()), r1(d.total_line.mean()), rec(d.fav_cover)])
tp = dict(title="Temperature, outdoor games", note="", cols=["Temperature", "Games", "Overs", "Avg points", "Avg total", "Favorite vs spread"], rows=rows)
rows = []
for lab, lo, hi in [("Under 5 mph", 0, 4), ("5 to 9 mph", 5, 9), ("10 to 14 mph", 10, 14), ("15 to 19 mph", 15, 19), ("20 mph and up", 20, 99)]:
    d = od[od.wind.between(lo, hi)]; rows.append([lab, len(d), rec(d.over), r1(d.total.mean()), r1(d.total_line.mean())])
wd = dict(title="Wind, outdoor games", note="Wind is measured at kickoff, so this is not quite a pre-game number. Unders at 10–19 mph are the only weather effect here that has beaten the total; the 20+ row is 41 games.", cols=["Wind", "Games", "Overs", "Avg points", "Avg total"], rows=rows)
rows = [[lab, len(d), rec(d.over), rec(d.fav_cover), rec(d.home_cover)] for lab, d in [("Sunday early (1 PM)", H[(H.weekday == "Sunday") & (H.hour <= 13)]), ("Sunday late afternoon", H[(H.weekday == "Sunday") & H.hour.between(14, 18)]),
        ("Sunday night", H[(H.weekday == "Sunday") & (H.hour >= 19)]), ("Monday night", H[H.weekday == "Monday"]), ("Thursday night", H[H.weekday == "Thursday"]), ("Saturday", H[H.weekday == "Saturday"])]]
ts = dict(title="Time slot", note="", cols=["Slot", "Games", "Overs", "Favorite vs spread", "Home vs spread"], rows=rows)
out["cond"] = dict(label="Conditions", tables=[sf, tp, wd, ts])

# ================= 9. TEAM TENDENCIES (this season)
q = P26.groupby(["game_id", "qtr"]).agg(h=("total_home_score", "max"), a=("total_away_score", "max"), ht=("home_team", "first"), awt=("away_team", "first")).reset_index().sort_values(["game_id", "qtr"])
q["hq"] = q.groupby("game_id").h.diff().fillna(q.h); q["aq"] = q.groupby("game_id").a.diff().fillna(q.a); q = q[q.qtr <= 4]
ql = pd.concat([q.assign(team=q.ht, pf=q.hq, pa=q.aq), q.assign(team=q.awt, pf=q.aq, pa=q.hq)])
ng = ql.groupby("team").game_id.nunique(); rows = []
for tm, d in ql.groupby("team"):
    by = d.groupby("qtr")[["pf", "pa"]].sum() / ng[tm]; f = lambda k: r1(by.pf.get(k, 0) - by.pa.get(k, 0))
    rows.append([tm, r1(by.pf.get(1, 0)), r1(by.pf.get(2, 0)), r1(by.pf.get(3, 0)), r1(by.pf.get(4, 0)), r1(by.pf.loc[by.index <= 2].sum() - by.pa.loc[by.index <= 2].sum()), r1(by.pf.loc[by.index >= 3].sum() - by.pa.loc[by.index >= 3].sum()), f(1), f(4)])
rows.sort(key=lambda r: -(r[5] or 0))
qt = dict(title=f"{SEASON} scoring by quarter, per game", note="Sorted by first-half margin. Four games a team is a small sample; treat it as a description of the season so far.", cols=["Team", "Q1 pts", "Q2 pts", "Q3 pts", "Q4 pts", "1st-half margin", "2nd-half margin", "Q1 margin", "Q4 margin"], rows=rows)
f4 = P26[(P26.down == 4) & P26.wp.between(.1, .9) & (P26.ydstogo <= 2) & P26.yardline_100.between(1, 60) & P26.play_type.isin(["run", "pass", "punt", "field_goal"])]
go = f4.groupby("posteam").play_type.agg(n="size", go=lambda x: x.isin(["run", "pass"]).sum())
tp2 = P26[(P26.two_point_attempt == 1) | (P26.extra_point_attempt == 1)].groupby("posteam").two_point_attempt.agg(n="size", two="sum")
dr = P26[P26.posteam.notna() & P26.fixed_drive.notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(rz=("yardline_100", lambda y: (y <= 20).any()), td=("fixed_drive_result", lambda r: (r == "Touchdown").any()), n=("yardline_100", "size"))
rzt = dr[dr.rz].groupby("posteam").td.agg(n="size", td="sum"); dall = dr.groupby("posteam").td.agg(n="size", td="sum")
rows = []
for tm in sorted(go.index.union(rzt.index)):
    g1 = go.loc[tm] if tm in go.index else None; t2 = tp2.loc[tm] if tm in tp2.index else None; z = rzt.loc[tm] if tm in rzt.index else None; da = dall.loc[tm]
    rows.append([tm, f"{int(g1.go)}/{int(g1.n)}" if g1 is not None else "—", pctf(g1.go / g1.n) if g1 is not None and g1.n else None, int(t2.two) if t2 is not None else 0, f"{int(z.td)}/{int(z.n)}" if z is not None else "—", pctf(z.td / z.n) if z is not None and z.n else None, pctf(da.td / da.n)])
rows.sort(key=lambda r: r[0])
lgo = go.go.sum() / go.n.sum(); lrz = rzt.td.sum() / rzt.n.sum()
dc = dict(title=f"{SEASON} decisions and finishing", note=f"Fourth-and-short = fourth down with 2 or fewer yards to go, outside the team's own 40, game still in the balance. League: goes for it {lgo:.0%} of the time; scores a touchdown on {lrz:.0%} of red-zone trips. Four games in, the fourth-down counts are tiny.",
          cols=["Team", "4th-and-short: went for it", "Go rate", "Two-point tries", "Red-zone TDs / trips", "Red-zone TD rate", "TD on share of all drives"], rows=rows)
out["teams"] = dict(label="Team tendencies", tables=[qt, dc])

# ================= 10. PLAYER ODDS BY SITUATION
cols = ["game_id", "season", "week", "season_type", "posteam", "play_type", "two_point_attempt", "rusher_player_id", "receiver_player_id", "rush_touchdown", "pass_touchdown", "rushing_yards", "receiving_yards", "complete_pass"]
pb = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet")) if int(f[-12:-8]) < SEASON])
pb = pb[(pb.season_type == "REG") & pb.play_type.isin(["run", "pass"]) & (pb.two_point_attempt == 0)]
ru = pb[pb.play_type == "run"].groupby(["game_id", "season", "week", "posteam", "rusher_player_id"]).agg(car=("play_type", "size"), ryd=("rushing_yards", "sum"), rtd=("rush_touchdown", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
rc = pb[pb.play_type == "pass"].groupby(["game_id", "season", "week", "posteam", "receiver_player_id"]).agg(tgt=("play_type", "size"), cyd=("receiving_yards", "sum"), ctd=("pass_touchdown", "sum")).reset_index().rename(columns={"receiver_player_id": "pid"})
pg = ru.merge(rc, on=["game_id", "season", "week", "posteam", "pid"], how="outer").fillna(0)
pos = pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]).drop_duplicates("gsis_id").set_index("gsis_id").position
pg["pos"] = pg.pid.map(pos).replace({"FB": "RB"}); pg = pg[pg.pos.isin(["RB", "WR", "TE"])].sort_values(["pid", "season", "week"])
pg["td"] = ((pg.rtd + pg.ctd) > 0).astype(int); pg["touch"] = pg.car + pg.tgt
gp = pg.groupby(["pid", "season"])
for c in ["touch", "ryd", "cyd"]: pg["pr_" + c] = gp[c].transform(lambda x: x.shift(1).expanding().mean())
pg["pr_n"] = gp.cumcount(); pg = pg[pg.pr_n >= 2]
imp = pd.concat([S.assign(posteam=S.home_team, implied=S.total_line / 2 + S.spread_line / 2)[["game_id", "posteam", "implied"]], S.assign(posteam=S.away_team, implied=S.total_line / 2 - S.spread_line / 2)[["game_id", "posteam", "implied"]]])
pg = pg.merge(imp, on=["game_id", "posteam"]).dropna(subset=["implied"])
pg["rk"] = pg.groupby(["game_id", "posteam", "pos"]).pr_touch.rank(ascending=False, method="first")
pg["ib"] = pd.cut(pg.implied, [0, 17.49, 20.99, 23.99, 26.99, 60], labels=["Under 17.5", "17.5 to 20.5", "21 to 23.5", "24 to 26.5", "27 or more"])
roles = [("Lead running back", (pg.pos == "RB") & (pg.rk == 1)), ("Second running back", (pg.pos == "RB") & (pg.rk == 2)), ("Top wide receiver", (pg.pos == "WR") & (pg.rk == 1)), ("Second wide receiver", (pg.pos == "WR") & (pg.rk == 2)),
         ("Third wide receiver", (pg.pos == "WR") & (pg.rk == 3)), ("Top tight end", (pg.pos == "TE") & (pg.rk == 1))]
rows = []
for lab, m in roles:
    d = pg[m]; rows.append([lab, len(d)] + [pctf(d[d.ib == b].td.mean()) for b in pg.ib.cat.categories] + [pctf(d.td.mean())])
tdr = dict(title=f"How often each role scores a touchdown, by the team's implied points, {YRS}", note="Role = rank on the team by touches per game so far that season (carries plus targets), from Week 3 on. Useful as a sanity check on the ATD tab: fair decimal odds are 1 divided by the rate.",
           cols=["Role", "Player-games"] + [f"Team implied {b}" for b in pg.ib.cat.categories] + ["All games"], rows=rows)
rows = []
for lab, lo, hi in [("40 to 54", 40, 54.99), ("55 to 69", 55, 69.99), ("70 to 84", 70, 84.99), ("85 to 99", 85, 99.99), ("100 or more", 100, 999)]:
    a, b = pg[(pg.pos == "RB") & pg.pr_ryd.between(lo, hi)], pg[(pg.pos != "RB") & pg.pr_cyd.between(lo, hi)]
    rows.append([lab, len(a), pctf((a.ryd >= 100).mean()) if len(a) else None, pctf((a.ryd >= lo).mean()) if len(a) else None, len(b), pctf((b.cyd >= 100).mean()) if len(b) else None, pctf((b.cyd >= lo).mean()) if len(b) else None])
ms = dict(title="Reaching 100 yards, by season average coming in", note="A back averaging 85–99 rushing yards reaches 100 well under half the time; a player's own average is closer to a coin flip than a floor. <b>Matches own average</b> = reaches the bottom of his bracket.",
          cols=["Season average (yards)", "RB games", "RB reaches 100 rushing", "RB matches own average", "WR/TE games", "WR/TE reaches 100 receiving", "WR/TE matches own average"], rows=rows)
out["players"] = dict(label="Player odds", tables=[tdr, ms])

# ================= verdicts: the answer in a sentence, and a tag saying how much weight it deserves
def V(t, tag, verdict): t["tag"] = tag; t["verdict"] = verdict
def z(r): n = r["w"] + r["l"]; return (r["pct"] - r["be"]) / 100 / np.sqrt(r["be"] / 100 * (1 - r["be"] / 100) / n)
top, bot = luck_now["rows"][0], luck_now["rows"][-1]
V(luck_now, "ref", f"{top[0]} has won {top[4]:.1f} more games than its points say and {bot[0]} {abs(bot[4]):.1f} fewer. Both should drift back toward their point differential.")
lh = luck_hist["rows"]
V(luck_hist, "priced", f"Lucky teams do fall back, from winning {lh[0][2]} of games to {lh[0][3]}, but they cover {lh[0][4]['pct']}% afterwards. The spread sees it coming.")
tr_ = teas["rows"]; dogleg = tr_[0][3]; clear = [r[0] for r in tr_ if r[2]["pct"] >= 73.9]
V(teas, "thin", f"Underdogs of 1 to 2.5 teased up six win {dogleg['pct']}% of legs against a 73.9% bar. " + (f"Favorites clear it only at a spread of exactly {' or '.join(clear)}. " if clear else "No favorite bucket clears it. ") + "Books that charge more than 1.83 for two teams erase the edge.")
V(teas_t, "bad", f"No total, teased either way, wins more than {max(max(r[2]['pct'], r[3]['pct']) for r in teas_t['rows'])}% of legs. The bar is 73.9%.")
big = [r for r in mlt["rows"] if r[0].startswith("10") or r[0].startswith("14")]
V(mlt, "bad", f"Underdogs of 10 or more return {big[0][4]} and {big[1][4]} on the moneyline. The smallest underdogs (1 to 2.5) lose the least, {mlt['rows'][0][4]}.")
V(kn, "ref", f"{kn['rows'][0][1]} of games are decided by exactly {kn['rows'][0][0]} and {kn['rows'][1][1]} by {kn['rows'][1][0]}. Half a point is worth the most around those two numbers.")
V(kp, "ref", f"A spread of exactly 3 pushes {kp['rows'][0][2]} of the time; a spread of 7, {kp['rows'][1][2]}.")
V(kt, "ref", f"Totals do not cluster. The most common final total, {kt['rows'][0][0]}, is {kt['rows'][0][1]} of games.")
vet = max([r for r in co["rows"] if r[2] >= 150], key=lambda r: r[4]["pct"])
V(co, "noise", f"Among coaches with 150+ games, the best is {vet[0]} at {vet[4]['pct']}% against the spread. Everyone above that on the list has a short track record.")
vq = max([r for r in qbt["rows"] if r[2] >= 100], key=lambda r: r[4]["pct"])
V(qbt, "noise", f"Among quarterbacks with 100+ starts, the best is {vq[0]} at {vq[4]['pct']}% against the spread. Shorter records above him are small samples.")
V(bk, "priced", f"A quarterback in his first three starts wins only {bk['rows'][0][2]['pct']}% of the time but covers {bk['rows'][0][3]['pct']}%. The spread already accounts for him.")
long_ = [r for r in rf["rows"] if r[1] >= 100]
V(rf, "noise", f"Referees with 100+ games range from {min(r[2]['pct'] for r in long_)}% to {max(r[2]['pct'] for r in long_)}% overs, which is about what chance alone produces.")
V(rq, "noise", f"Crews that throw the most flags: {rq['rows'][-1][3]['pct']}% overs. The fewest: {rq['rows'][0][3]['pct']}%. Penalties do not move totals.")
sp_ = out["spots"]["tables"][0]; pcs = [r[1]["pct"] for r in sp_["rows"]]
V(sp_, "noise", f"All {len(pcs)} spots land between {min(pcs)}% and {max(pcs)}% against the spread over ten years. None is worth a bet by itself.")
V(hs, "ref", f"Home teams won by {hs['rows'][-2][3]} points a game in {hs['rows'][-2][0]} and the market priced {hs['rows'][-2][4]}. Home field is worth about two points now, not three.")
V(ht, "noise", f"The largest measured home edge is {ht['rows'][0][0]} at {ht['rows'][0][4]} points, on about 45 home games. Too few to adjust a line by.")
V(sf, "priced", f"Turf games score more ({sf['rows'][0][3]} points against {sf['rows'][1][3]} on grass) and the total is already set higher.")
V(tp, "priced", f"Freezing games score {tp['rows'][0][3]} points against a total of {tp['rows'][0][4]}, but that is {tp['rows'][0][1]} games. The total already drops with the temperature.")
wu = rec(-od[od.wind.between(10, 19)].over)
V(wd, "holds" if z(wu) >= 2 else "thin", f"Unders hit {wu['pct']}% in outdoor games with 10 to 19 mph wind ({wu['w'] + wu['l']} games). Wind is the one weather effect the total has not fully absorbed.")
pr_ = H[((H.weekday == "Sunday") & (H.hour >= 19)) | (H.weekday == "Monday")]; pu = rec(-pr_.over)
V(ts, "holds" if z(pu) >= 2 else "thin", f"Sunday night and Monday night unders: {pu['pct']}% over {pu['w'] + pu['l']} games. Every other slot is a coin flip.")
V(qt, "ref", f"{qt['rows'][0][0]} has the best first-half margin ({qt['rows'][0][5]:+.1f} a game) and {qt['rows'][-1][0]} the worst ({qt['rows'][-1][5]:+.1f}). Four games each.")
V(dc, "ref", f"The league goes for it on {lgo:.0%} of fourth-and-shorts and scores a touchdown on {lrz:.0%} of red-zone trips.")
V(tdr, "ref", f"A lead back scores in {tdr['rows'][0][2]} of games when his team is implied under 17.5 points and {tdr['rows'][0][6]} at 27 or more. A top receiver: {tdr['rows'][2][2]} and {tdr['rows'][2][6]}.")
V(ms, "ref", f"A back averaging 85 to 99 rushing yards reaches 100 in {ms['rows'][3][2]} of games. A season average is nearer a coin flip than a floor.")
groups = dict(
    lines=dict(label="Spreads & totals", tables=[kn, kp, kt]),
    teasers=dict(label="Teasers & moneylines", tables=[teas, teas_t, mlt]),
    teams=dict(label="Teams", tables=[luck_now, luck_hist, hs, ht, qt, dc]),
    people=dict(label="Coaches, QBs & referees", tables=[co, qbt, bk, rf, rq]),
    spots=dict(label="Schedule & conditions", tables=[sp_, wd, ts, sf, tp]),
    players=dict(label="Player odds", tables=[tdr, ms]))
for k, gp_ in groups.items():
    for i, t in enumerate(gp_["tables"]): t["id"] = f"{k}{i}"

# ================= what the front page needs to match trends to this week's games
c1_, c2_, c3_ = H[(H.sp >= 7) & (H.total_line < 42)], H[(H.sp >= 7) & (H.total_line >= 48)], H[(H.sp < 3) & (H.total_line < 42)]
up = S[(S.season == SEASON) & S.result.isna()]; hr = up.gametime.fillna("13:00").str[:2].astype(int)
bigdog = TH[TH.dog & (TH.sp >= 10)]
# spots that apply to each team in games not yet played, with the ten-year record for that spot
S26 = S[S.season == SEASON].copy(); S26["hour"] = S26.gametime.fillna("13:00").str[:2].astype(int)
U = long(S26.assign(spread_line=S26.spread_line.fillna(0)))
gU = U.groupby(["team", "season"])
U["prev_margin"] = gU.margin.shift(1); U["prev_home"] = gU.home.shift(1); U["prev2_home"] = gU.home.shift(2); U["prev_wd"] = gU.weekday.shift(1)
U["prev_line"] = gU.line.shift(1); U["prev_win"] = gU.win.shift(1); U["meet"] = U.groupby(["team", "season", "opp"]).cumcount() + 1
spot_rec = {r[0]: (r[1], r[2]) for r in sp_["rows"]}; up_spots = {}
for lab, m in spot_list(U):
    for r in U[m & U.result.isna()].itertuples():
        up_spots.setdefault(r.game_id, {}).setdefault(r.team, []).append(dict(label=lab, ats=spot_rec[lab][0], ou=spot_rec[lab][1]))
def rank(rows, i, team, rev=True):
    v = sorted(rows, key=lambda r: -float(str(r[i] or 0).rstrip("%")) if rev else float(str(r[i] or 0).rstrip("%")))
    return [r[0] for r in v].index(team) + 1
facts = {}
for r in luck_now["rows"]: facts[r[0]] = dict(rec=r[1], woe=r[4], one=r[5], tom=r[6])
for r in dc["rows"]:
    if r[0] in facts and r[5]: facts[r[0]].update(rz=r[5], rz_rank=rank([x for x in dc["rows"] if x[5]], 5, r[0]))
for r in qt["rows"]:
    if r[0] in facts: facts[r[0]].update(h1=r[5], h1_rank=rank(qt["rows"], 5, r[0]), h2=r[6], h2_rank=rank(qt["rows"], 6, r[0]))
for r in co["rows"]:
    if r[1] in facts: facts[r[1]].update(coach=r[0], coach_n=r[2], coach_ats=r[4])
# pass and run efficiency this season, offense and defense: expected points added per play, ranked 1 (best) to 32
_e = pd.read_parquet(f"data/pbp_{SEASON}.parquet", columns=["season_type", "posteam", "defteam", "play_type", "epa", "yards_gained"])
_e = _e[(_e.season_type == "REG") & _e.play_type.isin(["pass", "run"]) & _e.epa.notna()]
for side, col in (("o", "posteam"), ("d", "defteam")):
    for kind in ("all", "pass", "run"):
        g = (_e if kind == "all" else _e[_e.play_type == kind]).groupby(col).agg(epa=("epa", "mean"), ypp=("yards_gained", "mean"), n=("epa", "size"))
        rk = g.epa.rank(ascending=(side == "d"), method="min")
        for tm_, r in g.iterrows():
            if tm_ in facts: facts[tm_][f"{side}_{kind}"] = dict(rank=int(rk[tm_]), epa=round(float(r.epa), 3), ypp=round(float(r.ypp), 1), n=int(r.n))
for r in ht["rows"]:
    if r[0] in facts: facts[r[0]].update(home_edge=r[4])
applies = dict(teaser_dog=dogleg, fav7_low=rec(c1_.over), fav7_high=rec(-c2_.over), close_low=rec(-c3_.over), wind=wu, prime=pu,
               bigdog=dict(ret=f"{bigdog.pl.mean() * 100:+.0f}%", n=len(bigdog), win=pctf((bigdog.win > 0).mean())),
               prime_games=sorted(up[((up.weekday == "Sunday") & (hr >= 19)) | (up.weekday == "Monday")].game_id),
               luck={r[0]: r[4] for r in luck_now["rows"]}, up_spots=up_spots, facts=facts, lg_rz=round(lrz * 100), luck_ats=lh[0][4], unluck_ats=lh[2][4])
json.dump(dict(built=pd.Timestamp.now().strftime("%b %-d, %Y"), sections=groups, applies=applies), open("trends.json", "w"), allow_nan=False)
print("trends.json:", {k: len(v["tables"]) for k, v in groups.items()})
for v in groups.values():
    for t in v["tables"]: print(f"  [{t['tag']}] {t['title']}: {t['verdict']}")
print({k: v for k, v in applies.items() if k != "luck"})
