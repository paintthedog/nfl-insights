"""Build the LINES tab data: Week N game cards (model vs market) + ATS/OU trend tables. Writes lines_<season>_wk<week>.json."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
import lines_model as L

SEASON, WEEK = 2026, int(sys.argv[1]) if len(sys.argv) > 1 else 4
pbp, sched = L.load()
# games the books haven't posted a spread for yet were filled with this model's number so the stat models can run;
# undo that here so the page never shows our own number as "the line"
import os
FILL = json.load(open("spread_fill.json")) if os.path.exists("spread_fill.json") else {}
sched.loc[sched.game_id.isin(FILL), "spread_line"] = np.nan
try:   # game-day forecasts for this week's outdoor games (forecasts_wk<week>.csv, keyed by home team)
    _fc = pd.read_csv(f"forecasts_wk{WEEK}.csv").set_index("team")
    _m = (sched.season == SEASON) & (sched.week == WEEK) & sched.result.isna() & sched.home_team.isin(_fc.index)
    sched.loc[_m, "wind"] = sched.loc[_m, "home_team"].map(_fc.wind); sched.loc[_m, "temp"] = sched.loc[_m, "home_team"].map(_fc.temp)
except FileNotFoundError:
    pass
played = sched[sched.result.notna()]
upcoming = sched[(sched.season == SEASON) & (sched.week == WEEK) & sched.result.isna()]
g = L.build(pbp, played, predict_games=upcoming)
hist = g[~g._pred & g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)]
mm, mt = L.fit(hist)
bt = pd.read_pickle("lines_backtest.pkl")
sd_m, sd_t = float(np.std(bt.result - bt.m_margin)), float(np.std(bt.total - bt.m_total))
fut = L.predict(g[g._pred].copy(), mm, mt)

# ---- game cards
names = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
         "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
         "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
         "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders"}
import os as _os
_dm = pd.read_parquet("data/team_dmiss.parquet").set_index(["game_id", "team"]) if _os.path.exists("data/team_dmiss.parquet") else None
def _miss(gid, tm):
    try: r = _dm.loc[(gid, tm)]; return round(float(r.sec), 1), round(float(r.f7), 1)
    except Exception: return 0.0, 0.0
def ratings(r, s):
    team = r.home_team if s == "h" else r.away_team
    sec, f7 = _miss(r.game_id, team) if _dm is not None else (0.0, 0.0)
    return dict(off=round(float(r[f"{s}_r_off"]) * 100, 1), deff=round(float(r[f"{s}_r_def"]) * 100, 1), pace=round(float(r[f"{s}_pace"]), 1),
                pass_rate=None if pd.isna(r[f"{s}_pass_rate"]) else round(float(r[f"{s}_pass_rate"]) * 100), qb_change=int(r[f"{s}_qb_change"]),
                d_sec=sec, d_f7=f7)
games = []
for _, r in fut.sort_values("gameday").iterrows():
    sp, tl = (None if pd.isna(r.spread_line) else float(r.spread_line)), (None if pd.isna(r.total_line) else float(r.total_line))
    games.append(dict(
        game_id=r.game_id, day=pd.to_datetime(r.gameday).strftime("%a %b %-d"), home=r.home_team, away=r.away_team,
        home_name=names.get(r.home_team, r.home_team), away_name=names.get(r.away_team, r.away_team),
        spread=sp, total=tl, m_margin=round(float(r.m_margin), 1), m_total=round(float(r.m_total), 1),
        p_home_cover=None if sp is None else round(float(1 - norm.cdf((sp - r.m_margin) / sd_m)), 3),
        p_over=None if tl is None else round(float(1 - norm.cdf((tl - r.m_total) / sd_t)), 3),
        wind=int(r.wind_eff), dome=int(r.dome), home_rest=int(r.home_rest), away_rest=int(r.away_rest), div=int(r.div_game),
        h=ratings(r, "h"), a=ratings(r, "a")))

# ---- trend tables (regular season, closing line)
h = sched[sched.result.notna() & sched.spread_line.notna()].copy()
h["home_cover"] = np.sign(h.result - h.spread_line); h["over"] = np.sign(h.total - h.total_line)
long = pd.concat([h.assign(team=h.home_team, cover=h.home_cover, fav=(h.spread_line > 0), margin_vs=h.result - h.spread_line),
                  h.assign(team=h.away_team, cover=-h.home_cover, fav=(h.spread_line < 0), margin_vs=-(h.result - h.spread_line))])
def rec(df, col):
    w, l, p = int((df[col] > 0).sum()), int((df[col] < 0).sum()), int((df[col] == 0).sum())
    return dict(w=w, l=l, p=p, pct=round(w / (w + l) * 100, 1) if w + l else None)
team_rows = []
for tm in sorted(set(long[long.season == SEASON].team.unique())):
    row = dict(team=tm, name=names.get(tm, tm))
    for lab, yr in [("cur", SEASON), ("prev", SEASON - 1)]:
        d = long[(long.team == tm) & (long.season == yr)]
        row[lab] = dict(ats=rec(d, "cover"), ou=rec(d, "over"), avg_cover=round(float(d.margin_vs.mean()), 1) if len(d) else None,
                        fav_ats=rec(d[d.fav], "cover"), dog_ats=rec(d[~d.fav], "cover"))
    team_rows.append(row)

# situational trends: this season, last season, and 2016-2025 (the hold-up test)
h["wind_eff"] = np.where(~h.roof.isin(["dome", "closed"]), h.wind.fillna(8), 0)
long = long.merge(h[["game_id", "wind_eff"]], on="game_id")
long["is_home"] = long.team == long.home_team
long["rest"] = np.where(long.is_home, long.home_rest, long.away_rest); long["opp_rest"] = np.where(long.is_home, long.away_rest, long.home_rest)
long["short"] = long.rest <= 5; long["bye"] = long.rest >= 13
long["dog"] = ~long.fav; long["big_dog"] = long.spread_line.abs().where(long.dog, 0) >= 7
long["primetime"] = long.gametime.fillna("13:00").str[:2].astype(int) >= 19
situations = [
    ("Home teams", long.is_home), ("Home underdogs", long.is_home & long.dog), ("Road favorites", ~long.is_home & long.fav),
    ("Underdogs of 7+", long.big_dog), ("Favorites of 7+", long.fav & (long.spread_line.abs() >= 7)),
    ("Team off a bye", long.bye), ("Opponent off a bye", long.opp_rest >= 13), ("Short week (Thu)", long.short),
    ("Division games", long.div_game == 1), ("Primetime", long.primetime), ("Dome games", long.roof.isin(["dome", "closed"])),
    ("Wind 15+ mph", long.wind_eff >= 15),
]
def ou_rec(df):   # per game, not per team-row: dedupe
    d = df.drop_duplicates("game_id"); return rec(d, "over")
sit_rows = []
for label, mask in situations:
    row = dict(label=label)
    for lab, yrs in [("cur", [SEASON]), ("prev", [SEASON - 1]), ("ten", list(range(2016, SEASON)))]:
        d = long[mask & long.season.isin(yrs)]
        row[lab] = dict(ats=rec(d, "cover"), ou=ou_rec(d))
    sit_rows.append(row)

# league-wide over/under by season
lg = [dict(season=int(s), **ou_rec(h[h.season == s]), avg_total=round(float(h[h.season == s].total.mean()), 1), avg_line=round(float(h[h.season == s].total_line.mean()), 1))
      for s in sorted(h.season.unique())]

# backtest summary for the page
hc, oh = np.sign(bt.result - bt.spread_line), np.sign(bt.total - bt.total_line)
def wr(pred, line, truth, x):
    pk = np.sign(pred - line); m = (np.abs(pred - line) >= x) & (truth != 0)
    return dict(edge=x, n=int(m.sum()), win=round(float((pk[m] == truth[m]).mean() * 100), 1))
summary = dict(n=int(len(bt)), seasons="2019–2026", line_mae_margin=round(float(np.mean(np.abs(bt.result - bt.spread_line))), 2),
               model_mae_margin=round(float(np.mean(np.abs(bt.result - bt.m_margin))), 2),
               line_mae_total=round(float(np.mean(np.abs(bt.total - bt.total_line))), 2),
               model_mae_total=round(float(np.mean(np.abs(bt.total - bt.m_total))), 2),
               ats=[wr(bt.m_margin, bt.spread_line, hc, x) for x in [0, 2, 3, 5]],
               ou=[wr(bt.m_total, bt.total_line, oh, x) for x in [0, 2, 3, 5]], breakeven=52.4)
# ---- by size of the spread and of the total (one row per game, from the favorite's side)
gm = h[(h.game_type == "REG") & (h.season >= 2016)].copy()
gm["sp"] = gm.spread_line.abs(); gm["fav_m"] = np.where(gm.spread_line >= 0, gm.result, -gm.result)
gm["fav_cover"] = np.sign(gm.fav_m - gm.sp); gm["fav_win"] = np.sign(gm.fav_m); gm["road_fav"] = gm.spread_line < 0
def _rows(defs):
    out = []
    for label, mask in defs:
        row = dict(label=label)
        for lab, yrs in [("cur", [SEASON]), ("prev", [SEASON - 1]), ("ten", list(range(2016, SEASON)))]:
            d = gm[mask & gm.season.isin(yrs)]; dd = d[d.sp > 0]
            row[lab] = dict(n=int(len(d)), cover=rec(dd, "fav_cover"), win=rec(dd, "fav_win"), ou=rec(d, "over"),
                            pts=round(float(d.total.mean()), 1) if len(d) else None, line=round(float(d.total_line.mean()), 1) if len(d) else None)
        out.append(row)
    return out
by_spread = _rows([("Pick'em to 2.5", gm.sp <= 2.5), ("3", gm.sp == 3), ("3.5 to 6.5", gm.sp.between(3.5, 6.5)), ("7", gm.sp == 7), ("7.5 to 9.5", gm.sp.between(7.5, 9.5)),
                   ("10 to 13.5", gm.sp.between(10, 13.5)), ("14 or more", gm.sp >= 14), ("Road favorites of 7+", gm.road_fav & (gm.sp >= 7)), ("Home favorites of 7+", ~gm.road_fav & (gm.sp >= 7))])
by_total = _rows([("Under 38", gm.total_line < 38), ("38 to 40.5", gm.total_line.between(38, 40.5)), ("41 to 43.5", gm.total_line.between(41, 43.5)), ("44 to 46.5", gm.total_line.between(44, 46.5)),
                  ("47 to 49.5", gm.total_line.between(47, 49.5)), ("50 to 52.5", gm.total_line.between(50, 52.5)), ("53 or more", gm.total_line >= 53)])
combos = _rows([("Favorite of 7+ and total under 42", (gm.sp >= 7) & (gm.total_line < 42)), ("Favorite of 7+ and total 48+", (gm.sp >= 7) & (gm.total_line >= 48)),
                ("Spread under 3 and total under 42", (gm.sp < 3) & (gm.total_line < 42)), ("Spread under 3 and total 48+", (gm.sp < 3) & (gm.total_line >= 48))])
# how steady the 10-year combo results are: seasons (of 10) in which overs finished above 50%
for r, (label, mask) in zip(combos, [(0, (gm.sp >= 7) & (gm.total_line < 42)), (0, (gm.sp >= 7) & (gm.total_line >= 48)), (0, (gm.sp < 3) & (gm.total_line < 42)), (0, (gm.sp < 3) & (gm.total_line >= 48))]):
    yr = gm[mask & (gm.season < SEASON)].groupby("season").over.agg(lambda x: (x > 0).sum() / max(((x > 0) | (x < 0)).sum(), 1))
    r["over_seasons"] = int((yr > 0.5).sum()); r["n_seasons"] = int(len(yr))
json.dump(dict(by_spread=by_spread, by_total=by_total, combos=combos, games=games, teams=team_rows, situations=sit_rows, league_ou=lg, summary=summary, sd_margin=round(sd_m, 1), sd_total=round(sd_t, 1)),
          open(f"lines_{SEASON}_wk{WEEK}.json", "w"))
print(f"{len(games)} games; model vs market:")
for x in games: print(f"  {x['away']}@{x['home']}: line {x['spread']} / model {x['m_margin']:+.1f} | total {x['total']} / model {x['m_total']}")

# ---- no market spread yet: lend the stat models this model's margin (rounded to the half point) and record that we did
_nf = fut[fut.spread_line.isna()]
if len(_nf) or FILL:
    _sc = pd.read_parquet("data/sched.parquet")
    for r in _nf.itertuples(): FILL[r.game_id] = round(float(r.m_margin) * 2) / 2
    for gid, v in FILL.items(): _sc.loc[_sc.game_id == gid, "spread_line"] = v
    _sc.to_parquet("data/sched.parquet"); json.dump(FILL, open("spread_fill.json", "w"))
    print("no market spread yet, model number lent to the stat models:", {r.game_id: FILL[r.game_id] for r in _nf.itertuples()})
