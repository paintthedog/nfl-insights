"""Board consistency check (Oct 18): DraftKings' lines checked against each other, no model of ours involved.
Usage: python market_consistency.py W   -> consistency_wk<W>.csv + a printed report
A sportsbook's own quants run this daily: every line on the board has to agree with every other line. We do it from the outside with
the Sunday pull. Each check fits a simple relation ACROSS DK's own board for the week, then flags the biggest leftovers.
  1. moneyline vs spread        win chance implied by the moneyline vs the chance the spread implies (league history)
  2. team touchdowns vs points  sum of a team's anytime-TD chances (no-vig) vs the points its spread+total imply
  3. player TD vs player usage  a player's TD chance vs what his own rushing/receptions ladders say about his usage
  4. lead back vs game script   RB1's rushing-yards median vs the team's implied points and spread
A flag is not automatically a bet: the leftover may be injury news the ladders carry and the TD price doesn't (or the reverse).
It is the shortlist of lines where DK's board disagrees with itself; the paper record tests whether the outlier or the rest of
the board was right ("Board consistency" bet type: bet the side that brings the outlier back toward the board)."""
import sys, re, json
from math import erf, sqrt, log
import numpy as np, pandas as pd

W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
M = pd.read_csv("market_log.csv"); M = M[M.week == W].sort_values("pulled_at").drop_duplicates(["game_id", "market", "subject", "side", "point"], keep="last")
TD = pd.read_csv("props_log.csv"); TD = TD[(TD.week == W) & (TD.book == "draftkings")].sort_values("pulled_at").drop_duplicates(["game_id", "player"], keep="last")
sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
site = json.load(open(f"site/data_wk{W}.json")); POS = {norm(p["name"]): (p["pos"], p["team"]) for p in site["proj"]["players"]}
# spread -> home win chance, fit on 2015-2025 results (logistic on the closing spread; sched spread_line is positive when home is favored)
_h = sched[(sched.season >= 2015) & (sched.season < 2026) & sched.result.notna() & sched.spread_line.notna() & (sched.result != 0)]
import statsmodels.api as sm
_m = sm.Logit((_h.result > 0).astype(int), sm.add_constant(_h.spread_line)).fit(disp=0); A_, B_ = float(_m.params.iloc[0]), float(_m.params.iloc[1])
p_home_win = lambda home_fav_by: 1 / (1 + np.exp(-(A_ + B_ * home_fav_by)))
flags = []


def flag(check, gid, subject, market, dk_value, board_value, gap, note, side=None, price=None, point=None):
    """side/price set only when the 'back toward the board' side is something DK actually sells (no 'No' on anytime TD)"""
    flags.append(dict(week=W, check=check, game_id=gid, subject=subject, market=market, dk=round(dk_value, 3), board=round(board_value, 3), gap=round(gap, 3), note=note, side=side, price=price, point=point))


# ---- the game frame from DK's spread and total
G = {}
for gid, x in M.groupby("game_id"):
    sp = x[x.market == "game_spreads"]; tt = x[(x.market == "game_totals") & (x.side == "Over")]; ml = x[x.market == "game_h2h"]
    if gid not in sw.index or not len(sp) or not len(tt): continue
    home, away = sw.loc[gid].home_team, sw.loc[gid].away_team
    hs = sp[sp.subject == home]; T = float(tt.point.iloc[0])
    if not len(hs): continue
    h_hcp = float(hs.point.iloc[0])                       # home handicap: -3 = home favored by 3
    imp = {home: T / 2 - h_hcp / 2, away: T / 2 + h_hcp / 2}
    G[gid] = dict(home=home, away=away, total=T, h_hcp=h_hcp, imp=imp, ml={r.subject: float(r.price) for r in ml.itertuples()})

# ---- 1. moneyline vs spread
for gid, g in G.items():
    if len(g["ml"]) < 2: continue
    hp, ap = g["ml"].get(g["home"]), g["ml"].get(g["away"])
    if not hp or not ap: continue
    p_ml = (1 / hp) / (1 / hp + 1 / ap)                      # no-vig home win chance from the moneyline
    p_sp = p_home_win(-g["h_hcp"])                           # home win chance the spread implies (league history)
    gap = p_ml - p_sp
    if abs(gap) >= 0.04:
        side = g["away"] if gap > 0 else g["home"]           # moneyline is pricier than the spread says for the side with the higher ML chance -> the other side is the value
        flag("Moneyline vs spread", gid, f"{g['away']}@{g['home']}", "ml", p_ml, p_sp, gap, f"moneyline says {g['home']} wins {p_ml:.0%}, the spread ({g['h_hcp']:+g}) says {p_sp:.0%}; {side} moneyline is the cheaper side", side=side, price=g["ml"][side])

# ---- helpers: a DK ladder -> median (the point where the no-vig over chance crosses 50%)
def ladder_median(x):
    x = x.sort_values("point"); p = (1 / x.price.values) / 1.05; pts = x.point.values
    if len(x) < 3 or p[0] < 0.5 or p[-1] > 0.5: return None
    i = np.searchsorted(-p, -0.5)                             # first rung with p <= 0.5
    a, b = pts[i - 1], pts[i]; pa, pb = p[i - 1], p[i]
    return float(a + (pa - 0.5) / (pa - pb) * (b - a))

med = {}
for (gid, subj, mk), x in M[M.market.isin(["player_rush_yds_alternate", "player_receptions_alternate"]) & (M.side == "Over")].groupby(["game_id", "subject", "market"]):
    m = ladder_median(x)
    if m is not None: med[(gid, norm(subj), "rush" if "rush" in mk else "rec")] = m

# ---- 2. team touchdown mass vs implied points
TD["nv"] = 1 / TD.price / 1.07
TD["team"] = [POS.get(norm(n), (None, None))[1] for n in TD.player]
TD["pos"] = [POS.get(norm(n), (None, None))[0] for n in TD.player]
team_mass = TD.dropna(subset=["team"]).groupby(["game_id", "team"]).nv.sum().reset_index()
team_mass["imp"] = [G.get(r.game_id, {}).get("imp", {}).get(r.team) for r in team_mass.itertuples()]
team_mass = team_mass.dropna(subset=["imp"])
if len(team_mass) >= 8:
    b = np.polyfit(team_mass.imp, team_mass.nv, 1); team_mass["fit"] = np.polyval(b, team_mass.imp); team_mass["res"] = team_mass.nv - team_mass.fit
    sd = team_mass.res.std()
    for r in team_mass.itertuples():
        if abs(r.res) >= 1.5 * sd:
            flag("Team TDs vs points", r.game_id, r.team, "team_td_mass", r.nv, r.fit, r.res,
                 f"DK's TD prices on {r.team} add up to {r.nv:.2f} scorers; teams with {r.imp:.1f} implied points average {r.fit:.2f}" + (" (TD prices generous: lean under on its scorers)" if r.res > 0 else " (TD prices stingy: its scorers are priced long)"))

# ---- 3. player TD chance vs usage ladders (within position, across the board)
rows = []
for r in TD.itertuples():
    if r.pos not in ("RB", "WR", "TE"): continue
    ru, rc = med.get((r.game_id, norm(r.player), "rush")), med.get((r.game_id, norm(r.player), "rec"))
    if ru is None and rc is None: continue
    imp = G.get(r.game_id, {}).get("imp", {}).get(r.team)
    if imp is None: continue
    rows.append(dict(game_id=r.game_id, player=r.player, team=r.team, pos=r.pos, nv=r.nv, rush=ru or 0.0, rec=rc or 0.0, imp=imp, price=r.price))
U = pd.DataFrame(rows)
for pos, x in U.groupby("pos"):
    if len(x) < 12: continue
    X = np.column_stack([np.ones(len(x)), x.rush, x.rec, x.imp]); y = np.log(x.nv / (1 - x.nv))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); fit = 1 / (1 + np.exp(-(X @ beta))); res = x.nv.values - fit; sd = res.std()
    for (i, r), f, e in zip(x.iterrows(), fit, res):
        if abs(e) >= 1.8 * sd and r.nv >= 0.12:
            use = (f"{r.rush:.0f} rush yds" if r.rush else "") + (" / " if r.rush and r.rec else "") + (f"{r.rec:.1f} catches" if r.rec else "")
            flag("Player TD vs usage", r.game_id, r.player, "atd", r.nv, f, e, f"DK prices {r.player} ({r.pos}, {r.team}) at {r.nv:.0%} to score; his usage ladders ({use}) and team points fit {f:.0%}" + (" — TD price short vs his usage" if e > 0 else " — TD price long vs his usage"),
                 side=None if e > 0 else "Yes", price=None if e > 0 else r.price)

# ---- 4. lead back's rushing median vs game script
rb = U[U.pos == "RB"].sort_values("rush", ascending=False).drop_duplicates(["game_id", "team"])
rb = rb[rb.rush > 0].copy()
if len(rb) >= 10:
    rb["hcp"] = [-(G[r.game_id]["h_hcp"]) if r.team == G[r.game_id]["home"] else G[r.game_id]["h_hcp"] for r in rb.itertuples()]   # positive = favored by that much
    X = np.column_stack([np.ones(len(rb)), rb.imp, rb.hcp]); beta, *_ = np.linalg.lstsq(X, rb.rush, rcond=None)
    fit = X @ beta; res = rb.rush.values - fit; sd = res.std()
    for (i, r), f, e in zip(rb.iterrows(), fit, res):
        if abs(e) >= 1.5 * sd:
            flag("Lead back vs game script", r.game_id, r.player, "rush_yds", r.rush, f, e, f"DK's ladder puts {r.player} ({r.team}) at a {r.rush:.0f}-yard median; a lead back on a team with {r.imp:.1f} implied points, favored by {r.hcp:+.1f}, averages {f:.0f}" + (" — ladder high vs the game script" if e > 0 else " — ladder low vs the game script"))

F = pd.DataFrame(flags).sort_values(["check", "gap"], key=lambda s: s.abs() if s.name == "gap" else s, ascending=[True, False]) if flags else pd.DataFrame()
F.to_csv(f"consistency_wk{W}.csv", index=False)
# paper record rows: the board's own number stands in for model_p (this test has no model of ours in it)
paper = []
for r in (F[F.side.notna()].itertuples() if len(F) else []):
    if r.market == "ml":
        g = G[r.game_id]; p = (1 - r.board) if r.side == g["away"] else r.board
        paper.append(dict(week=W, type="Board consistency", game_id=r.game_id, subject=r.side, market="ml", side="ML", line=None, price=round(float(r.price), 3), model_p=round(float(p), 4), legs="", placed_at=""))
    elif r.market == "atd":
        paper.append(dict(week=W, type="Board consistency", game_id=r.game_id, subject=r.subject, market="atd", side="Yes", line=None, price=round(float(r.price), 3), model_p=round(float(r.board), 4), legs="", placed_at=""))
PAPER = pd.DataFrame(paper)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 160)
print(f"Week {W}: {len(G)} games framed, {len(med)} ladders, {len(TD)} TD prices; {len(F)} flags")
if len(F): print(F[["check", "subject", "dk", "board", "gap", "note"]].to_string(index=False))
print(f"\n{len(PAPER)} bettable rows for the paper record (Board consistency)")
if __name__ == "__main__" and len(PAPER) and "--paper" in sys.argv:
    f = f"paper_wk{W}.csv"; old = pd.read_csv(f); old = old[old.type != "Board consistency"]
    sw2 = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id"); now = pd.Timestamp.now(tz="America/New_York")
    kick = lambda gid: pd.Timestamp(f"{sw2.loc[gid].gameday} {sw2.loc[gid].gametime}").tz_localize("America/New_York")
    PAPER = PAPER[[kick(g) > now for g in PAPER.game_id]]; PAPER["placed_at"] = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
    pd.concat([old, PAPER], ignore_index=True).to_csv(f, index=False); print(f"{f}: +{len(PAPER)} Board consistency rows")
