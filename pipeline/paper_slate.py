"""Mechanical paper slate for a week, built from the game-morning pull. Usage: python paper_slate.py 6
Every rule is fixed in advance; nothing is hand-picked. Output: paper_wk<W>.csv (one row per paper bet, parlays as one row with legs)."""
import sys, json, re, itertools, numpy as np, pandas as pd
from math import erf, sqrt
W = int(sys.argv[1]); rng = np.random.default_rng(1000 + W)
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
D = json.load(open(f"site/data_wk{W}.json")); L = json.load(open(f"lines_2026_wk{W}.json"))
QS = D["proj"]["quants"]
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
def p_over(arr, line):
    arr = np.asarray(arr, float)
    if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
    for i in range(1, len(arr)):
        if line <= arr[i]:
            t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
    return max(0, (1 - QS[-1]) * .5)
nv = lambda a, b: (1 / a) / (1 / a + 1 / b)
games = {g["game_id"]: g for g in L["games"]}
gid_of_team = {}
for g in L["games"]: gid_of_team[g["home"]] = g["game_id"]; gid_of_team[g["away"]] = g["game_id"]
SDM, SDT = L["sd_margin"], L["sd_total"]; SDTEAM = sqrt((SDT ** 2 + SDM ** 2) / 4)
bets, now = [], pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
def add(type_, gid, subj, market, side, line, price, p, legs=None):
    bets.append(dict(week=W, type=type_, game_id=gid, subject=subj, market=market, side=side, line=line, price=round(float(price), 3), model_p=round(float(p), 4),
                     legs=json.dumps(legs) if legs else "", placed_at=now))
# ---- anytime TDs (tracking by band + the two betting rules)
band = lambda x: "TD all: under 2.5" if x < 2.5 else "TD all: 2.5–3.9" if x <= 3.9 else "TD all: 4.0–6.0" if x <= 6.0 else "TD all: over 6.0"
# Oct 21: the 4.0–6.0 band is no longer logged (8 hits in 62, -26.7u through Week 6; Clinton doesn't bet past 4.5 anyway)
atd = [r for r in D["td"] if r.get("dk")]
for r in atd:
    gid = gid_of_team.get(r["team"]); ev = r["p"] * r["dk"] - 1
    if not 3.9 < r["dk"] <= 6.0: add(band(r["dk"]), gid, r["name"], "atd", "Yes", None, r["dk"], r["p"])
    if 2.5 <= r["dk"] <= 3.9 and ev >= 0.03: add("TD 2.5–3.9, model likes", gid, r["name"], "atd", "Yes", None, r["dk"], r["p"])
    if r["dk"] < 2.5 and ev <= -0.10: add("TD skip list (not bet)", gid, r["name"], "atd", "Yes", None, r["dk"], r["p"])
# ---- TD parlays from DIFFERENT games (no correlation between legs, so the bet is "are DK's short scorer prices fair?").
# Leg pool: the shortest-priced anytime TD in each game, priced 1.15-3.00 (floor lowered from 1.25 after Week 5: it excluded Gibbs at 1.24) (the favorite end of the favorite-longshot bias).
# Fixed construction, nothing hand-picked: games ordered by that leg's price, shortest first.
pool = {}
for r in atd:
    gid = gid_of_team.get(r["team"])
    if gid and 1.15 <= r["dk"] <= 3.0 and (gid not in pool or r["dk"] < pool[gid]["price"]):
        pool[gid] = dict(gid=gid, player=r["name"], stat="atd", line=0.5, price=float(r["dk"]), p=round(float(r["p"]), 4))
pool = sorted(pool.values(), key=lambda l: l["price"])
def td_parlay(type_, combo):
    add(type_, "", " + ".join(f"{l['player']} TD" for l in combo), "parlay", "All", None, float(np.prod([l["price"] for l in combo])),
        float(np.prod([l["p"] for l in combo])), legs=list(combo))
for l in pool: add("TD singles: likeliest scorer per game", l["gid"], l["player"], "atd", "Yes", None, l["price"], l["p"])
for i in range(0, len(pool) - 1, 2):
    if i // 2 < 4: td_parlay("TD parlays: likeliest per game, 2 legs", pool[i:i + 2])
for i in range(0, len(pool) - 2, 3):
    if i // 3 < 3: td_parlay("TD parlays: likeliest per game, 3 legs", pool[i:i + 3])
top5 = {g["game_id"] for g in sorted([g for g in L["games"] if g.get("total")], key=lambda g: -g["total"])[:5]}
# Oct 21: cut "TD parlays: top-5 totals" (2 and 3 legs; 1 win in 20, -18u) and "TD parlays: top-5 favorites' lead RB" (4 in 20, -10u).
# Their past rows stay in the record.
# Both teams' shortest scorer in the week's single highest-total game (same game: recorded at the singles' product, not DK's SGP price)
if L["games"]:
    hi = max([g for g in L["games"] if g.get("total")], key=lambda g: g["total"], default=None)
    if hi:
        both = []
        for tm in (hi["away"], hi["home"]):
            c = [r for r in atd if r["team"] == tm and 1.15 <= r["dk"] <= 3.0]
            if c: r = min(c, key=lambda r: r["dk"]); both.append(dict(gid=hi["game_id"], player=r["name"], stat="atd", line=0.5, price=float(r["dk"]), p=round(float(r["p"]), 4)))
        if len(both) == 2: td_parlay("TD parlays: top-total game, both teams' shortest", both)
# ---- receptions (main line)
recs = []
for p in D["proj"]["players"]:
    dk = ((p.get("lines") or {}).get("rec") or {}).get("draftkings"); q = (p.get("q") or {}).get("rec")
    if not dk or not q: continue
    po, mo = p_over(q, dk["pt"]), nv(dk["o"], dk["u"]); gid = gid_of_team.get(p["team"])
    recs.append((p, dk, po, mo, gid))
    if abs(po - mo) >= 0.05:
        side = "Over" if po > mo else "Under"
        add("Receptions, model differs 5+ pts", gid, p["name"], "rec", side, dk["pt"], dk["o"] if side == "Over" else dk["u"], po if side == "Over" else 1 - po)
# ---- extra markets from market_log
try:
    M = pd.read_csv("market_log.csv"); M = M[M.week == W].sort_values("pulled_at").drop_duplicates(["game_id", "market", "subject", "side", "point"], keep="last")
except FileNotFoundError:
    M = pd.DataFrame(columns=["game_id", "market", "subject", "side", "point", "price"])
# QB interceptions 0.5
try:
    ip = pd.read_csv(f"int_props_2026_wk{W}.csv")
    pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"]); nm = {norm(n): g for g, n in zip(pl.gsis_id, pl.display_name)}
    pint = dict(zip(ip.passer_player_id, ip.p_int))
    m = M[M.market == "player_pass_interceptions"]
    for (gid, subj), x in m.groupby(["game_id", "subject"]):
        o = x[x.side == "Over"]; u = x[x.side == "Under"]
        if not len(o) or not len(u) or float(o.point.iloc[0]) != 0.5: continue
        pi = pint.get(nm.get(norm(subj)))
        if pi is None: continue
        mo = nv(o.price.iloc[0], u.price.iloc[0])
        if abs(pi - mo) >= 0.05:
            side = "Over" if pi > mo else "Under"
            add("QB INTs (0.5), model differs 5+ pts", gid, subj, "ints", side, 0.5, o.price.iloc[0] if side == "Over" else u.price.iloc[0], pi if side == "Over" else 1 - pi)
except FileNotFoundError:
    pass
# team totals
for (gid, team), x in M[M.market == "team_totals"].groupby(["game_id", "subject"]):
    g = games.get(gid); o = x[x.side == "Over"]; u = x[x.side == "Under"]
    if not g or not len(o) or not len(u): continue
    mp = (g["m_total"] + g["m_margin"]) / 2 if team == g["home"] else (g["m_total"] - g["m_margin"]) / 2
    ln = float(o.point.iloc[0]); po = 1 - Phi((ln - mp) / SDTEAM)
    if abs(mp - ln) >= 2:
        side = "Over" if mp > ln else "Under"
        add("Team totals, model differs 2+ pts", gid, team, "team_total", side, ln, o.price.iloc[0] if side == "Over" else u.price.iloc[0], po if side == "Over" else 1 - po)
# game totals + moneylines
gt_all, ml_all = [], []
for gid, g in games.items():
    x = M[(M.game_id == gid) & (M.market == "game_totals")]
    o, u = x[x.side == "Over"], x[x.side == "Under"]
    if len(o) and len(u):
        ln = float(o.point.iloc[0]); po = 1 - Phi((ln - g["m_total"]) / SDT); gt_all.append((gid, ln, o.price.iloc[0], u.price.iloc[0], po))
        if abs(g["m_total"] - ln) >= 3:
            side = "Over" if g["m_total"] > ln else "Under"
            add("Game totals, model differs 3+ pts", gid, f"{g['away']}@{g['home']}", "game_total", side, ln, o.price.iloc[0] if side == "Over" else u.price.iloc[0], po if side == "Over" else 1 - po)
    h = M[(M.game_id == gid) & (M.market == "game_h2h")]
    hp, ap = h[h.subject == g["home"]], h[h.subject == g["away"]]
    if len(hp) and len(ap):
        ph = Phi(g["m_margin"] / SDM); mh = nv(hp.price.iloc[0], ap.price.iloc[0]); ml_all.append((gid, g, hp.price.iloc[0], ap.price.iloc[0], ph))
        if abs(ph - mh) >= 0.04:
            team, price, p = (g["home"], hp.price.iloc[0], ph) if ph > mh else (g["away"], ap.price.iloc[0], 1 - ph)
            add("Moneyline, model differs 4+ pts", gid, team, "ml", "ML", None, price, p)
# ---- Oct 21: trend unders (no model involved; the edge, if any, is a long-run pattern in the results). model_p = the pattern's
# historical hit rate from trends.json, so the record shows whether it still holds.
_sched = pd.read_parquet("data/sched.parquet"); _sw = _sched[(_sched.season == 2026) & (_sched.week == W)].set_index("game_id")
try: _TA = json.load(open("trends.json"))["applies"]
except Exception: _TA = {}
for gid, g in games.items():
    x = M[(M.game_id == gid) & (M.market == "game_totals") & (M.side == "Under")]
    if not len(x) or gid not in _sw.index: continue
    ln, up = float(x.point.iloc[0]), float(x.price.iloc[0]); sub = f"{g['away']}@{g['home']}"
    wx = (D.get("weather") or {}).get(g["home"]) or {}
    if not wx.get("dome") and not wx.get("unverified") and wx.get("wind") is not None and 10 <= wx["wind"] <= 19 and _TA.get("wind"):
        add("Trend unders: wind 10–19 mph", gid, sub, "game_total", "Under", ln, up, _TA["wind"]["pct"] / 100)
    if int(str(_sw.loc[gid].gametime)[:2]) >= 19 and _TA.get("prime"):
        add("Trend unders: prime time", gid, sub, "game_total", "Under", ln, up, _TA["prime"]["pct"] / 100)
# ---- Oct 21: 6-point teasers on small underdogs (+1.5 to +2.5 -> +7.5 to +8.5, crossing 3 and 7). DK prices every 6-point teaser
# the same, but NFL margins cluster on 3 and 7, so these legs cover more often than the price assumes. The favorite half of the
# classic teaser (-7.5 to -8.5) tested worse (71% since 2016) and is left out. Legs from different games, paired in kickoff order.
# The Odds API has no teaser prices: price is a REFERENCE of 1.80 for a 2-team teaser (check DK's actual price in the app).
TEASE_PRICE = 1.80
_hist = _sched[(_sched.season >= 2016) & (_sched.season < 2026) & (_sched.game_type == "REG") & _sched.result.notna() & _sched.spread_line.notna()]
_lw = []
for r in _hist.itertuples():
    for home in (True, False):
        line = -r.spread_line if home else r.spread_line; mg = r.result if home else -r.result
        if 1.5 <= line <= 2.5 and mg + line + 6 != 0: _lw.append(mg + line + 6 > 0)
P_TLEG = float(np.mean(_lw)) if _lw else 0.75
tlegs = []
for gid, g in games.items():
    x = M[(M.game_id == gid) & (M.market == "game_spreads")]
    for _, r in x.iterrows():
        if 1.5 <= float(r.point) <= 2.5 and gid in _sw.index:
            tlegs.append(dict(gid=gid, player=r.subject, stat="spread", line=float(r.point) + 6, price=round(TEASE_PRICE ** 0.5, 4), p=round(P_TLEG, 4),
                              label=f"{r.subject} {float(r.point) + 6:+g}", kick=f"{_sw.loc[gid].gameday} {_sw.loc[gid].gametime}"))
tlegs.sort(key=lambda l: (l["kick"], l["gid"]))
for i in range(0, len(tlegs) - 1, 2):
    c = tlegs[i:i + 2]
    if c[0]["gid"] == c[1]["gid"]: continue
    add("Teasers: small underdogs +6, 2 legs", "", " + ".join(l["label"] for l in c), "parlay", "All", None, TEASE_PRICE, P_TLEG ** 2,
        legs=[{k: v for k, v in l.items() if k != "kick"} for l in c])
# ---- 2-3 leg parlays from alternate rungs (receptions, rushing yards), legs from different games
P = {norm(p["name"]): p for p in D["proj"]["players"]}
legs = []
for _, r in M[M.market.isin(["player_receptions_alternate", "player_rush_yds_alternate"]) & (M.side == "Over")].iterrows():
    p = P.get(norm(r.subject)); st = "rec" if "receptions" in r.market else "rush_yds"
    q = (p or {}).get("q", {}).get(st) if p else None
    if not q or r.price > 1.75 or r.price < 1.15: continue
    pm = p_over(q, float(r.point))
    if pm >= 1 / r.price + 0.03: legs.append(dict(gid=r.game_id, player=r.subject, stat=st, line=float(r.point), price=float(r.price), p=round(pm, 4)))
cands = []
for k in (2, 3):
    for combo in itertools.combinations(sorted(legs, key=lambda l: -(l["p"] * l["price"]))[:40], k):
        if len({l["gid"] for l in combo}) < k: continue
        pay = float(np.prod([l["price"] for l in combo]))
        if 2.0 <= pay <= 2.6: cands.append((float(np.prod([l["p"] for l in combo])) * pay - 1, pay, combo))
used = {}
for ev, pay, combo in sorted(cands, key=lambda c: -c[0]):
    if sum(1 for b in bets if b["type"] == "Parlays (alt lines, 2–3 legs)") >= 8: break
    if any(used.get((l["player"], l["stat"], l["line"]), 0) >= 2 for l in combo): continue
    for l in combo: used[(l["player"], l["stat"], l["line"])] = used.get((l["player"], l["stat"], l["line"]), 0) + 1
    add("Parlays (alt lines, 2–3 legs)", "", " + ".join(f"{l['player']} {l['line']}+ {'rec' if l['stat']=='rec' else 'rush yds'}" for l in combo),
        "parlay", "All", None, pay, float(np.prod([l["p"] for l in combo])), legs=list(combo))
# ---- Oct 21: steady-leg parlays. Alternate rungs DK prices 1.25-1.50 (about 67-80% legs) on players with a real season record
# (8+ catches for pass catchers, 25+ carries for backs), where the model at least agrees with DK (model chance >= DK's implied).
# No model edge required: the question is whether stacking DK's own short prices across games gets to ~2.0 without bleeding.
_pb = pd.read_parquet("data/pbp_2026.parquet", columns=["week", "receiver_player_id", "rusher_player_id", "complete_pass", "rush_attempt"])
_pb = _pb[_pb.week < W]
_CAT = _pb.groupby("receiver_player_id").complete_pass.sum(); _CAR = _pb.groupby("rusher_player_id").rush_attempt.sum()
def _record_ok(p, st):
    return (_CAT.get(p.get("pid"), 0) >= 8) if st == "rec" else (_CAR.get(p.get("pid"), 0) >= 25)
steady = []
for _, r in M[M.market.isin(["player_receptions_alternate", "player_rush_yds_alternate"]) & (M.side == "Over")].iterrows():
    p = P.get(norm(r.subject)); st = "rec" if "receptions" in r.market else "rush_yds"
    q = (p or {}).get("q", {}).get(st) if p else None
    if not q or not 1.25 <= r.price <= 1.50 or (p.get("p_play") or 1) < 0.9 or not _record_ok(p, st): continue
    pm = p_over(q, float(r.point))
    sd = p.get("steady") or {}
    if pm >= 1 / r.price: steady.append(dict(gid=r.game_id, player=r.subject, stat=st, line=float(r.point), price=float(r.price), p=round(pm, 4),
                                             stdy=sd.get("score"), stdy_flag=bool(sd.get("flags"))))   # Oct 22: steadiness tagged on every leg
best = {}
for l in steady:      # one rung per player and stat: the one closest to 1.40
    k = (l["player"], l["stat"])
    if k not in best or abs(l["price"] - 1.40) < abs(best[k]["price"] - 1.40): best[k] = l
def steady_parlays(legs, type_):
    legs = sorted(legs, key=lambda l: -l["p"])[:24]
    sc = []
    for k in (2, 3):
        for combo in itertools.combinations(legs, k):
            if len({l["gid"] for l in combo}) < k: continue
            pay = float(np.prod([l["price"] for l in combo]))
            if 1.9 <= pay <= 2.4: sc.append((float(np.prod([l["p"] for l in combo])), pay, combo))
    used_s = {}
    for jp, pay, combo in sorted(sc, key=lambda c: -c[0]):
        if sum(1 for b in bets if b["type"] == type_) >= 6: break
        if any(used_s.get((l["player"], l["stat"]), 0) >= 1 for l in combo): continue
        for l in combo: used_s[(l["player"], l["stat"])] = 1
        add(type_, "", " + ".join(f"{l['player']} {l['line']:g}+ {'rec' if l['stat']=='rec' else 'rush yds'}" for l in combo),
            "parlay", "All", None, pay, jp, legs=list(combo))
steady_parlays(list(best.values()), "Parlays: steady legs 1.25–1.50")
# Oct 22 (exploratory, Clinton's steadiness rating): the same rule using only players rated Steady 60+ with no role change this week.
# Compared head to head with the parent rule above; the per-leg stdy tags also let us grade legs by score band.
steady_parlays([l for l in best.values() if (l["stdy"] or 0) >= 60 and not l["stdy_flag"]], "Parlays: steady legs, rated 60+")
# ---- Clinton's rule (Oct 18): prop bets go to players with the BEST MATCHUPS. A matchup is "best" when the opponent ranks top-8 in fantasy
# points allowed to the player's position this season (the site's opp_rank) AND has allowed at least a league-average rate of TDs to that
# position (dvp 'td' vs league). Two bet types: the TD play and the receptions over, both limited to those players. Mechanical, graded like the rest.
_dv = {(r["def"], r["pos"]): r for r in D["dvp"]["cur"]}
_lgtd = {}
for r in D["dvp"]["cur"]: _lgtd.setdefault(r["pos"], []).append(r["td"])
_lgtd = {k: sum(v) / len(v) for k, v in _lgtd.items()}
_PL = {norm(p["name"]): p for p in D["proj"]["players"]}
def best_matchup(name):
    p = _PL.get(norm(name)); 
    if not p or p.get("opp_rank") is None or p["pos"] not in ("RB", "WR", "TE"): return False
    dv = _dv.get((p["opp"], p["pos"]))
    return p["opp_rank"] <= 8 and dv is not None and dv["td"] >= _lgtd.get(p["pos"], 0)
for r in atd:
    if best_matchup(r["name"]) and 1.5 <= r["dk"] <= 4.0 and (_PL[norm(r["name"])].get("p_play") or 1) >= 0.85:
        add("Matchup props: TD, opp top-8 vs position", gid_of_team.get(r["team"]), r["name"], "atd", "Yes", None, r["dk"], r["p"])
for p, dk, po, mo, gid in recs:
    if best_matchup(p["name"]) and po >= mo and (p.get("p_play") or 1) >= 0.85:
        add("Matchup props: receptions over, opp top-8 vs position", gid, p["name"], "rec", "Over", dk["pt"], dk["o"], po)
# ---- control group: random sides from the same markets (what the book's cut alone costs)
band_atd = [r for r in atd if 2.5 <= r["dk"] <= 3.9]
for r in (rng.choice(band_atd, size=min(15, len(band_atd)), replace=False) if band_atd else []):
    add("Control: random", gid_of_team.get(r["team"]), r["name"], "atd", "Yes", None, r["dk"], r["p"])
for i in (rng.choice(len(recs), size=min(10, len(recs)), replace=False) if recs else []):
    p, dk, po, mo, gid = recs[i]; side = "Over" if rng.random() < .5 else "Under"
    add("Control: random", gid, p["name"], "rec", side, dk["pt"], dk["o"] if side == "Over" else dk["u"], po if side == "Over" else 1 - po)
for i in (rng.choice(len(gt_all), size=min(5, len(gt_all)), replace=False) if gt_all else []):
    gid, ln, op, up, po = gt_all[i]; side = "Over" if rng.random() < .5 else "Under"; g = games[gid]
    add("Control: random", gid, f"{g['away']}@{g['home']}", "game_total", side, ln, op if side == "Over" else up, po if side == "Over" else 1 - po)
out = pd.DataFrame(bets)
# same-game stacks (Oct 18): fixed rules in sgp_slate.py, graded on hit rate vs the joint chance (no DK same-game prices in the feed)
try:
    import sgp_slate, ml_td_slate; out = pd.concat([out, sgp_slate.build(W), ml_td_slate.build(W)], ignore_index=True)
except Exception as e: print("same-game stacks skipped:", e)
out.to_csv(f"paper_wk{W}.csv", index=False)
# board consistency (Oct 18): DK's lines against each other, no model of ours; bettable flags join the record
import subprocess
r_ = subprocess.run([sys.executable, "market_consistency.py", str(W), "--paper"], capture_output=True, text=True)
print(r_.stdout.strip().splitlines()[-1] if r_.returncode == 0 and r_.stdout.strip() else f"board consistency skipped: {r_.stderr.strip()[-300:]}")
print(f"paper_wk{W}.csv: {len(out)} rows"); print(out.groupby("type").agg(n=("price", "size"), avg_price=("price", "mean"), avg_model=("model_p", "mean")).round(3).to_string() if len(out) else "")
