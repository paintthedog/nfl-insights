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
atd = [r for r in D["td"] if r.get("dk")]
for r in atd:
    gid = gid_of_team.get(r["team"]); ev = r["p"] * r["dk"] - 1
    add(band(r["dk"]), gid, r["name"], "atd", "Yes", None, r["dk"], r["p"])
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
tp = [l for l in pool if l["gid"] in top5]
for c in itertools.combinations(tp, 2): td_parlay("TD parlays: top-5 totals, 2 legs", c)
for c in itertools.combinations(tp, 3): td_parlay("TD parlays: top-5 totals, 3 legs", c)
# Favored team's lead back in each top-5-total game (Week 5: favorites' lead backs scored, the trailing side's often didn't).
fav_rb = []
for g in L["games"]:
    if g["game_id"] not in top5 or g.get("spread") is None: continue
    fav = g["away"] if g["spread"] < 0 else g["home"]          # nflverse sign: negative = away team favored
    rbs = [r for r in atd if r["team"] == fav and r.get("pos") == "RB" and 1.15 <= r["dk"] <= 3.0]
    if rbs:
        r = min(rbs, key=lambda r: r["dk"]); fav_rb.append(dict(gid=g["game_id"], player=r["name"], stat="atd", line=0.5, price=float(r["dk"]), p=round(float(r["p"]), 4)))
for k in (2, 3):
    for c in itertools.combinations(fav_rb, k): td_parlay("TD parlays: top-5 favorites' lead RB", c)
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
out = pd.DataFrame(bets); out.to_csv(f"paper_wk{W}.csv", index=False)
print(f"paper_wk{W}.csv: {len(out)} rows"); print(out.groupby("type").agg(n=("price", "size"), avg_price=("price", "mean"), avg_model=("model_p", "mean")).round(3).to_string() if len(out) else "")
