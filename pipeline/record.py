"""Season record: graded paper bets + weekly model-vs-market accuracy -> record.json (feeds the Record tab)."""
import glob, json, re, numpy as np, pandas as pd, lines_model as L
SEASON = 2026
pbp = pd.read_parquet(f"data/pbp_{SEASON}.parquet")
pbp = pbp[pbp.two_point_attempt.ne(1)]
done = set(pbp.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"])
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
NM = {}
for g, n in zip(pl.gsis_id, pl.display_name): NM.setdefault(norm(n), []).append(g)
touched = set(pbp.passer_player_id.dropna()) | set(pbp.rusher_player_id.dropna()) | set(pbp.receiver_player_id.dropna()) | set(pbp.td_player_id.dropna())
def pid(n, gid):
    c = NM.get(norm(n), [])
    if len(c) == 1: return c[0]
    gm = pbp[pbp.game_id == gid]; ids = set(gm.passer_player_id.dropna()) | set(gm.rusher_player_id.dropna()) | set(gm.receiver_player_id.dropna())
    h = [x for x in c if x in ids]; return h[0] if h else (c[0] if c else None)
tds = pbp[pbp.touchdown.eq(1) & pbp.td_player_id.notna()].groupby(["game_id", "td_player_id"]).size()
ST = {"player_pass_yds": ("passer_player_id", "passing_yards"), "player_rush_yds": ("rusher_player_id", "rushing_yards"),
      "player_reception_yds": ("receiver_player_id", "receiving_yards"), "player_receptions": ("receiver_player_id", "complete_pass")}
STV = {m: pbp.groupby(["game_id", c])[v].sum() for m, (c, v) in ST.items()}

# ---------------- 1. paper bets
def btype(r):
    c = r["cat"]
    if c == "TD": return "Anytime TD, 4.5 or shorter" if r["price"] <= 4.5 else "Anytime TD, longer than 4.5"
    if c == "OU": return "Player over" if " Over " in r["bet"] else "Player under"
    return {"spread": "Spreads", "total": "Game totals", "prime under": "Game totals"}.get(c, "Teasers & other")
bets = []
for f in sorted(glob.glob("bets_wk*_graded.csv")):
    w = int(re.search(r"wk(\d+)", f).group(1)); slate = "v2" if "_v2" in f else "v1"
    d = pd.read_csv(f); d["week"] = w; d["slate"] = slate; d["type"] = d.apply(btype, axis=1); bets.append(d)
bets = pd.concat(bets) if bets else pd.DataFrame()
g = bets[bets.result.isin(["win", "loss"])]
types = []
for t, x in g.groupby("type"):
    n = len(x); u = x.pnl.sum(); exp = x.model_p.sum(); w = int((x.result == "win").sum())
    types.append(dict(type=t, n=int(n), w=w, units=round(float(u), 2), roi=round(float(u / n), 3), exp_w=round(float(exp), 1)))

# ---------------- 1b. current strategy: mechanical paper slates (paper_wk*_graded.csv)
ORDER = ["Matchup props: TD, opp top-8 vs position", "Matchup props: receptions over, opp top-8 vs position", "TD 2.5–3.9, model likes", "Receptions, model differs 5+ pts", "QB INTs (0.5), model differs 5+ pts", "Team totals, model differs 2+ pts",
         "Game totals, model differs 3+ pts", "Moneyline, model differs 4+ pts", "Board consistency", "Parlays (alt lines, 2–3 legs)", "Parlays: steady legs 1.25–1.50",
         "Teasers: small underdogs +6, 2 legs", "Trend unders: wind 10–19 mph", "Trend unders: prime time",
         "TD parlays: likeliest per game, 2 legs", "TD parlays: likeliest per game, 3 legs", "TD parlays: top-5 totals, 2 legs", "TD parlays: top-5 totals, 3 legs", "TD parlays: top-5 favorites' lead RB", "TD parlays: top-total game, both teams' shortest",
         "TD skip list (not bet)", "TD singles: likeliest scorer per game",
         "ML + top TD scorer, 2 legs", "Same-game stacks: favorite + lead RB", "Same-game stacks: top two pass catchers", "Same-game stacks: QB yards + WR1 catches",
         "TD all: under 2.5", "TD all: 2.5–3.9", "TD all: 4.0–6.0", "TD all: over 6.0", "Control: random"]
_SKIP = {"TD skip list (not bet)"}; _TRACK = {"ML + top TD scorer, 2 legs", "Same-game stacks: favorite + lead RB", "Same-game stacks: top two pass catchers", "Same-game stacks: QB yards + WR1 catches", "TD singles: likeliest scorer per game", "TD all: under 2.5", "TD all: 2.5–3.9", "TD all: 4.0–6.0", "TD all: over 6.0"}
# Oct 18: 8 reader-facing groups for the Record tab; the fine-grained types remain as sub-rows
FAMILY = {"TD singles": ["TD 2.5–3.9, model likes", "TD skip list (not bet)", "TD singles: likeliest scorer per game"],
          "TD parlays (across games)": ["TD parlays: likeliest per game, 2 legs", "TD parlays: likeliest per game, 3 legs", "TD parlays: top-5 totals, 2 legs", "TD parlays: top-5 totals, 3 legs", "TD parlays: top-5 favorites' lead RB", "TD parlays: top-total game, both teams' shortest"],
          "ML + top TD scorer, 2 legs": ["ML + top TD scorer, 2 legs"],
          "Same-game stacks": ["Same-game stacks: favorite + lead RB", "Same-game stacks: top two pass catchers", "Same-game stacks: QB yards + WR1 catches"],
          "Matchup props (opp top-8 vs position)": ["Matchup props: TD, opp top-8 vs position", "Matchup props: receptions over, opp top-8 vs position"],
          "Receptions and alt-line parlays": ["Receptions, model differs 5+ pts", "Parlays (alt lines, 2–3 legs)", "Parlays: steady legs 1.25–1.50"],
          "Teasers (small underdogs +6)": ["Teasers: small underdogs +6, 2 legs"],
          "Trend unders": ["Trend unders: wind 10–19 mph", "Trend unders: prime time"],
          "Game lines": ["Team totals, model differs 2+ pts", "Game totals, model differs 3+ pts", "Moneyline, model differs 4+ pts"],
          "Board consistency (DK vs DK)": ["Board consistency"],
          "Interceptions": ["QB INTs (0.5), model differs 5+ pts"],
          "Every TD price (tracking)": ["TD all: under 2.5", "TD all: 2.5–3.9", "TD all: 4.0–6.0", "TD all: over 6.0"],
          "Control: random picks": ["Control: random"]}
# Pre-registered main tests (Oct 22, before any of them had a graded sample): bets needed before the Read can say anything.
# Chosen for what Clinton bets and the structural angles he wants tested. Do not add to or reshuffle this list mid-season.
PRIMARY = {"ML + top TD scorer, 2 legs": 100,
           "Matchup props: TD, opp top-8 vs position": 100, "Matchup props: receptions over, opp top-8 vs position": 100,
           "Same-game stacks: favorite + lead RB": 75, "Same-game stacks: top two pass catchers": 75, "Same-game stacks: QB yards + WR1 catches": 75,
           "Teasers: small underdogs +6, 2 legs": 60}
PRIMARY_FAMILIES = sorted({f for f, ts in FAMILY.items() if set(ts) & set(PRIMARY)})
FAM_OF = {t: f for f, ts in FAMILY.items() for t in ts}
GROUP = {t: ("skip" if t in _SKIP else "track" if t in _TRACK else "control" if t.startswith("Control") else "bet") for t in ORDER}
pap = [pd.read_csv(f).assign(week=int(re.search(r"wk(\d+)", f).group(1))) for f in sorted(glob.glob("paper_wk*_graded.csv"))]
pap = pd.concat(pap) if pap else pd.DataFrame(columns=["type", "result", "model_p", "price", "pnl", "week", "subject", "market", "side", "line", "actual", "paid_price"])
strategy = []
for t in ORDER:
    x = pap[pap.type == t]; gx = x[x.result.isin(["win", "loss"])]
    n = len(gx); w = int((gx.result == "win").sum()); u = float(gx.pnl.sum()) if n else 0.0
    hit = w / n if n else None; mp = float(gx.model_p.mean()) if n else None; be = float((1 / gx.price).mean()) if n else None
    # Oct 22: pre-registered main tests get a fixed sample size and a 2-sigma bar; everything else is exploratory and needs
    # 150 bets and a 3-sigma bar, because with ~25 rules running some will look good by luck alone.
    need = PRIMARY.get(t, 150)
    z_be = (w - float((1 / gx.price).sum())) / max(float(((1 / gx.price) * (1 - 1 / gx.price)).sum()) ** 0.5, 1e-9) if n else 0.0
    z_md = (w - float(gx.model_p.sum())) / max(float((gx.model_p * (1 - gx.model_p)).sum()) ** 0.5, 1e-9) if n else 0.0
    bar = 2.0 if t in PRIMARY else 3.0
    if n < need: read = f"Too early ({n} of {need})" if t in PRIMARY else "Too early"
    elif u > 0 and z_be >= bar: read = "Earning trust"
    elif u / n < -0.08: read = "Don't bet"
    else: read = "No edge yet"
    if GROUP[t] == "track":   # no real price: graded on whether the model's chance was right
        read = (f"Too early ({n} of {need})" if n < need else "Model about right" if abs(z_md) < 2 else "Model too optimistic" if z_md < 0 else "Model too cautious") if t in PRIMARY else "Tracking only"
    if GROUP[t] == "control": read = "Benchmark"
    cv = gx["clv"].dropna() if "clv" in gx else pd.Series(dtype=float); cp = gx["clv_pts"].dropna() if "clv_pts" in gx else pd.Series(dtype=float)
    clv_ = dict(n=int(len(cv)), avg=round(float(cv.mean()), 4), beat=round(float((cv > 0).mean()), 3)) if len(cv) else None   # Oct 21
    clvp = dict(n=int(len(cp)), avg=round(float(cp.mean()), 2), beat=round(float((cp > 0).mean()), 3)) if len(cp) else None
    strategy.append(dict(type=t, family=FAM_OF.get(t, "Other"), group=GROUP[t], n=n, w=w, pending=int((x.result == "pending").sum()), hit=hit, model=mp, breakeven=be, units=round(u, 2), read=read, primary=t in PRIMARY, need=need,
                         clv=clv_, clv_pts=clvp))
# this week's tests: per family, for the latest week with a slate
thisweek = None
if len(pap):
    cw = int(pap.week.max()); x = pap[pap.week == cw]; fams = []
    for f, ts in FAMILY.items():
        y = x[x.type.isin(ts)]
        if not len(y): continue
        g_ = y[y.result.isin(["win", "loss"])]
        fams.append(dict(family=f, n=len(y), done=len(g_), w=int((g_.result == "win").sum()), expected=round(float(y.model_p.sum()), 1), units=round(float(g_.pnl.sum()), 2),
                         bets=[dict(type=b.type, bet=(f"{b.subject} " + ("anytime TD" if b.market == "atd" else (f"{b.side} {b.line:g}" if b.market not in ("ml", "parlay") else ("moneyline" if b.market == "ml" else "")))).strip(),
                                    price=float(b.paid_price if pd.notna(b.paid_price) else b.price), model_p=float(b.model_p), result=b.result, actual=None if pd.isna(b.actual) else float(b.actual), pnl=float(b.pnl),
                                    fair=None if pd.isna(getattr(b, "fair_price", np.nan)) else float(b.fair_price)) for b in y.itertuples()]))
    thisweek = dict(week=cw, families=fams)

# ---------------- 2. model vs market by week
weekly = []
ou_all = pd.read_csv("props_log_ou.csv") if glob.glob("props_log_ou.csv") else pd.DataFrame()
td_all = pd.read_csv("props_log.csv") if glob.glob("props_log.csv") else pd.DataFrame()
lb, ls = L.load(); gg = L.build(lb, ls)
for w in sorted(set(td_all.week) | set(ou_all.week)):
    row = dict(week=int(w))
    # anytime TD: last pull before games, games finished, player took a snap with a touch
    a = td_all[(td_all.week == w) & td_all.model_p.notna()].sort_values("pulled_at").drop_duplicates(["player", "game_id"], keep="last")
    a = a[a.game_id.isin(done)].copy()
    a["pid"] = [pid(n, gid) for n, gid in zip(a.player, a.game_id)]; a = a[a.pid.isin(touched)]
    if len(a) >= 30:
        a["hit"] = [int(tds.get((gid, p), 0) > 0) for gid, p in zip(a.game_id, a.pid)]
        mk = (1 / a.price) * 0.85
        row["td"] = dict(n=len(a), hits=int(a.hit.sum()), model_exp=round(a.model_p.sum(), 1), mkt_exp=round(mk.sum(), 1),
                         brier_model=round(float(np.mean((a.model_p - a.hit) ** 2)), 4), brier_mkt=round(float(np.mean((mk - a.hit) ** 2)), 4))
        ev = a.model_p * a.price - 1; a["pnl"] = np.where(a.hit == 1, a.price - 1, -1.0)
        mid = a[(ev >= .03) & a.price.between(2.5, 3.9)]
        fade = a[(ev <= -.10) & (a.price >= 1.4) & (a.price < 2.5)]
        band = a[a.price.between(2.5, 3.9)]
        row["td_band"] = dict(n=len(band), hits=int(band.hit.sum()), model_exp=round(float(band.model_p.sum()), 1), mkt_raw=round(float((1 / band.price).sum()), 1),
                              all_units=round(float(band.pnl.sum()), 2),
                              dislike=dict(n=int((ev[band.index] <= -.03).sum()), units=round(float(band.pnl[ev[band.index] <= -.03].sum()), 2)))
        row["td_value"] = dict(n=len(mid), w=int(mid.hit.sum()), units=round(mid.pnl.sum(), 2))
        row["td_fades"] = dict(n=len(fade), w=int(fade.hit.sum()), units_if_bet=round(fade.pnl.sum(), 2))
    # player O/U lines
    u = ou_all[(ou_all.week == w) & ou_all.model_mean.notna()].sort_values("pulled_at").drop_duplicates(["market", "player"], keep="last")
    u = u[u.game_id.isin(done)].copy()
    u["pid"] = [pid(n, gid) for n, gid in zip(u.player, u.game_id)]; u = u[u.pid.isin(touched)]
    if len(u) >= 30:
        u["act"] = [float(STV[m].get((gid, p), 0)) for m, gid, p in zip(u.market, u.game_id, u.pid)]
        row["props"] = {m.replace("player_", ""): dict(n=len(x), mae_model=round(float(np.mean(np.abs(x.act - x.model_mean))), 2),
                        mae_line=round(float(np.mean(np.abs(x.act - x.point))), 2)) for m, x in u.groupby("market")}
    # games (walk-forward: core model fit on everything before this week)
    tr = gg[gg.result.notna() & gg.spread_line.notna() & (gg.min_games >= 3) & ((gg.season < SEASON) | ((gg.season == SEASON) & (gg.week < w)))]
    te = gg[(gg.season == SEASON) & (gg.week == w) & gg.result.notna() & gg.spread_line.notna()]
    if len(te) >= 4:
        mm, mt = L.fit(tr); o = L.predict(te.copy(), mm, mt)
        side = np.sign(o.m_margin - o.spread_line) == np.sign(o.result - o.spread_line)
        ou = np.sign(o.m_total - o.total_line) == np.sign(o.total - o.total_line)
        push_s = (o.result == o.spread_line); push_t = (o.total == o.total_line)
        row["games"] = dict(n=len(o), mae_margin_model=round(float(np.mean(np.abs(o.result - o.m_margin))), 1), mae_margin_line=round(float(np.mean(np.abs(o.result - o.spread_line))), 1),
                            mae_total_model=round(float(np.mean(np.abs(o.total - o.m_total))), 1), mae_total_line=round(float(np.mean(np.abs(o.total - o.total_line))), 1),
                            ats_w=int((side & ~push_s).sum()), ats_l=int((~side & ~push_s).sum()), ou_w=int((ou & ~push_t).sum()), ou_l=int((~ou & ~push_t).sum()))
    weekly.append(row)

# ---------------- 3. verdicts (rule-based, honest about sample size)
def verdict(name, n, edge_ok, edge_bad, note):
    if n < 40: lab = "Too early"
    elif edge_ok: lab = "Earning trust"
    elif edge_bad: lab = "Don't bet"
    else: lab = "No edge yet"
    return dict(name=name, label=lab, n=n, note=note)
tdw = [x for x in weekly if "td" in x]
bm = np.mean([x["td"]["brier_model"] for x in tdw]) if tdw else None; bk = np.mean([x["td"]["brier_mkt"] for x in tdw]) if tdw else None
ntd = sum(x["td"]["n"] for x in tdw)
val = [x["td_value"] for x in weekly if "td_value" in x]; fad = [x["td_fades"] for x in weekly if "td_fades" in x]
vu, vn = sum(v["units"] for v in val), sum(v["n"] for v in val); fu, fn = sum(v["units_if_bet"] for v in fad), sum(v["n"] for v in fad)
pw = [x for x in weekly if "props" in x]
def pm(key):
    xs = [x["props"][key] for x in pw if key in x["props"]]; n = sum(v["n"] for v in xs)
    return n, (sum(v["mae_model"] * v["n"] for v in xs) / n if n else None), (sum(v["mae_line"] * v["n"] for v in xs) / n if n else None)
gw = [x["games"] for x in weekly if "games" in x]
gn = sum(v["n"] for v in gw); atsw = sum(v["ats_w"] for v in gw); atsl = sum(v["ats_l"] for v in gw); ouw = sum(v["ou_w"] for v in gw); oul = sum(v["ou_l"] for v in gw)
verdicts = []
if ntd: verdicts.append(verdict("Anytime TD probabilities", ntd, bm < bk - 0.002, bm > bk + 0.004,
    f"Accuracy score {bm:.3f} vs DK {bk:.3f} (lower is better) over {ntd} priced players."))
if vn: verdicts.append(verdict("TDs at 2.5–3.9 the model likes (+3% edge or more)", vn, vu > 0, vu < -0.25 * vn,
    f"{vn} bets, {vu:+.1f} units flat-staked. Needs ~100+ bets before the sign means much."))
bw = [x["td_band"] for x in weekly if "td_band" in x]
if bw:
    bn = sum(x["n"] for x in bw); bh = sum(x["hits"] for x in bw); be = sum(x["model_exp"] for x in bw); bk_ = sum(x["mkt_raw"] for x in bw)
    dn = sum(x["dislike"]["n"] for x in bw); du = sum(x["dislike"]["units"] for x in bw)
    verdicts.insert(1, verdict("All TDs priced 2.5–3.9", bn, bh > be + 2 and abs(bh - be) < abs(bh - bk_), False,
        f"{bh} of {bn} scored; model expected {be:.1f}, DK's prices implied {bk_:.1f} (with its margin). Betting the {dn} it disliked would have returned {du:+.1f}u — " + ("so skipping them was right." if du < 0 else "so its dislikes in this range haven't held up.")))
if fn: verdicts.append(verdict("TD fades under 2.5 (short favorites the model dislikes)", fn, fu < 0, fu > 0.1 * fn,
    f"Betting these would have returned {fu:+.1f} units over {fn} — negative means skipping them was right."))
for key, nm in [("receptions", "Receptions"), ("reception_yds", "Receiving yards"), ("rush_yds", "Rushing yards"), ("pass_yds", "Passing yards")]:
    n, mm_, ml_ = pm(key)
    if n: verdicts.append(verdict(f"{nm} vs DK line", n, mm_ < ml_ - 0.02 * ml_, mm_ > ml_ * 1.03,
        f"Average miss {mm_:.1f} vs DK {ml_:.1f} over {n} lines." + (" DK's line already includes what the model knows — treat gaps as information, not bets." if key != "receptions" else "")))
import os
BT = json.load(open("data/lines_bt_summary.json")) if os.path.exists("data/lines_bt_summary.json") else None
if BT:
    for key, nm in [("spreads", "Spread leans"), ("totals", "Game total leans")]:
        b = BT[key]; r = b["w"] / b["n"]
        verdicts.append(verdict(nm, b["n"], r > .545, r < .50,
            f"Backtest {BT['seasons']}: when the model differs from DK by 3+ points it's {b['w']}–{b['n']-b['w']} ({r:.1%}); 52.4% breaks even. Average miss {b['mae_model']} vs DK {b['mae_line']}. Weeks {min(x['week'] for x in weekly if 'games' in x)}–{max(x['week'] for x in weekly if 'games' in x)} this season: {atsw if key=='spreads' else ouw}–{atsl if key=='spreads' else oul}."))
    gn = 0
if gn: verdicts.append(verdict("Spreads", gn, atsw + atsl and atsw / (atsw + atsl) > .55, atsw + atsl and atsw / (atsw + atsl) < .45,
    f"Model side {atsw}–{atsl} against the spread (core model, no-look-ahead)."))
if gn: verdicts.append(verdict("Game totals", gn, ouw + oul and ouw / (ouw + oul) > .55, ouw + oul and ouw / (ouw + oul) < .45,
    f"Model side {ouw}–{oul} on totals."))
paper_list = [dict(week=int(b.week), type=b.type, bet=(f"{b.subject} " + ("anytime TD" if b.market == "atd" else (f"{b.side} {b.line:g}" if b.market != "ml" and b.market != "parlay" else ("moneyline" if b.market == "ml" else "")))).strip(),
                   price=float(b.paid_price if pd.notna(b.paid_price) else b.price), model_p=float(b.model_p), result=b.result, actual=None if pd.isna(b.actual) else float(b.actual), pnl=float(b.pnl))
              for b in pap.itertuples() if GROUP.get(b.type) in ("bet", "skip", "control") or str(b.type).startswith("Same-game") or str(b.type).startswith("ML + top")]
out = dict(strategy=strategy, thisweek=thisweek, families=[f for f in FAMILY if f in PRIMARY_FAMILIES] + [f for f in FAMILY if f not in PRIMARY_FAMILIES], primary_families=PRIMARY_FAMILIES, paper=paper_list, asof=pd.Timestamp.now(tz="America/New_York").strftime("%a %b %-d, %-I:%M %p ET"), verdicts=verdicts, types=types, weekly=weekly,
           bets=[dict(week=int(r.week), slate=r.slate, type=r.type, bet=r.bet, price=float(r.price), model_p=float(r.model_p), result=r.result,
                      actual=None if pd.isna(r.actual) else float(r.actual), pnl=float(r.pnl)) for r in bets.itertuples()] if len(bets) else [],
           totals=dict(n=len(g), w=int((g.result == "win").sum()), units=round(float(g.pnl.sum()), 2)) if len(g) else None)
json.dump(out, open("record.json", "w"), allow_nan=False, default=lambda o: None)
print(json.dumps(out["verdicts"], indent=1)); print("types:", out["types"]); print("weeks:", [w["week"] for w in weekly])
