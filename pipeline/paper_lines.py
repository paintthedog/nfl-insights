"""Automatic game-line paper slate (Oct 25). Clinton stopped the manual Odds API pulls: their timing was his call, which made the
record partly about when he pulled. Every price here comes from nflverse's schedule feed (data/sched.parquet: spread, total,
moneylines and their prices), refreshed by refresh_data.py on every daily run, so the timing is fixed: the bet is logged at the
LAST daily run before kickoff (8:52 a.m. ET game-day morning; Thursday and Monday games get that morning's run).
Nothing is hand-picked. Re-running replaces this script's rows only for games that have not kicked off.
Rules (all mechanical):
  Teasers: small underdogs +6, 2 legs    main test, unchanged rule (+1.5 to +2.5 teased to +7.5..+8.5), legs paired in kickoff order
  Teaser legs: small underdogs +6        each of those legs graded alone (reference price 1.80**0.5)
  Teasers (wide): underdogs +1 to +3, all pairs   exploratory comparison; the 2016-2025 backtest says it dilutes
  Moneyline, model differs 4+ pts        model win chance (raw margin) vs the feed's no-vig moneyline
  Game totals, model differs 3+ pts      model total vs the feed's total
  Trend unders: wind 10-19 mph / prime time    the long-run patterns in trends.json
  Control: random                        3 random game-total sides, the benchmark for what the vig alone costs
Usage: python paper_lines.py W   (refresh_all.sh runs it for every week it builds)"""
import sys, json, itertools, os
from math import erf, sqrt
import numpy as np, pandas as pd

Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
dec = lambda a: None if pd.isna(a) else (1 + a / 100 if a > 0 else 1 + 100 / -a)
TEASE = 1.80
TYPES = ["Teasers: small underdogs +6, 2 legs", "Teaser legs: small underdogs +6", "Teasers (wide): underdogs +1 to +3, all pairs",
         "Moneyline, model differs 4+ pts", "Game totals, model differs 3+ pts", "Trend unders: wind 10–19 mph", "Trend unders: prime time", "Control: random"]
SRC = "nflverse"   # every row carries market_src so the record can tell these from the old DK-pull rows


def leg_rate(sched, lo, hi):
    h = sched[(sched.season >= 2016) & (sched.season < 2026) & (sched.game_type == "REG") & sched.result.notna() & sched.spread_line.notna()]
    out = []
    for r in h.itertuples():
        for home in (True, False):
            line = -r.spread_line if home else r.spread_line; mg = r.result if home else -r.result
            if lo <= line <= hi and mg + line + 6 != 0: out.append(mg + line + 6 > 0)
    return float(np.mean(out)) if out else 0.75


def build(W, now=None, locked=frozenset()):
    """locked: game_ids already used in a main-rule teaser that has partly kicked off (kept as logged), so they are not paired twice."""
    now = now or pd.Timestamp.now(tz="America/New_York"); stamp = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
    rng = np.random.default_rng(2000 + W)
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
    L = json.load(open(f"lines_2026_wk{W}.json")); SDM, SDT = L["sd_margin"], L["sd_total"]
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")
    games = [g for g in L["games"] if g["game_id"] in sw.index and kick(g["game_id"]) > now]
    try: TA = json.load(open("trends.json"))["applies"]
    except Exception: TA = {}
    try: fc = pd.read_csv(f"forecasts_wk{W}.csv").set_index("team")
    except Exception: fc = pd.DataFrame()
    rows = []
    def add(t, gid, subj, market, side, line, price, p, legs=None):
        rows.append(dict(week=W, type=t, game_id=gid, subject=subj, market=market, side=side, line=line, price=round(float(price), 3),
                         model_p=round(float(p), 4), legs=json.dumps(legs) if legs else "", placed_at=stamp, market_src=SRC))
    p_main, p_wide = leg_rate(sched, 1.5, 2.5), leg_rate(sched, 1.0, 3.0)
    legs = []; totals = []
    for g in sorted(games, key=lambda g: (str(sw.loc[g["game_id"]].gameday), str(sw.loc[g["game_id"]].gametime), g["game_id"])):
        gid = g["game_id"]; s = sw.loc[gid]
        if pd.notna(s.spread_line):              # nflverse spread_line = home margin the market expects; home handicap = -spread_line
            for team, hcap in ((s.home_team, -s.spread_line), (s.away_team, s.spread_line)):
                if 1.0 <= hcap <= 3.0:
                    legs.append(dict(gid=gid, player=team, stat="spread", line=float(hcap) + 6, price=round(TEASE ** 0.5, 4),
                                     p=round(p_main if 1.5 <= hcap <= 2.5 else p_wide, 4), label=f"{team} {float(hcap) + 6:+g}", main=bool(1.5 <= hcap <= 2.5)))
        hp, ap = dec(s.home_moneyline), dec(s.away_moneyline)
        if hp and ap:
            ph = Phi(g["m_margin"] / SDM); mh = (1 / hp) / (1 / hp + 1 / ap)
            if abs(ph - mh) >= 0.04:
                team, price, p = (s.home_team, hp, ph) if ph > mh else (s.away_team, ap, 1 - ph)
                add("Moneyline, model differs 4+ pts", gid, team, "ml", "ML", None, price, p)
        op, up = dec(s.over_odds), dec(s.under_odds)
        if pd.notna(s.total_line) and op and up:
            ln = float(s.total_line); po = 1 - Phi((ln - g["m_total"]) / SDT); totals.append((gid, ln, op, up, po))
            sub = f"{g['away']}@{g['home']}"
            if abs(g["m_total"] - ln) >= 3:
                side = "Over" if g["m_total"] > ln else "Under"
                add("Game totals, model differs 3+ pts", gid, sub, "game_total", side, ln, op if side == "Over" else up, po if side == "Over" else 1 - po)
            dome = str(s.roof) in ("dome", "closed")
            wind = float(fc.wind[s.home_team]) if s.home_team in fc.index else None
            if not dome and wind is not None and 10 <= wind <= 19 and TA.get("wind"):
                add("Trend unders: wind 10–19 mph", gid, sub, "game_total", "Under", ln, up, TA["wind"]["pct"] / 100)
            if int(str(s.gametime)[:2]) >= 19 and TA.get("prime"):
                add("Trend unders: prime time", gid, sub, "game_total", "Under", ln, up, TA["prime"]["pct"] / 100)
    strip = lambda l: {k: v for k, v in l.items() if k != "main"}
    main = [l for l in legs if l["main"] and l["gid"] not in locked]
    for i in range(0, len(main) - 1, 2):
        c = main[i:i + 2]
        if c[0]["gid"] != c[1]["gid"]:
            add(TYPES[0], "", " + ".join(l["label"] for l in c), "parlay", "All", None, TEASE, c[0]["p"] * c[1]["p"], legs=[strip(l) for l in c])
    for l in main: add(TYPES[1], l["gid"], l["label"], "parlay", "All", None, l["price"], l["p"], legs=[strip(l)])
    for a, b in itertools.combinations(legs, 2):
        if a["gid"] != b["gid"]: add(TYPES[2], "", f"{a['label']} + {b['label']}", "parlay", "All", None, TEASE, a["p"] * b["p"], legs=[strip(a), strip(b)])
    for i in (rng.choice(len(totals), size=min(3, len(totals)), replace=False) if totals else []):
        gid, ln, op, up, po = totals[i]; side = "Over" if rng.random() < .5 else "Under"
        add("Control: random", gid, gid.split("_", 2)[2].replace("_", "@"), "game_total", side, ln, op if side == "Over" else up, po if side == "Over" else 1 - po)
    return pd.DataFrame(rows), {g["game_id"] for g in games}


if __name__ == "__main__":
    W = int(sys.argv[1])
    if W < 8: sys.exit(0)   # Week 7 keeps its DraftKings-pull slate; the automatic slate starts with Week 8
    _, live = build(W)
    f = f"paper_wk{W}.csv"; old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=_.columns)
    if "market_src" not in old: old["market_src"] = "draftkings"
    def mine_live(r):   # this script's rows for games that have not started (a multi-game teaser counts as live only if every leg is)
        if r.get("market_src") != SRC or r.type not in TYPES: return False
        gids = [l["gid"] for l in json.loads(r.legs)] if isinstance(r.legs, str) and r.legs else [r.game_id]
        return all(g in live for g in gids)
    keep = old[~old.apply(mine_live, axis=1)] if len(old) else old
    locked = {l["gid"] for r in keep[(keep.type == TYPES[0]) & (keep.get("market_src") == SRC)].itertuples() for l in json.loads(r.legs)} if len(keep) else set()
    new, live = build(W, locked=frozenset(locked))
    pd.concat([keep, new], ignore_index=True).to_csv(f, index=False)
    print(f"{f}: {len(new)} game-line paper bets for {len(live)} games not yet started" + (": " + ", ".join(f"{k} {v}" for k, v in new.type.value_counts().items()) if len(new) else ""))
