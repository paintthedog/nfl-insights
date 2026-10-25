"""Oct 25 (Clinton: "we're not testing nearly enough bets"): EXPLORATORY rules that widen the main tests. The main tests stay exactly
as pre-registered on Oct 22; these run beside them, are graded the same way, and need 150 bets and 3 standard errors (record.py).
  - Teaser legs: small underdogs +6       each leg of the main teaser rule (+1.5 to +2.5) graded on its own, reference price 1.80**0.5
  - Teasers (wide): underdogs +1 to +3, all pairs   every 2-leg pairing of qualifying underdogs from different games, ref price 1.80
  - Same-game stacks (underdog): ...      sgp_slate's three stacks on the DK underdog instead of the favorite
  - Matchup props: receptions over, opp top-8 (no TD filter)   Clinton's matchup rule without the TD-rate half
Only games that have not kicked off are logged. paper_slate.py calls build(); `python paper_extra.py W` re-logs a week in place
(replaces only these types, and only for games not yet started)."""
import sys, json, re, itertools
import numpy as np, pandas as pd

TL, TW = "Teaser legs: small underdogs +6", "Teasers (wide): underdogs +1 to +3, all pairs"
RX = "Matchup props: receptions over, opp top-8 (no TD filter)"
TEASE_PRICE = 1.80
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()


def _p_leg(sched, lo, hi):
    """Historical 6-point teaser leg cover rate for underdogs getting lo..hi, 2016-2025 regular season (pushes dropped)."""
    h = sched[(sched.season >= 2016) & (sched.season < 2026) & (sched.game_type == "REG") & sched.result.notna() & sched.spread_line.notna()]
    out = []
    for r in h.itertuples():
        for home in (True, False):
            line = -r.spread_line if home else r.spread_line; mg = r.result if home else -r.result
            if lo <= line <= hi and mg + line + 6 != 0: out.append(mg + line + 6 > 0)
    return float(np.mean(out)) if out else 0.75


def build(W, now=None):
    import sgp_slate
    now = now or pd.Timestamp.now(tz="America/New_York"); stamp = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
    D = json.load(open(f"site/data_wk{W}.json")); L = json.load(open(f"lines_2026_wk{W}.json")); QS = D["proj"]["quants"]
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")
    live = {g["game_id"] for g in L["games"] if g["game_id"] in sw.index and kick(g["game_id"]) > now}
    M = pd.read_csv("market_log.csv"); M = M[M.week == W].sort_values("pulled_at").drop_duplicates(["game_id", "market", "subject", "side", "point"], keep="last")
    rows = []
    def add(t, gid, subj, market, side, line, price, p, legs=None):
        rows.append(dict(week=W, type=t, game_id=gid, subject=subj, market=market, side=side, line=line, price=round(float(price), 3),
                         model_p=round(float(p), 4), legs=json.dumps(legs) if legs else "", placed_at=stamp))
    # ---- teasers
    p_main, p_wide = _p_leg(sched, 1.5, 2.5), _p_leg(sched, 1.0, 3.0)
    legs = []
    for _, r in M[M.market == "game_spreads"].iterrows():
        pt = float(r.point)
        if r.game_id in live and 1.0 <= pt <= 3.0:
            legs.append(dict(gid=r.game_id, player=r.subject, stat="spread", line=pt + 6, price=round(TEASE_PRICE ** 0.5, 4),
                             p=round(p_main if 1.5 <= pt <= 2.5 else p_wide, 4), label=f"{r.subject} {pt + 6:+g}", main=1.5 <= pt <= 2.5))
    strip = lambda l: {k: v for k, v in l.items() if k != "main"}
    for l in legs:
        if l["main"]: add(TL, l["gid"], l["label"], "parlay", "All", None, l["price"], l["p"], legs=[strip(l)])
    for a, b in itertools.combinations(legs, 2):
        if a["gid"] != b["gid"]:
            add(TW, "", f"{a['label']} + {b['label']}", "parlay", "All", None, TEASE_PRICE, a["p"] * b["p"], legs=[strip(a), strip(b)])
    # ---- underdog same-game stacks (sgp_slate already skips games that have started)
    dog = sgp_slate.build(W, now=now, side="dog")
    # ---- matchup receptions without the TD-rate filter
    def p_over(arr, line):
        arr = np.asarray(arr, float)
        if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
        for i in range(1, len(arr)):
            if line <= arr[i]:
                t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
        return max(0, (1 - QS[-1]) * .5)
    gid_of = {}
    for g in L["games"]: gid_of[g["home"]] = g["game_id"]; gid_of[g["away"]] = g["game_id"]
    for p in D["proj"]["players"]:
        dk = ((p.get("lines") or {}).get("rec") or {}).get("draftkings"); q = (p.get("q") or {}).get("rec"); gid = gid_of.get(p["team"])
        if not dk or not q or gid not in live or p["pos"] not in ("RB", "WR", "TE") or p.get("opp_rank") is None: continue
        po = p_over(q, dk["pt"]); mo = (1 / dk["o"]) / (1 / dk["o"] + 1 / dk["u"])
        if p["opp_rank"] <= 8 and po >= mo and (p.get("p_play") or 1) >= 0.85:
            add(RX, gid, p["name"], "rec", "Over", dk["pt"], dk["o"], po)
    # Clinton's standing rule: no skill player without a real season record (8+ catches for pass catchers, 25+ carries for backs).
    pb = pd.read_parquet("data/pbp_2026.parquet", columns=["week", "receiver_player_id", "rusher_player_id", "complete_pass", "rush_attempt"]); pb = pb[pb.week < W]
    cat, car = pb.groupby("receiver_player_id").complete_pass.sum(), pb.groupby("rusher_player_id").rush_attempt.sum()
    pid = {norm(p["name"]): p.get("pid") for p in D["proj"]["players"]}
    ok = lambda name, st: (car.get(pid.get(norm(name)), 0) >= 25) if st == "rush_yds" else (cat.get(pid.get(norm(name)), 0) >= 8) if st == "rec" else True
    rows = [r for r in rows if r["type"] != RX or ok(r["subject"], "rec")]
    if len(dog): dog = dog[dog.legs.map(lambda s: all(ok(l["player"], l["stat"]) for l in json.loads(s)))]
    out = pd.concat([pd.DataFrame(rows), dog], ignore_index=True)
    return out, live


TYPES = None
if __name__ == "__main__":
    import sgp_slate
    W = int(sys.argv[1]); new, live = build(W)
    types = {TL, TW, RX, *sgp_slate.TYPES_DOG.values()}
    f = f"paper_wk{W}.csv"; old = pd.read_csv(f)
    started = lambda r: not (r.game_id in live or (r.game_id == "" or pd.isna(r.game_id)) and all(l["gid"] in live for l in json.loads(r.legs)))
    keep = old[~old.type.isin(types) | old.apply(started, axis=1)]
    pd.concat([keep, new], ignore_index=True).to_csv(f, index=False)
    print(f"{f}: +{len(new)} exploratory rows for {len(live)} games not yet started")
    if len(new): print(new.groupby("type").size().to_string())
