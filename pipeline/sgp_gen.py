"""Same-game parlay generator (Nov 23, Clinton: "within one game, two or at most three bets that might double your money").

For every game not yet kicked off it builds 2-3 leg same-game combos whose MODEL fair price (1 / joint chance) lands near 2.0,
from the menu Clinton picked: player yards/catches thresholds (round alt-style numbers) plus alternate spreads and game totals.
No sportsbook prices anywhere: the fair price is what the model says the combo is worth, so a DraftKings same-game price ABOVE it
is the only case worth betting, and DK prices these links too. Moneylines and TD legs are not on the menu (Clinton, Nov 23).

Legs
  player: QB pass yds (150-350 by 25), lead RB rush yds, WR1/WR2/TE1 catches and receiving yds (combo_slate rungs); the model's
          published distribution gives the chance. Eligible: p_play 0.85+ and a real season record (8+ catches / 25+ carries,
          Clinton's standing rule). Each leg's chance must be 0.60-0.90.
  spread: the favorite at an alternate half-point spread (no pushes), always laying points: "wins" stories need a win by 1+
          or more. (A "blowout" story, favorite covers the closing spread + its RB + the other QB, was dropped Nov 23: a
          ~50% cover leg can't reach a 2.0 fair price with two more legs.) Chance = the site's line-anchored cover chance carried to the
          alternate number along the historical distribution of (margin - closing spread), so key numbers 3 and 7 keep their weight.
  total:  over/under an alternate half-point total, the same way from (points - closing total).
Joint chance: parlay_joint.joint_signed (Gaussian copula on historical same-game links, fitted per role; COV = margin vs the
spread, TOT = points vs the total; added Nov 23).

Stories (fixed order = how much evidence each has; see TEST_LEDGER Nov 23 for the 2024-25 replay backtest):
  see STORIES below. Per game the generator keeps the best combo of each story (fair price 1.90-2.30, the most balanced legs),
  and the site shows the first three in story order.
Usage (from pipeline/): python sgp_gen.py W         log week W -> sgpgen_wk{W}.csv (replaces only games not yet started)
                        python sgp_gen.py grade     grade every sgpgen_wk*.csv -> sgp_report.json
                        python sgp_gen.py backtest 2024 2025   on replay/out (replay/run.py), copula fitted through 2023"""
import sys, json, glob, itertools, re
import numpy as np, pandas as pd
from parlay_joint import joint_signed

FAIR_LO, FAIR_HI = 1.90, 2.30          # fair-price window ("double your money")
P_LO, P_HI = 0.60, 0.90                # each leg
KS = {"rec": list(range(1, 13)), "pass_yds": list(range(150, 351, 25)),
      "rush_yds": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]}
KS["rec_yds"] = KS["rush_yds"]
UNIT = {"rec": "catches", "rush_yds": "rush yds", "pass_yds": "pass yds", "rec_yds": "rec yds"}
# (key, name, anchor team, leg slots). Slots: COV = anchor covers an alt spread, OPPCOV, OVER, UNDER, or a role-stat of the anchor
# (QB pass_yds ...) or of the opponent ("OPP QB pass_yds").
STORIES = [   # Nov 23 order: live 2026 combos first (win + lead RB 62.5% vs 50% said; QB + WR1 56.8 vs 52.9), then the 2024-25 backtest z
    ("cover_rb", "Favorite wins (alt spread) + its lead RB yards", "fav", ["COV", "RB1 rush_yds"]),
    ("qb_wr1", "QB yards + WR1 catches", "each", ["QB pass_yds", "WR1 rec"]),
    ("top2", "Top two pass catchers' catches", "each", ["WR1 rec", "TE1 rec|WR2 rec"]),
    ("over_qb_wr1", "Over + QB yards + WR1 yards", "each", ["OVER", "QB pass_yds", "WR1 rec_yds"]),
    ("qb_te1", "QB yards + top TE catches", "each", ["QB pass_yds", "TE1 rec"]),
    ("over_qbs", "Shootout: over + both QBs' yards", "fav", ["OVER", "QB pass_yds", "OPP QB pass_yds"]),
    ("under_rb", "Under + favorite wins (alt spread) + its RB yards", "fav", ["UNDER", "COV", "RB1 rush_yds"]),
]
SHOW = 3


def p_over(arr, line, QS):
    arr = np.asarray(arr, float)
    if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
    for i in range(1, len(arr)):
        if line <= arr[i]:
            t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
    return max(0, (1 - QS[-1]) * .5)


def residuals(S, sched):
    """(margin - spread) and (points - total) from the ten seasons before S (regular season, closing lines)."""
    h = sched[(sched.season < S) & (sched.season >= S - 10) & sched.result.notna() & sched.spread_line.notna() & sched.total_line.notna()]
    return (h.result - h.spread_line).values.astype(float), (h.total - h.total_line).values.astype(float)


def game_legs(g, RM, RT):
    """alt spreads per team and alt totals, anchored on the site's cover / over chances at the closing line."""
    if g.get("spread") is None or g.get("total") is None: return {}, []
    pc = min(max(g.get("p_home_cover") or .5, .02), .98); po = min(max(g.get("p_over") or .5, .02), .98)
    dm = -np.quantile(RM, 1 - pc); dt = -np.quantile(RT, 1 - po)        # P(resid + d > 0) = the site's chance
    home_m = g["spread"] + dm + RM; tot = g["total"] + dt + RT
    spreads = {}
    for team, sgn in ((g["home"], 1), (g["away"], -1)):
        out = []
        for h in np.arange(-24.5, 25, 1.0):                             # team +h covers when its margin + h > 0
            p = float(np.mean(sgn * home_m + h > 0))
            if P_LO <= p <= P_HI: out.append(dict(kind="spread", team=team, h=float(h), p=round(p, 4), text=f"{team} {h:+g}"))
        spreads[team] = out
    totals = []
    base = np.floor(g["total"]) + .5
    for x in np.arange(base - 16, base + 16.5, 1.0):
        po_ = float(np.mean(tot > x))
        if P_LO <= po_ <= P_HI: totals.append(dict(kind="total", side="over", x=float(x), p=round(po_, 4), text=f"Over {x:g}"))
        if P_LO <= 1 - po_ <= P_HI: totals.append(dict(kind="total", side="under", x=float(x), p=round(1 - po_, 4), text=f"Under {x:g}"))
    return spreads, totals


def roles(P, team, REC, CAR):
    tp = [p for p in P if p["team"] == team and (p.get("p_play") or 1) >= .85]
    real = lambda p: p["pos"] == "QB" or (REC.get(p.get("pid"), 0) >= 8 if p["pos"] in ("WR", "TE") else CAR.get(p.get("pid"), 0) >= 25)
    R = {}
    qbs = [p for p in tp if p["pos"] == "QB"]
    if qbs: R["QB"] = max(qbs, key=lambda p: p["mean"].get("pass_yds") or 0)
    rbs = sorted([p for p in tp if p["pos"] == "RB" and real(p)], key=lambda p: -(p["mean"].get("car") or 0))
    if rbs: R["RB1"] = rbs[0]
    wr = sorted([p for p in tp if p["pos"] == "WR" and real(p)], key=lambda p: -(p["mean"].get("tgt") or 0))
    te = sorted([p for p in tp if p["pos"] == "TE" and real(p)], key=lambda p: -(p["mean"].get("tgt") or 0))
    if wr: R["WR1"] = wr[0]
    if len(wr) > 1: R["WR2"] = wr[1]
    if te: R["TE1"] = te[0]
    return R


def player_rungs(p, st, QS):
    q = (p.get("q") or {}).get(st)
    if not q or max(q) <= 0: return []
    out = []
    for k in KS[st]:
        pr = p_over(q, k - .5, QS)
        if P_LO <= pr <= P_HI: out.append(dict(kind="player", player=p["name"], pid=p.get("pid"), team=p["team"], stat=st, k=k,
                                               line=k - .5, p=round(float(pr), 4), text=f"{p['name']} {k}+ {UNIT[st]}"))
    return out


def build_game(g, P, QS, REC, CAR, RM, RT, ph):
    """all stories for one game -> list of combos (best per story and anchor team)."""
    spreads, totals = game_legs(g, RM, RT)
    fav = g["home"] if ph >= .5 else g["away"]
    teams = {g["home"]: g["away"], g["away"]: g["home"]}
    RL = {t: roles(P, t, REC, CAR) for t in teams}
    out = []
    for key, name, anchor, slots in STORIES:
        for A in ([fav] if anchor == "fav" else list(teams)):
            B = teams[A]; cand = []
            for s in slots:
                opts = []
                for alt in s.split("|"):
                    # the favorite's spread leg never gets points: COV = wins by 1+ or more (the alt-spread version of "wins"),
                    # COVMAIN = covers the closing spread or more (Nov 23: without this the most-balanced rule teased favorites to +14.5)
                    if alt == "COV": opts += [dict(l, label="COV", sign=1) for l in spreads.get(A, []) if l["h"] <= -0.5]
                    elif alt == "COVMAIN": opts += [dict(l, label="COV", sign=1) for l in spreads.get(A, []) if l["h"] <= min(-0.5, -abs(g["spread"]))]
                    elif alt == "OVER": opts += [dict(l, label="TOT", sign=1) for l in totals if l["side"] == "over"]
                    elif alt == "UNDER": opts += [dict(l, label="TOT", sign=-1) for l in totals if l["side"] == "under"]
                    else:
                        opp = alt.startswith("OPP "); role, st = alt.replace("OPP ", "").split()
                        p = RL[B if opp else A].get(role)
                        if p: opts += [dict(l, label=alt, sign=1, role=role) for l in player_rungs(p, st, QS)]
                if not opts: break
                cand.append(opts)
            if len(cand) < len(slots): continue
            best = None
            for combo in itertools.product(*cand):
                ids = [(l.get("pid"), l.get("stat")) for l in combo if l["kind"] == "player"]
                if len({i[0] for i in ids}) < len(ids): continue           # one leg per player
                ip = float(np.prod([l["p"] for l in combo]))
                if ip > 1 / FAIR_LO * 1.25 or ip < 1 / FAIR_HI * .7: continue   # cheap prefilter before the copula
                jp = joint_signed([(l["label"], l["p"], l["sign"]) for l in combo])
                fair = 1 / max(jp, 1e-9)
                if not FAIR_LO <= fair <= FAIR_HI: continue
                score = (min(l["p"] for l in combo), -abs(fair - 2.0))
                if best is None or score > best[0]: best = (score, combo, jp, ip)
            if best:
                _, combo, jp, ip = best
                out.append(dict(story=key, story_name=name, team=A, model_p=round(jp, 4), indep_p=round(ip, 4), fair=round(1 / jp, 2),
                                subject=" + ".join(l["text"] for l in combo),
                                legs=[{k: v for k, v in l.items() if k not in ("role",)} for l in combo]))
    return out


def season_counts(S, W):
    pbp = pd.read_parquet(f"data/pbp_{S}.parquet", columns=["week", "receiver_player_id", "complete_pass", "rusher_player_id", "rush_attempt", "two_point_attempt"])
    pbp = pbp[(pbp.week < W) & pbp.two_point_attempt.ne(1)]
    return pbp.groupby("receiver_player_id").complete_pass.sum().to_dict(), pbp[pbp.rush_attempt.eq(1)].groupby("rusher_player_id").size().to_dict()


def ph_of(g, L):
    from sgp_slate import anchored_home_win
    return anchored_home_win(g, L["sd_margin"], L.get("sd_win"))


def pick_shown(combos):
    """per game: story order, one combo per story, no repeated subject."""
    order = {s[0]: i for i, s in enumerate(STORIES)}
    seen, shown = set(), []
    for c in sorted(combos, key=lambda c: (order[c["story"]], -c["model_p"])):
        if c["story"] in seen: continue
        seen.add(c["story"]); shown.append(c)
    return shown[:SHOW]


# ---------------------------------------------------------------- live
def log(W, S=2026):
    D = json.load(open(f"site/data_wk{W}.json")); L = json.load(open(f"lines_{S}_wk{W}.json"))
    QS = D["proj"]["quants"]; P = D["proj"]["players"]
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == S) & (sched.week == W)].set_index("game_id")
    now = pd.Timestamp.now(tz="America/New_York")
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")
    REC, CAR = season_counts(S, W); RM, RT = residuals(S, sched)
    rows = []
    stamp = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
    for g in L["games"]:
        gid = g["game_id"]
        if gid not in sw.index or kick(gid) <= now: continue
        allc = build_game(g, P, QS, REC, CAR, RM, RT, ph_of(g, L))
        shown = {id(c) for c in pick_shown(allc)}
        for c in allc:
            rows.append(dict(week=W, game_id=gid, story=c["story"], story_name=c["story_name"], team=c["team"], subject=c["subject"],
                             n_legs=len(c["legs"]), model_p=c["model_p"], indep_p=c["indep_p"], fair=c["fair"], shown=int(id(c) in shown),
                             legs=json.dumps(c["legs"]), placed_at=stamp))
    new = pd.DataFrame(rows); f = f"sgpgen_wk{W}.csv"
    try: old = pd.read_csv(f)
    except FileNotFoundError: old = pd.DataFrame(columns=["game_id"])
    if len(old):
        old = old[old.game_id.map(lambda gid: gid in sw.index and kick(gid) <= now)]   # keep what was logged before kickoff
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
    print(f"{f}: {len(new)} SGP combos for {new.game_id.nunique() if len(new) else 0} games not yet started ({len(old)} kept)")


# ---------------------------------------------------------------- grading
def actuals(S):
    """per (game_id, pid): catches, rush yds, pass yds, rec yds; plus who had an offensive snap."""
    pbp = pd.read_parquet(f"data/pbp_{S}.parquet", columns=["game_id", "receiver_player_id", "complete_pass", "receiving_yards", "rusher_player_id",
                                                           "rushing_yards", "passer_player_id", "passing_yards", "two_point_attempt", "desc"])
    pbp = pbp[pbp.two_point_attempt.ne(1)]
    done = set(pbp.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
    A = {}
    for st, col, idc in (("rec", "complete_pass", "receiver_player_id"), ("rec_yds", "receiving_yards", "receiver_player_id"),
                         ("rush_yds", "rushing_yards", "rusher_player_id"), ("pass_yds", "passing_yards", "passer_player_id")):
        A[st] = pbp.groupby(["game_id", idc])[col].sum().to_dict()
    sn = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_snaps"])
    on = set(map(tuple, sn[sn.offense_snaps.fillna(0) > 0][["game_id", "gsis_id"]].values))
    return A, on, done


def leg_result(l, gid, A, on, sch):
    g = sch.loc[gid]
    if l["kind"] == "spread":
        m = (g.home_score - g.away_score) * (1 if l["team"] == g.home_team else -1)
        return "hit" if m + l["h"] > 0 else "miss"
    if l["kind"] == "total":
        t = g.home_score + g.away_score
        return "hit" if (t > l["x"]) == (l["side"] == "over") else "miss"
    k = (gid, l["pid"])
    if k not in on and not any(k in A[s] for s in A): return "void"      # never on the field: scratched
    return "hit" if float(A[l["stat"]].get(k, 0) or 0) > l["line"] else "miss"


def summarize(G, by="story"):
    out = []
    for key, g in ([(k, G[G[by] == k]) for k in G[by].unique()] + [("all", G)]):
        if not len(g): continue
        se = float(np.sqrt((g.model_p * (1 - g.model_p)).sum()) / len(g))
        out.append(dict(story=key, n=int(len(g)), hits=int(g.hit.sum()), hit_rate=round(float(g.hit.mean()), 3), said=round(float(g.model_p.mean()), 3),
                        said_indep=round(float(g.indep_p.mean()), 3), z=round(float((g.hit.mean() - g.model_p.mean()) / se), 2) if se else None,
                        brier_joint=round(float(((g.model_p - g.hit) ** 2).mean()), 4), brier_indep=round(float(((g.indep_p - g.hit) ** 2).mean()), 4),
                        units=round(float((g.hit * g.fair - 1).sum()), 1) if "fair" in g else None))
    order = {s[0]: i for i, s in enumerate(STORIES)}
    return sorted(out, key=lambda r: order.get(r["story"], 99))


def grade(S=2026):
    A, on, done = actuals(S); sch = pd.read_parquet("data/sched.parquet").set_index("game_id")
    rows, legrows, shown = [], [], []
    for f in sorted(glob.glob("sgpgen_wk*.csv")):
        d = pd.read_csv(f)
        for r in d.itertuples():
            legs = json.loads(r.legs)
            res = [leg_result(l, r.game_id, A, on, sch) if r.game_id in done else "pending" for l in legs]
            st = "pending" if "pending" in res else "void" if "void" in res else "hit" if all(x == "hit" for x in res) else "miss"
            if r.shown: shown.append(dict(week=int(r.week), game_id=r.game_id, story=r.story, story_name=r.story_name, subject=r.subject,
                                          model_p=r.model_p, fair=r.fair, status=st, legs=[dict(t=l["text"], r=x) for l, x in zip(legs, res)]))
            for l, x in zip(legs, res):
                if x in ("hit", "miss"): legrows.append(dict(kind=l["kind"] if l["kind"] != "player" else l["stat"], p=l["p"], hit=x == "hit",
                                                             key=(r.game_id, l["text"])))
            if st in ("hit", "miss"): rows.append(dict(week=r.week, story=r.story, shown=r.shown, model_p=r.model_p, indep_p=r.indep_p, fair=r.fair, hit=st == "hit"))
    G = pd.DataFrame(rows); LG = pd.DataFrame(legrows)
    rep = {"asof": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET"),
           "stories": [dict(key=k, name=n, legs=len(s)) for k, n, _, s in STORIES],
           "all": summarize(G) if len(G) else [], "shown": summarize(G[G.shown == 1]) if len(G) else [],
           "legs": [], "combos": [c for c in shown if c["week"] >= (max([c["week"] for c in shown]) - 2 if shown else 0)]}
    bt = "data/sgp_backtest.json"
    try: rep["backtest"] = json.load(open(bt))
    except FileNotFoundError: rep["backtest"] = None
    if len(LG):
        LG = LG.drop_duplicates("key")
        for k, g in LG.groupby("kind"):
            rep["legs"].append(dict(kind=k, n=int(len(g)), hit_rate=round(float(g.hit.mean()), 3), said=round(float(g.p.mean()), 3)))
    json.dump(rep, open("sgp_report.json", "w"), indent=1)
    print("SGP combos graded:", len(G))
    for t in rep["shown"]: print(f"  shown {t['story']:<12} n={t['n']:>3} hit {t['hit_rate']:.0%} said {t['said']:.0%} z={t['z']}")


# ---------------------------------------------------------------- backtest on replays
def backtest(seasons, corr_file="data/joint_corr_to2023.json"):
    import parlay_joint as PJ
    PJ._C = json.load(open(corr_file))                                  # copula fitted before the tested seasons
    sched = pd.read_parquet("data/sched.parquet"); sch = sched.set_index("game_id")
    rows, legrows = [], []
    for S in seasons:
        A, on, done = actuals(S); RM, RT = residuals(S, sched)
        for f in sorted(glob.glob(f"replay/out/proj_{S}_wk*.json"), key=lambda x: int(re.findall(r"wk(\d+)", x)[0])):
            W = int(re.findall(r"wk(\d+)", f)[0]); D = json.load(open(f)); lf = f"replay/out/lines_{S}_wk{W}.json"
            try: L = json.load(open(lf))
            except FileNotFoundError: continue
            REC, CAR = season_counts(S, W); QS = D["quants"]
            for g in L["games"]:
                gid = g["game_id"]
                if gid not in done or gid not in sch.index: continue
                allc = build_game(g, D["players"], QS, REC, CAR, RM, RT, ph_of(g, L))
                shown = {id(c) for c in pick_shown(allc)}
                for c in allc:
                    res = [leg_result(l, gid, A, on, sch) for l in c["legs"]]
                    for l, x in zip(c["legs"], res):
                        if x != "void": legrows.append(dict(season=S, kind=l["kind"] if l["kind"] != "player" else l["stat"], p=l["p"], hit=x == "hit", key=(gid, l["text"])))
                    if "void" in res: continue
                    rows.append(dict(season=S, week=W, game_id=gid, story=c["story"], shown=int(id(c) in shown), model_p=c["model_p"], indep_p=c["indep_p"],
                                     fair=c["fair"], n_legs=len(c["legs"]), hit=all(x == "hit" for x in res)))
            print(S, W, len(rows), flush=True)
    G = pd.DataFrame(rows); LG = pd.DataFrame(legrows).drop_duplicates("key")
    G.to_pickle("replay/out/sgp_backtest.pkl"); LG.to_pickle("replay/out/sgp_backtest_legs.pkl")
    LG["b"] = pd.cut(LG.p, [.6, .7, .8, .9], include_lowest=True)
    out = {"seasons": seasons, "games": int(G.game_id.nunique()), "all": summarize(G), "shown": summarize(G[G.shown == 1]),
           "by_season": {str(S): summarize(G[G.season == S]) for S in seasons},
           "legs": [dict(kind=k, n=int(len(g)), hit_rate=round(float(g.hit.mean()), 3), said=round(float(g.p.mean()), 3)) for k, g in LG.groupby("kind")],
           "legs_by_band": [dict(kind=k, band=str(b), n=int(len(g)), hit_rate=round(float(g.hit.mean()), 3), said=round(float(g.p.mean()), 3))
                            for (k, b), g in LG.groupby(["kind", "b"], observed=True)]}
    json.dump(out, open("data/sgp_backtest.json", "w"), indent=1)
    for t in out["all"]: print(f"  {t['story']:<12} n={t['n']:>4} hit {t['hit_rate']:.1%} said {t['said']:.1%} indep {t['said_indep']:.1%} z={t['z']} units {t['units']}")
    for l in out["legs"]: print("  leg", l)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "grade": grade()
    elif a and a[0] == "backtest": backtest([int(x) for x in a[1:]] or [2024, 2025])
    else:
        for W in map(int, a): log(W)
