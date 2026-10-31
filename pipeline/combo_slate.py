"""Odds-free combination predictions (Oct 31, Clinton's request). The question is not "would this bet pay" but "when the model says
a combination of outcomes has an X% chance, does it happen X% of the time?" No sportsbook prices are used anywhere.

Every leg is chosen by the model alone: the bettable round threshold ("50+ rush yds", "3+ catches", "225+ pass yds") where
the model's chance of reaching it is closest to 70% (Oct 31, Clinton: thresholds like a sportsbook's alt lines, not x.5 lines).
Receptions 1+..12+; rushing/receiving yards 10, 15, 20, 25, 30, 40 ... 100, 125, 150; passing yards 150 to 350 in 25s. Win legs use the site's line-anchored win chance (nflverse spread, free).
TD legs use the site's anytime-TD chance. Eligible players: projected to play (p_play 0.85+) and a real season record
(8+ catches for pass catchers, 25+ carries for backs).

Rules (mechanical, every game not yet kicked off; re-running replaces rows only for games that have not started):
  Combo: win + lead RB yards        projected winner wins + its lead back over a rushing-yards line
  Combo: QB yards + WR1 catches     each team: QB over a passing-yards line + his top wideout over a receptions line
  Combo: top two pass catchers      each team: its top two pass catchers (WR/TE by targets) over receptions lines
  Combo: win + likeliest TD scorer  projected winner wins + the game's likeliest TD scorer (either team) scores
  Combo: steady pair across games   the steadiest eligible player per game (Steady 60+, no role-change flag), games paired in kickoff order
  Combo: steady trio across games   same legs, three games at a time
  Control: random pair across games 8 random pairs of 70% legs from different games (the benchmark: these should hit ~49%)

model_p = joint chance (same-game legs through parlay_joint's historical correlations; cross-game legs multiplied).
indep_p = the legs multiplied as if unrelated. The report scores both, so it also tells whether the correlation model helps.
Usage: python combo_slate.py W       log week W (refresh_all.sh runs it for each week it builds)
       python combo_slate.py backfill 7 8   fill games never logged live from their last pre-kickoff archive (marked backfilled)
       python combo_slate.py grade   grade every combos_wk*.csv and write combos_report.json"""
import sys, json, re, glob
import numpy as np, pandas as pd
from parlay_joint import joint
from sgp_slate import anchored_home_win

TYPES = ["Combo: win + lead RB yards", "Combo: QB yards + WR1 catches", "Combo: top two pass catchers", "Combo: win + likeliest TD scorer",
         "Combo: steady pair across games", "Combo: steady trio across games", "Control: random pair across games"]
NARR = ["Combo: favorite's RB yards + underdog's WR1 catches", "Combo: QB yards + top TE catches", "Combo: lead RB yards + own QB yards",
        "Combo: shootout, both QBs' yards", "Combo: ground battle, both lead RBs' yards", "Combo: upset, underdog wins + its RB yards",
        "Combo: blowout, favorite wins + its RB yards + underdog QB yards", "Combo: WR1 duel, both top receivers' catches"]   # Oct 31: game-script stories
TYPES = TYPES[:4] + NARR + TYPES[4:]
TARGET = 0.70
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
UNIT = {"rec": "catches", "rush_yds": "rush yds", "pass_yds": "pass yds", "rec_yds": "rec yds"}


def season_counts(W):
    pbp = pd.read_parquet("data/pbp_2026.parquet", columns=["week", "receiver_player_id", "complete_pass", "rusher_player_id", "rush_attempt", "two_point_attempt"])
    pbp = pbp[(pbp.week < W) & pbp.two_point_attempt.ne(1)]
    return pbp.groupby("receiver_player_id").complete_pass.sum().to_dict(), pbp[pbp.rush_attempt.eq(1)].groupby("rusher_player_id").size().to_dict()


def build(W, now=None, D=None, only=None):
    """D: a pre-kickoff archive copy of the site data (backfill); only: restrict to these game_ids."""
    D = D or json.load(open(f"site/data_wk{W}.json")); L = D["lines"] if "games" in (D.get("lines") or {}) else json.load(open(f"lines_2026_wk{W}.json")); QS = D["proj"]["quants"]
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
    now = now or pd.Timestamp.now(tz="America/New_York")
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")
    REC, CAR = season_counts(W)
    rng = np.random.default_rng(3000 + W)

    def p_over(arr, line):
        arr = np.asarray(arr, float)
        if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
        for i in range(1, len(arr)):
            if line <= arr[i]:
                t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
        return max(0, (1 - QS[-1]) * .5)

    def leg(p, st, gid, label):
        q = (p.get("q") or {}).get(st)
        if not q: return None
        ks = list(range(1, 13)) if st == "rec" else list(range(150, 351, 25)) if st == "pass_yds" else [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
        k = min(ks, key=lambda k: abs(p_over(q, k - .5) - TARGET))    # "k+" = at least k = more than k - 0.5
        pr = p_over(q, k - .5)
        if not .55 <= pr <= .85: return None
        return dict(gid=gid, player=p["name"], team=p["team"], stat=st, line=k - .5, k=k, p=round(float(pr), 4), label=label)

    def real(p):
        if (p.get("p_play") or 1) < 0.85: return False
        if p["pos"] == "QB": return True
        return REC.get(p.get("pid"), 0) >= 8 if p["pos"] in ("WR", "TE") else CAR.get(p.get("pid"), 0) >= 25 or REC.get(p.get("pid"), 0) >= 8

    P = D["proj"]["players"]; TD = {(t["name"], t["team"]): t for t in D.get("td", [])}
    SDM = L["sd_margin"]; rows = []; steady_legs = []; pool = []
    games = sorted([g for g in L["games"] if g["game_id"] in sw.index and kick(g["game_id"]) > now and (only is None or g["game_id"] in only)],
                   key=lambda g: (str(sw.loc[g["game_id"]].gameday), str(sw.loc[g["game_id"]].gametime), g["game_id"]))
    for g in games:
        gid = g["game_id"]; ph = anchored_home_win(g, SDM)
        fav, pw = (g["home"], ph) if ph >= .5 else (g["away"], 1 - ph)
        win = dict(gid=gid, player=fav, team=fav, stat="ml", line=None, p=round(float(pw), 4), label="WIN")
        dog = g["away"] if fav == g["home"] else g["home"]; R = {}
        for team in (g["home"], g["away"]):
            tp = [p for p in P if p["team"] == team and real(p)]
            rbs = sorted([p for p in tp if p["pos"] == "RB" and CAR.get(p.get("pid"), 0) >= 25], key=lambda p: -(p["mean"].get("car") or 0))
            pc = sorted([p for p in tp if p["pos"] in ("WR", "TE")], key=lambda p: -(p["mean"].get("tgt") or 0))
            wr = [p for p in pc if p["pos"] == "WR"]; te = [p for p in pc if p["pos"] == "TE"]
            qbs = [p for p in tp if p["pos"] == "QB"]
            if team == fav and rbs:
                l = leg(rbs[0], "rush_yds", gid, "RB1 rush_yds")
                if l: rows.append((TYPES[0], [gid], [win, l]))
            if qbs and wr:
                qb = max(qbs, key=lambda p: p["mean"].get("pass_yds") or 0)
                lq, lw = leg(qb, "pass_yds", gid, "QB pass_yds"), leg(wr[0], "rec", gid, "WR1 rec")
                if lq and lw: rows.append((TYPES[1], [gid], [lq, lw]))
            two = []
            for p in pc[:2]:
                role = "WR1" if wr and p is wr[0] else "TE1" if te and p is te[0] else "WR2"
                l = leg(p, "rec", gid, f"{role} rec")
                if l: two.append(l)
            if len(two) == 2 and two[0]["label"] != two[1]["label"]: rows.append((TYPES[2], [gid], two))
            qb_ = max(qbs, key=lambda p: p["mean"].get("pass_yds") or 0) if qbs else None
            R[team] = dict(qb=qb_ and leg(qb_, "pass_yds", gid, "QB pass_yds"), rb=rbs and leg(rbs[0], "rush_yds", gid, "RB1 rush_yds"),
                           wr1=wr and leg(wr[0], "rec", gid, "WR1 rec"), te1=te and leg(te[0], "rec", gid, "TE1 rec"))
            for p in tp:                                   # every eligible leg goes into the pool for steady and control combos
                st = "pass_yds" if p["pos"] == "QB" else "rush_yds" if p["pos"] == "RB" else "rec"
                l = leg(p, st, gid, None)
                if l:
                    s = p.get("steady") or {}
                    pool.append(dict(l, steady=s.get("score"), flags=len(s.get("flags") or [])))
        # game-script stories. Labels are from one team's side ("OPP " = the other team) so parlay_joint applies the right correlations
        o = lambda l: dict(l, label="OPP " + l["label"])
        F, U = R.get(fav, {}), R.get(dog, {})
        dwin = dict(win, player=dog, team=dog, p=round(1 - win["p"], 4))
        if F.get("rb") and U.get("wr1"): rows.append((NARR[0], [gid], [F["rb"], o(U["wr1"])]))
        for T in (F, U):
            if T.get("qb") and T.get("te1"): rows.append((NARR[1], [gid], [T["qb"], T["te1"]]))
            if T.get("rb") and T.get("qb"): rows.append((NARR[2], [gid], [T["rb"], T["qb"]]))
        if F.get("qb") and U.get("qb"): rows.append((NARR[3], [gid], [F["qb"], o(U["qb"])]))
        if F.get("rb") and U.get("rb"): rows.append((NARR[4], [gid], [F["rb"], o(U["rb"])]))
        if U.get("rb"): rows.append((NARR[5], [gid], [dwin, U["rb"]]))
        if F.get("rb") and U.get("qb"): rows.append((NARR[6], [gid], [win, F["rb"], o(U["qb"])]))
        if F.get("wr1") and U.get("wr1"): rows.append((NARR[7], [gid], [F["wr1"], o(U["wr1"])]))
        tds = [t for t in D.get("td", []) if t["team"] in (g["home"], g["away"])]
        tds = [t for t in tds if any(p["name"] == t["name"] and p["team"] == t["team"] and real(p) for p in P)]
        if tds:
            t = max(tds, key=lambda t: t["p"])
            rows.append((TYPES[3], [gid], [win, dict(gid=gid, player=t["name"], team=t["team"], stat="atd", line=None, p=round(float(t["p"]), 4), label="TD")]))
        st = [l for l in pool if l["gid"] == gid and (l["steady"] or 0) >= 60 and l["flags"] == 0]
        if st: steady_legs.append(max(st, key=lambda l: (l["steady"], l["p"])))
    for i in range(0, len(steady_legs) - 1, 2): rows.append(("Combo: steady pair across games", [l["gid"] for l in steady_legs[i:i + 2]], steady_legs[i:i + 2]))
    for i in range(0, len(steady_legs) - 2, 3): rows.append(("Combo: steady trio across games", [l["gid"] for l in steady_legs[i:i + 3]], steady_legs[i:i + 3]))
    gids = sorted({l["gid"] for l in pool})
    for _ in range(8 if len(gids) >= 2 else 0):
        a, b = rng.choice(gids, 2, replace=False)
        la = [l for l in pool if l["gid"] == a]; lb = [l for l in pool if l["gid"] == b]
        rows.append(("Control: random pair across games", [a, b], [la[rng.integers(len(la))], lb[rng.integers(len(lb))]]))

    stamp = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ"); out = []
    for t, gl, legs in rows:
        legs = [{k: v for k, v in l.items() if k not in ("steady", "flags")} for l in legs]
        ip = float(np.prod([l["p"] for l in legs]))
        same = len(set(gl)) == 1 and all(l["label"] for l in legs) and not any(l["stat"] == "atd" for l in legs)
        jp = joint([(l["label"], l["p"]) for l in legs]) if same else ip
        subj = " + ".join(f"{l['player']} wins" if l["stat"] == "ml" else f"{l['player']} scores a TD" if l["stat"] == "atd"
                          else f"{l['player']} {l.get('k', l['line']):g}+ {UNIT[l['stat']]}" for l in legs)
        out.append(dict(week=W, type=t, games=",".join(dict.fromkeys(gl)), subject=subj, n_legs=len(legs), model_p=round(jp, 4),
                        indep_p=round(ip, 4), legs=json.dumps(legs), placed_at=stamp))
    return pd.DataFrame(out)


def log(W):
    new = build(W); f = f"combos_wk{W}.csv"
    try: old = pd.read_csv(f)
    except FileNotFoundError: old = pd.DataFrame()
    if len(old):
        sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
        now = pd.Timestamp.now(tz="America/New_York")
        started = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York") <= now
        old = old[old.games.map(lambda s: any(started(g) for g in s.split(",")))]   # keep whatever was logged before any of its games kicked off
        new = new[~new.games.map(lambda s: any(started(g) for g in s.split(",")))]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
    print(f"{f}: {len(new)} combos logged for games not yet started ({len(old)} kept)")


def backfill(W):
    """Oct 31: rebuild week W's combos after the fact from the site data archived BEFORE each game's kickoff (archive/data_wk{W}_*.json,
    saved by assemble_site.py on every publish). Same rules and numbers as a live run; rows carry backfilled=1 so the record says so.
    Only games whose combos were never logged live are filled; weeks before 7 have no archives (lost in the Oct 21 export)."""
    import glob as _g
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")
    arch = sorted(_g.glob(f"archive/data_wk{W}_*.json"))
    stamp = lambda f: pd.Timestamp(f[-18:-5].replace("_", " "), tz="America/New_York") if False else pd.to_datetime(f[-18:-5], format="%Y%m%d_%H%M").tz_localize("America/New_York")
    f = f"combos_wk{W}.csv"
    try: old = pd.read_csv(f)
    except FileNotFoundError: old = pd.DataFrame(columns=["games"])
    have = {g for s_ in old.games for g in str(s_).split(",")}
    pick = {}
    for gid in sw.index:
        if gid in have: continue
        pre = [a for a in arch if stamp(a) < kick(gid)]
        if pre: pick.setdefault(pre[-1], []).append(gid)
    new = []
    for a, gids in pick.items():
        d = build(W, now=stamp(a), D=json.load(open(a)), only=set(gids))
        if len(d): d["backfilled"] = 1; new.append(d)
    if new:
        old = pd.concat([old] + new, ignore_index=True); old.to_csv(f, index=False)
    print(f"{f}: backfilled {sum(len(d) for d in new)} combos for {sum(len(v) for v in pick.values())} games from {len(pick)} pre-kickoff archive(s)")


def grade():
    pbp = pd.read_parquet("data/pbp_2026.parquet"); pbp = pbp[pbp.two_point_attempt.ne(1)]
    done = set(pbp.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
    sch = pd.read_parquet("data/sched.parquet").set_index("game_id")
    pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"]); NM = {}
    for g, n in zip(pl.gsis_id, pl.display_name): NM.setdefault(norm(n), []).append(g)
    byg = {gid: d for gid, d in pbp[pbp.game_id.isin(done)].groupby("game_id")}

    def leg_result(l):
        gid = l["gid"]
        if gid not in done: return "pending"
        if l["stat"] == "ml":
            g = sch.loc[gid]; m = (g.home_score - g.away_score) * (1 if l["player"] == g.home_team else -1)
            return "void" if m == 0 else "hit" if m > 0 else "miss"
        gm = byg[gid]; ids = set(gm.passer_player_id.dropna()) | set(gm.rusher_player_id.dropna()) | set(gm.receiver_player_id.dropna()) | set(gm.td_player_id.dropna())
        c = [x for x in NM.get(norm(l["player"]), []) if x in ids]
        if not c: return "void"                                    # didn't touch the ball: the leg is dropped, like a scratched player
        p = c[0]
        if l["stat"] == "atd": return "hit" if (gm.touchdown.eq(1) & gm.td_player_id.eq(p)).any() else "miss"
        v = float(gm[gm.receiver_player_id.eq(p)].complete_pass.sum() if l["stat"] == "rec" else
                  gm[gm.passer_player_id.eq(p)].passing_yards.sum() if l["stat"] == "pass_yds" else gm[gm.rusher_player_id.eq(p)].rushing_yards.sum())
        return "hit" if v > l["line"] else "miss"

    rows, legrows, shown = [], [], []
    for f in sorted(glob.glob("combos_wk*.csv")):
        d = pd.read_csv(f)
        for r in d.itertuples():
            legs = json.loads(r.legs); res = [leg_result(l) for l in legs]
            for l, x in zip(legs, res):
                if x in ("hit", "miss"): legrows.append(dict(week=r.week, stat=l["stat"], p=l["p"], hit=x == "hit", key=(r.week, l["gid"], l["player"], l["stat"], l["line"])))
            st = "pending" if "pending" in res else "void" if "void" in res else "hit" if all(x == "hit" for x in res) else "miss"
            shown.append(dict(week=int(r.week), type=r.type, subject=r.subject, games=r.games, model_p=r.model_p, indep_p=r.indep_p, status=st,
                              legs=[dict(t=t, r=x) for t, x in zip(r.subject.split(" + "), res)]))
            if st in ("pending", "void"): continue                  # a scratched leg changes the question; leave the combo out
            rows.append(dict(week=r.week, type=r.type, model_p=r.model_p, indep_p=r.indep_p, hit=st == "hit", n_legs=r.n_legs))
    G = pd.DataFrame(rows); LG = pd.DataFrame(legrows)
    rep = {"asof": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET"), "types": [], "legs": [], "order": TYPES,
           "combos": [c for c in shown if c["week"] >= (max([c["week"] for c in shown]) - 2 if shown else 0)]}   # the site shows the last three weeks
    if len(G):
        for t in TYPES + ["All combos"]:
            g = G if t == "All combos" else G[G.type == t]
            if not len(g): continue
            se = float(np.sqrt((g.model_p * (1 - g.model_p)).sum()) / len(g))
            rep["types"].append(dict(type=t, legs=int(g.n_legs.iloc[0]) if t != "All combos" else None, n=int(len(g)), hits=int(g.hit.sum()), hit_rate=round(float(g.hit.mean()), 3),
                                     said=round(float(g.model_p.mean()), 3), said_indep=round(float(g.indep_p.mean()), 3),
                                     z=round(float((g.hit.mean() - g.model_p.mean()) / se), 2) if se else None,
                                     brier_joint=round(float(((g.model_p - g.hit) ** 2).mean()), 4),
                                     brier_indep=round(float(((g.indep_p - g.hit) ** 2).mean()), 4)))
    if len(LG):
        LG = LG.drop_duplicates("key")                              # a leg used in several combos counts once
        for s, g in LG.groupby("stat"):
            rep["legs"].append(dict(stat=s, n=int(len(g)), hit_rate=round(float(g.hit.mean()), 3), said=round(float(g.p.mean()), 3)))
    json.dump(rep, open("combos_report.json", "w"), indent=1)
    print("combos graded:", len(G), "| legs graded:", len(LG))
    for t in rep["types"]: print(f"  {t['type']:<36} n={t['n']:>3}  hit {t['hit_rate']:.0%}  said {t['said']:.0%} (indep {t['said_indep']:.0%})  z={t['z']}")
    for l in rep["legs"]: print(f"  leg {l['stat']:<9} n={l['n']:>3}  hit {l['hit_rate']:.0%}  said {l['said']:.0%}")


if __name__ == "__main__":
    if sys.argv[1] == "grade": grade()
    elif sys.argv[1] == "backfill": [backfill(int(w)) for w in sys.argv[2:]]
    else: log(int(sys.argv[1]))
