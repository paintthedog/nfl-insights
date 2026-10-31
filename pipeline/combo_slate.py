"""Odds-free combination predictions (Oct 31, Clinton's request). The question is not "would this bet pay" but "when the model says
a combination of outcomes has an X% chance, does it happen X% of the time?" No sportsbook prices are used anywhere.

Every leg is chosen by the model alone: the round line where the model's chance of going over is closest to 70%
(receptions x.5; yards in 5-yard steps ending .5). Win legs use the site's line-anchored win chance (nflverse spread, free).
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
       python combo_slate.py grade   grade every combos_wk*.csv and write combos_report.json"""
import sys, json, re, glob
import numpy as np, pandas as pd
from parlay_joint import joint
from sgp_slate import anchored_home_win

TYPES = ["Combo: win + lead RB yards", "Combo: QB yards + WR1 catches", "Combo: top two pass catchers", "Combo: win + likeliest TD scorer",
         "Combo: steady pair across games", "Combo: steady trio across games", "Control: random pair across games"]
TARGET = 0.70
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
UNIT = {"rec": "catches", "rush_yds": "rush yds", "pass_yds": "pass yds"}


def season_counts(W):
    pbp = pd.read_parquet("data/pbp_2026.parquet", columns=["week", "receiver_player_id", "complete_pass", "rusher_player_id", "rush_attempt", "two_point_attempt"])
    pbp = pbp[(pbp.week < W) & pbp.two_point_attempt.ne(1)]
    return pbp.groupby("receiver_player_id").complete_pass.sum().to_dict(), pbp[pbp.rush_attempt.eq(1)].groupby("rusher_player_id").size().to_dict()


def build(W, now=None):
    D = json.load(open(f"site/data_wk{W}.json")); L = json.load(open(f"lines_2026_wk{W}.json")); QS = D["proj"]["quants"]
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
        cands = [k + .5 for k in range(0, 15)] if st == "rec" else [k * 5 + 4.5 for k in range(1, 80)]
        best = min(cands, key=lambda c: abs(p_over(q, c) - TARGET))
        pr = p_over(q, best)
        if not .55 <= pr <= .85: return None
        return dict(gid=gid, player=p["name"], team=p["team"], stat=st, line=best, p=round(float(pr), 4), label=label)

    def real(p):
        if (p.get("p_play") or 1) < 0.85: return False
        if p["pos"] == "QB": return True
        return REC.get(p.get("pid"), 0) >= 8 if p["pos"] in ("WR", "TE") else CAR.get(p.get("pid"), 0) >= 25 or REC.get(p.get("pid"), 0) >= 8

    P = D["proj"]["players"]; TD = {(t["name"], t["team"]): t for t in D.get("td", [])}
    SDM = L["sd_margin"]; rows = []; steady_legs = []; pool = []
    games = sorted([g for g in L["games"] if g["game_id"] in sw.index and kick(g["game_id"]) > now],
                   key=lambda g: (str(sw.loc[g["game_id"]].gameday), str(sw.loc[g["game_id"]].gametime), g["game_id"]))
    for g in games:
        gid = g["game_id"]; ph = anchored_home_win(g, SDM)
        fav, pw = (g["home"], ph) if ph >= .5 else (g["away"], 1 - ph)
        win = dict(gid=gid, player=fav, team=fav, stat="ml", line=None, p=round(float(pw), 4), label="WIN")
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
            for p in tp:                                   # every eligible leg goes into the pool for steady and control combos
                st = "pass_yds" if p["pos"] == "QB" else "rush_yds" if p["pos"] == "RB" else "rec"
                l = leg(p, st, gid, None)
                if l:
                    s = p.get("steady") or {}
                    pool.append(dict(l, steady=s.get("score"), flags=len(s.get("flags") or [])))
        tds = [t for t in D.get("td", []) if t["team"] in (g["home"], g["away"])]
        tds = [t for t in tds if any(p["name"] == t["name"] and p["team"] == t["team"] and real(p) for p in P)]
        if tds:
            t = max(tds, key=lambda t: t["p"])
            rows.append((TYPES[3], [gid], [win, dict(gid=gid, player=t["name"], team=t["team"], stat="atd", line=None, p=round(float(t["p"]), 4), label="TD")]))
        st = [l for l in pool if l["gid"] == gid and (l["steady"] or 0) >= 60 and l["flags"] == 0]
        if st: steady_legs.append(max(st, key=lambda l: (l["steady"], l["p"])))
    for i in range(0, len(steady_legs) - 1, 2): rows.append((TYPES[4], [l["gid"] for l in steady_legs[i:i + 2]], steady_legs[i:i + 2]))
    for i in range(0, len(steady_legs) - 2, 3): rows.append((TYPES[5], [l["gid"] for l in steady_legs[i:i + 3]], steady_legs[i:i + 3]))
    gids = sorted({l["gid"] for l in pool})
    for _ in range(8 if len(gids) >= 2 else 0):
        a, b = rng.choice(gids, 2, replace=False)
        la = [l for l in pool if l["gid"] == a]; lb = [l for l in pool if l["gid"] == b]
        rows.append((TYPES[6], [a, b], [la[rng.integers(len(la))], lb[rng.integers(len(lb))]]))

    stamp = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ"); out = []
    for t, gl, legs in rows:
        legs = [{k: v for k, v in l.items() if k not in ("steady", "flags")} for l in legs]
        ip = float(np.prod([l["p"] for l in legs]))
        same = len(set(gl)) == 1 and all(l["label"] for l in legs) and not any(l["stat"] == "atd" for l in legs)
        jp = joint([(l["label"], l["p"]) for l in legs]) if same else ip
        subj = " + ".join(f"{l['player']} wins" if l["stat"] == "ml" else f"{l['player']} scores a TD" if l["stat"] == "atd"
                          else f"{l['player']} {l['line']:g}+ {UNIT[l['stat']]}" for l in legs)
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

    rows, legrows = [], []
    for f in sorted(glob.glob("combos_wk*.csv")):
        d = pd.read_csv(f)
        for r in d.itertuples():
            legs = json.loads(r.legs); res = [leg_result(l) for l in legs]
            for l, x in zip(legs, res):
                if x in ("hit", "miss"): legrows.append(dict(week=r.week, stat=l["stat"], p=l["p"], hit=x == "hit", key=(r.week, l["gid"], l["player"], l["stat"], l["line"])))
            if "pending" in res: continue
            live = [x for x in res if x != "void"]
            if len(live) < len(res): continue                       # a scratched leg changes the question; leave the combo out
            rows.append(dict(week=r.week, type=r.type, model_p=r.model_p, indep_p=r.indep_p, hit=all(x == "hit" for x in live), n_legs=r.n_legs))
    G = pd.DataFrame(rows); LG = pd.DataFrame(legrows)
    rep = {"asof": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET"), "types": [], "legs": []}
    if len(G):
        for t in TYPES + ["All combos"]:
            g = G if t == "All combos" else G[G.type == t]
            if not len(g): continue
            se = float(np.sqrt((g.model_p * (1 - g.model_p)).sum()) / len(g))
            rep["types"].append(dict(type=t, n=int(len(g)), hits=int(g.hit.sum()), hit_rate=round(float(g.hit.mean()), 3),
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
    else: log(int(sys.argv[1]))
