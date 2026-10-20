#!/usr/bin/env python3
"""Season-long projection accuracy tracker for NFL Insights.

  python tracker.py snap <path-to-site-index.html>   # save today's projections
  python tracker.py grade                             # grade every completed game, write out/

Snapshots: one JSON per run in snapshots/. For each player-game, grading uses the
latest snapshot taken BEFORE that game's kickoff (kickoffs from nflverse schedule).
Actuals: nflverse weekly player stats. Everything is rebuilt from scratch each grade.
"""
import csv, json, math, os, re, subprocess, sys, statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "snapshots")
OUT = os.path.join(HERE, "out")
DL = os.path.join(HERE, "dl")
ET = ZoneInfo("America/New_York")
SEASON = 2026

# model key -> (nflverse column, label, position groups it applies to)
STATS = {
    "pass_yds": ("passing_yards", "Passing yards", {"QB"}),
    "pass_td": ("passing_tds", "Passing TDs", {"QB"}),
    "ints": ("passing_interceptions", "Interceptions thrown", {"QB"}),
    "car": ("carries", "Carries", {"QB", "RB", "WR", "TE"}),
    "rush_yds": ("rushing_yards", "Rushing yards", {"QB", "RB", "WR", "TE"}),
    "tgt": ("targets", "Targets", {"RB", "WR", "TE"}),
    "rec": ("receptions", "Receptions", {"RB", "WR", "TE"}),
    "rec_yds": ("receiving_yards", "Receiving yards", {"RB", "WR", "TE"}),
    "fp_ppr": ("fantasy_points_ppr", "Fantasy points (PPR)", {"QB", "RB", "WR", "TE"}),
}
LINEMAP = {"rec": "rec", "rec_yds": "rec_yds", "rush_yds": "rush_yds", "pass_yds": "pass_yds"}


def extract_all(html_path):
    s = open(html_path, encoding="utf-8").read()
    i = s.index("const ALL = ") + len("const ALL = ")
    depth = 0
    for k, c in enumerate(s[i:]):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[i:i + k + 1])
    raise ValueError("could not brace-match ALL")


def cmd_snap(html_path):
    A = extract_all(html_path)
    now = datetime.now(timezone.utc)
    snap = {"taken_utc": now.isoformat(), "weeks": {}}
    for wk, d in A["data"].items():
        players = []
        for p in d.get("proj", {}).get("players", []):
            lines = {}
            for k, v in (p.get("lines") or {}).items():
                dk = v.get("draftkings") if isinstance(v, dict) else None
                if dk and dk.get("pt") is not None:
                    lines[k] = dk["pt"]
            players.append({
                "pid": p.get("pid"), "name": p.get("name"), "pos": p.get("pos"),
                "team": p.get("team"), "opp": p.get("opp"), "p_play": p.get("p_play"),
                "mean": {k: (p.get("mean") or {}).get(k) for k in STATS},
                "p_td": p.get("p_td"),
                "floor": ((p.get("var") or {}).get("ppr") or {}).get("floor"),
                "ceil": ((p.get("var") or {}).get("ppr") or {}).get("ceil"),
                "lines": lines,
            })
        games = [{"game_id": g.get("game_id"), "home": g.get("home"), "away": g.get("away"),
                  "m_margin": g.get("m_margin"), "m_total": g.get("m_total"),
                  "spread": g.get("spread"), "total": g.get("total")}
                 for g in (d.get("lines") or {}).get("games", [])]
        snap["weeks"][str(wk)] = {"built": d.get("built"), "players": players, "games": games}
    os.makedirs(SNAP, exist_ok=True)
    fn = os.path.join(SNAP, now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    json.dump(snap, open(fn, "w"))
    print("saved", fn, {w: len(v["players"]) for w, v in snap["weeks"].items()})


def fetch(url, dest):
    os.makedirs(DL, exist_ok=True)
    r = subprocess.run(["curl", "-sSL", "--fail", "-o", dest, url], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"download failed: {url}\n{r.stderr}")


def load_actuals():
    p = os.path.join(DL, f"stats_player_week_{SEASON}.csv")
    fetch(f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{SEASON}.csv", p)
    rows = {}
    for r in csv.DictReader(open(p)):
        if r["season_type"] != "REG":
            continue
        rows[(int(r["week"]), r["player_id"])] = r
    g = os.path.join(DL, "games.csv")
    fetch("https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv", g)
    games = {}
    for r in csv.DictReader(open(g)):
        if r["season"] != str(SEASON) or r["game_type"] != "REG":
            continue
        ko = datetime.strptime(f'{r["gameday"]} {r["gametime"]}', "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        games[r["game_id"]] = {**r, "kickoff": ko.astimezone(timezone.utc)}
    return rows, games


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cmd_grade():
    actuals, games = load_actuals()
    # team-week -> game
    tw = {}
    for gid, g in games.items():
        for t in (g["home_team"], g["away_team"]):
            tw[(int(g["week"]), t)] = g
    snaps = []
    for fn in sorted(os.listdir(SNAP)):
        if fn.endswith(".json"):
            s = json.load(open(os.path.join(SNAP, fn)))
            s["_t"] = datetime.fromisoformat(s["taken_utc"])
            snaps.append(s)
    # pick latest pre-kickoff projection per (week, pid) and per game
    best = {}
    best_game = {}
    for s in snaps:
        for wk, d in s["weeks"].items():
            wk = int(wk)
            for p in d["players"]:
                g = tw.get((wk, p["team"]))
                if not g or s["_t"] >= g["kickoff"]:
                    continue
                key = (wk, p["pid"])
                if key not in best or best[key][0] < s["_t"]:
                    best[key] = (s["_t"], p, g)
            for gm in d["games"]:
                g = games.get(gm["game_id"])
                if not g or s["_t"] >= g["kickoff"]:
                    continue
                if gm["game_id"] not in best_game or best_game[gm["game_id"]][0] < s["_t"]:
                    best_game[gm["game_id"]] = (s["_t"], gm, g)

    rows = []
    td_rows = []
    for (wk, pid), (t, p, g) in best.items():
        if g["result"] == "":
            continue  # not played yet
        a = actuals.get((wk, pid))
        if a is None:
            continue  # did not record a stat line (inactive / DNP) — not graded
        pos = p["pos"]
        for k, (col, label, poss) in STATS.items():
            if pos not in poss:
                continue
            pr, ac = num(p["mean"].get(k)), num(a.get(col))
            if pr is None or ac is None:
                continue
            line = p["lines"].get(LINEMAP.get(k, "~")) if LINEMAP.get(k) else None
            row = {"week": wk, "pid": pid, "name": p["name"], "pos": pos, "team": p["team"],
                   "opp": p["opp"], "stat": k, "label": label, "proj": round(pr, 2),
                   "actual": ac, "err": round(ac - pr, 2), "abs_err": round(abs(ac - pr), 2),
                   "dk_line": line, "snap_utc": t.isoformat()}
            if k == "fp_ppr":
                row["floor"], row["ceil"] = p["floor"], p["ceil"]
            rows.append(row)
        if p.get("p_td") is not None:
            tds = sum(num(a.get(c)) or 0 for c in ("rushing_tds", "receiving_tds"))
            td_rows.append({"week": wk, "name": p["name"], "pos": pos, "p": p["p_td"], "hit": 1 if tds > 0 else 0})

    game_rows = []
    for gid, (t, gm, g) in best_game.items():
        if g["result"] == "":
            continue
        margin = num(g["home_score"]) - num(g["away_score"])
        total = num(g["home_score"]) + num(g["away_score"])
        game_rows.append({"week": int(g["week"]), "game_id": gid,
                          "model_margin_err": abs(margin - gm["m_margin"]) if gm["m_margin"] is not None else None,
                          "dk_margin_err": abs(margin - gm["spread"]) if gm["spread"] is not None else None,
                          "model_total_err": abs(total - gm["m_total"]) if gm["m_total"] is not None else None,
                          "dk_total_err": abs(total - gm["total"]) if gm["total"] is not None else None})

    os.makedirs(OUT, exist_ok=True)
    rows.sort(key=lambda r: (r["week"], r["stat"], r["name"]))
    if rows:
        with open(os.path.join(OUT, "season_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list({k: 1 for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)
    summary = summarize(rows, td_rows, game_rows)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(render(summary))


def agg(rs):
    if not rs:
        return None
    errs = [r["err"] for r in rs]
    acts = [r["actual"] for r in rs]
    mae = st.mean(abs(e) for e in errs)
    mean_act = st.mean(acts)
    out = {"n": len(rs), "mae": round(mae, 2), "bias": round(st.mean(errs), 2),
           "mean_actual": round(mean_act, 2),
           # relative miss: MAE as a share of the average actual, so stats are comparable
           "rel_miss": round(mae / mean_act, 3) if mean_act > 0 else None}
    lined = [r for r in rs if r.get("dk_line") is not None]
    if lined:
        out["n_line"] = len(lined)
        out["mae_model_on_lined"] = round(st.mean(abs(r["err"]) for r in lined), 2)
        out["mae_dk"] = round(st.mean(abs(r["actual"] - r["dk_line"]) for r in lined), 2)
    return out


def summarize(rows, td_rows, game_rows):
    by = lambda f: defaultdict(list, {})
    s = {"generated_et": datetime.now(ET).strftime("%a %b %d %I:%M %p ET"),
         "weeks": sorted({r["week"] for r in rows}), "by_stat": {}, "by_pos_stat": {},
         "by_week_stat": {}, "players": {}, "range": None, "td": None, "games": None}
    g = defaultdict(list)
    for r in rows:
        g[r["stat"]].append(r)
    for k, rs in g.items():
        s["by_stat"][k] = {"label": STATS[k][1], **agg(rs)}
    g = defaultdict(list)
    for r in rows:
        g[(r["pos"], r["stat"])].append(r)
    for (pos, k), rs in g.items():
        s["by_pos_stat"][f"{pos}|{k}"] = agg(rs)
    g = defaultdict(list)
    for r in rows:
        g[(r["week"], r["stat"])].append(r)
    for (wk, k), rs in g.items():
        s["by_week_stat"][f"{wk}|{k}"] = agg(rs)
    # players: fantasy-point accuracy, min 3 graded games
    g = defaultdict(list)
    for r in rows:
        if r["stat"] == "fp_ppr":
            g[(r["name"], r["pos"], r["team"])].append(r)
    pl = []
    for (n, pos, team), rs in g.items():
        if len(rs) >= 3:
            a = agg(rs)
            pl.append({"name": n, "pos": pos, "team": team, **a})
    pl.sort(key=lambda x: x["mae"])
    s["players"] = {"most_reliable": pl[:15], "least_reliable": pl[-15:][::-1], "n_players": len(pl)}
    fp = [r for r in rows if r["stat"] == "fp_ppr" and r.get("floor") is not None]
    if fp:
        s["range"] = {"n": len(fp),
                      "below_floor": round(sum(r["actual"] < r["floor"] for r in fp) / len(fp), 3),
                      "above_ceil": round(sum(r["actual"] > r["ceil"] for r in fp) / len(fp), 3)}
    if td_rows:
        buckets = []
        for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]:
            b = [r for r in td_rows if lo <= r["p"] < hi]
            if b:
                buckets.append({"lo": lo, "hi": hi, "n": len(b), "said": round(st.mean(r["p"] for r in b), 3),
                                "hap": round(st.mean(r["hit"] for r in b), 3)})
        s["td"] = {"n": len(td_rows), "brier": round(st.mean((r["p"] - r["hit"]) ** 2 for r in td_rows), 4),
                   "buckets": buckets}
    if game_rows:
        def m(k):
            v = [r[k] for r in game_rows if r[k] is not None]
            return round(st.mean(v), 2) if v else None
        s["games"] = {"n": len(game_rows), "margin_model": m("model_margin_err"), "margin_dk": m("dk_margin_err"),
                      "total_model": m("model_total_err"), "total_dk": m("dk_total_err")}
    return s


def render(s):
    L = [f"Season projection accuracy — weeks {s['weeks']} (as of {s['generated_et']})", ""]
    L.append("By stat, closest to reality first (relative miss = average miss / average actual):")
    for k, v in sorted(s["by_stat"].items(), key=lambda kv: kv[1]["rel_miss"] or 9):
        dk = f"  vs DK on {v['n_line']} lines: model {v['mae_model_on_lined']} / DK {v['mae_dk']}" if v.get("n_line") else ""
        L.append(f"  {v['label']:<22} n={v['n']:<4} miss {v['mae']:<6} bias {v['bias']:+} rel {v['rel_miss']}{dk}")
    if s["range"]:
        r = s["range"]; L.append(f"\nPPR floor/ceiling: {r['below_floor']:.0%} below floor, {r['above_ceil']:.0%} above ceiling (n={r['n']})")
    if s["td"]:
        L.append(f"\nAnytime TD: Brier {s['td']['brier']} on {s['td']['n']}")
        for b in s["td"]["buckets"]:
            L.append(f"  {b['lo']:.0%}-{min(b['hi'],1):.0%}: said {b['said']:.0%}, happened {b['hap']:.0%} (n={b['n']})")
    if s["games"]:
        g = s["games"]; L.append(f"\nGames (n={g['n']}): margin miss model {g['margin_model']} / DK {g['margin_dk']}; total miss model {g['total_model']} / DK {g['total_dk']}")
    p = s["players"]
    if p["n_players"]:
        L.append(f"\nMost reliable PPR projections (3+ games, {p['n_players']} players):")
        for x in p["most_reliable"][:8]:
            L.append(f"  {x['name']} ({x['pos']} {x['team']}) miss {x['mae']} over {x['n']}")
        L.append("Least reliable:")
        for x in p["least_reliable"][:8]:
            L.append(f"  {x['name']} ({x['pos']} {x['team']}) miss {x['mae']} over {x['n']}, bias {x['bias']:+}")
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "snap":
        cmd_snap(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "grade":
        cmd_grade()
    else:
        print(__doc__)
