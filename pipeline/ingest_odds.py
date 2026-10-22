"""Add a saved DraftKings pull (odds_raw/wk<W>/*.txt, lines market|player|side|point|price) to props_log.csv / props_log_ou.csv.
Usage: python ingest_odds.py 5 "2026-10-08T15:36Z" """
import sys, glob, json, os, re, numpy as np, pandas as pd
W, pulled = int(sys.argv[1]), sys.argv[2]
KICK = json.load(open(f"odds_raw/wk{W}/kickoffs.json")) if os.path.exists(f"odds_raw/wk{W}/kickoffs.json") else {}
D = json.load(open(f"site/data_wk{W}.json")); QS = D["proj"]["quants"]
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
P = {norm(p["name"]): p for p in D["proj"]["players"]}; TD = {norm(r["name"]): r["p"] for r in D["td"]}
STAT = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds", "player_receptions": "rec"}
def p_over(arr, line):
    arr = np.asarray(arr, float)
    if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
    for i in range(1, len(arr)):
        if line <= arr[i]:
            t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
    return max(0, (1 - QS[-1]) * .5)
td, ou = [], {}
for f in sorted(x for x in glob.glob(f"odds_raw/wk{W}/*.txt") if not x.endswith("_x.txt")):   # _x files = extra markets (ingest_more.py)
    g = os.path.basename(f).split(".")[0].replace("_a", "").replace("_b", ""); away, home = g.split("_"); gid = f"2026_{W:02d}_{away}_{home}"
    for line in open(f):
        x = line.strip().split("|")
        if len(x) != 5 or x[0] == "LAST_UPDATE" or x[1].endswith("D/ST"): continue
        mk, pl, side, pt, pr = x; k = norm(pl)
        if mk == "player_anytime_td": td.append(dict(book="draftkings", player=pl, price=float(pr), model_p=TD.get(k), season=2026, week=W, game_id=gid, market=mk, pulled_at=pulled, kickoff=KICK.get(g)))
        else:
            r = ou.setdefault((mk, pl), dict(market=mk, book="draftkings", player=pl, point=float(pt), season=2026, week=W, game_id=gid, pulled_at=pulled, kickoff=KICK.get(g)))
            r["over" if side == "Over" else "under"] = float(pr)
ou = list(ou.values())
for r in ou:
    p = P.get(norm(r["player"])); s = STAT[r["market"]]; r["mkt_p_over"] = (1 / r["over"]) / (1 / r["over"] + 1 / r["under"])
    q = p and p.get("q", {}).get(s); r["model_mean"] = p["mean"].get(s) if p else None; r["model_p_over"] = p_over(q, r["point"]) if q else None
for fn, rows, cols in (("props_log.csv", td, ["book", "player", "price", "model_p", "season", "week", "game_id", "market", "pulled_at", "kickoff"]),
                       ("props_log_ou.csv", ou, ["market", "book", "player", "point", "over", "under", "model_mean", "model_p_over", "mkt_p_over", "season", "week", "game_id", "pulled_at", "kickoff"])):
    if not rows: continue
    new = pd.DataFrame(rows)[cols]; old = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=cols)
    pd.concat([old, new]).to_csv(fn, index=False)
print(f"week {W}: {len(td)} touchdown prices, {len(ou)} over/under lines; matched to a projection: TD {sum(r['model_p'] is not None for r in td)}, O/U {sum(r['model_mean'] is not None for r in ou)}")
miss = sorted({r["player"] for r in ou if r["model_mean"] is None}); print("O/U lines with no projected player:", miss)
