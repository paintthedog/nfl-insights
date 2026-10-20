"""Grade pending paper bets in the published site's record from nflverse box scores, and patch the HTML.
Usage: python grade_pending.py <in index.html> <out html> <week> <stats_player_week.csv> <games.csv>"""
import csv, json, re, sys
src, dst, WEEK, statsf, gamesf = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
s = open(src, encoding="utf-8").read()
i = s.index("const ALL = ") + len("const ALL = "); d = 0
for k, c in enumerate(s[i:]):
    d += c == "{"; d -= c == "}"
    if d == 0 and c == "}": j = i + k + 1; break
A = json.loads(s[i:j])

norm = lambda n: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n.lower()))
st = {}
for r in csv.DictReader(open(statsf)):
    if r["season"] == "2026" and r["week"] == str(WEEK) and r["season_type"] == "REG":
        f = lambda c: float(r[c] or 0)
        st[norm(r["player_display_name"])] = dict(td=f("rushing_tds") + f("receiving_tds"), rush=f("rushing_yards"),
                                                 rec=f("receptions"), rec_yds=f("receiving_yards"), pass_yds=f("passing_yards"), team=r["team"])
win = {}
for g in csv.DictReader(open(gamesf)):
    if g["season"] == "2026" and g["week"] == str(WEEK) and g["result"] != "":
        h, a = int(g["home_score"]), int(g["away_score"])
        win[g["home_team"]] = h > a; win[g["away_team"]] = a > h

def leg(t):
    t = t.strip()
    m = re.fullmatch(r"(\w+) wins", t)
    if m: return win[m.group(1)]
    m = re.fullmatch(r"(.+?) (?:anytime )?TD", t)
    if m: return st.get(norm(m.group(1)), {}).get("td", 0) >= 1
    m = re.fullmatch(r"(.+?) ([\d.]+)\+ (rush yds|rec yds|rec|receptions)", t)
    if m:
        key = {"rush yds": "rush", "rec yds": "rec_yds", "rec": "rec", "receptions": "rec"}[m.group(3)]
        return st[norm(m.group(1))][key] >= float(m.group(2))
    m = re.fullmatch(r"(.+?) (Over|Under) ([\d.]+)", t)
    if m:
        v = st[norm(m.group(1))]["rec"]; line = float(m.group(3))
        return v > line if m.group(2) == "Over" else v < line
    raise ValueError("can't parse leg: " + t)

def grade(b):
    ok = all(leg(x) for x in b["bet"].split(" + "))
    b["result"] = "win" if ok else "loss"
    b["pnl"] = round(b["price"] - 1, 3) if ok else -1.0
    if b["bet"].endswith("anytime TD"):
        b["actual"] = st.get(norm(b["bet"][:-11]), {}).get("td", 0.0)
    else:
        b["actual"] = 1.0 if ok else 0.0
    return b

R = A["record"]; graded = {}
for b in R["paper"]:
    if b["week"] == WEEK and b["result"] == "pending":
        graded[(b["type"], b["bet"])] = grade(b)
for f in R["thisweek"]["families"]:
    for b in f["bets"]:
        if b["result"] == "pending":
            key = (b["type"], b["bet"])
            new = graded.get(key) or grade(dict(b))
            graded[key] = new
            b.update({k: new[k] for k in ("result", "pnl", "actual")})
            f["done"] += 1; f["w"] += new["result"] == "win"; f["units"] = round(f["units"] + new["pnl"], 2)
            f["expected"] = round(f["expected"] + (b.get("model_p") or 0), 1)
# strategy rows (running update)
for srow in R["strategy"]:
    gs = [b for (t, _), b in graded.items() if t == srow["type"]]
    for b in gs:
        n0 = srow["n"]; n1 = n0 + 1
        srow["model"] = round(((srow["model"] or 0) * n0 + (b.get("model_p") or 0)) / n1, 3)
        srow["breakeven"] = round(((srow["breakeven"] or 0) * n0 + 1 / b["price"]) / n1, 3)
        srow["n"] = n1; srow["w"] += b["result"] == "win"; srow["pending"] = max(0, srow["pending"] - 1)
        srow["units"] = round(srow["units"] + b["pnl"], 2); srow["hit"] = round(srow["w"] / n1, 3)
R["asof"] = R["asof"] + " · Week 6 Monday night graded Oct 20"
out = s[:i] + json.dumps(A, ensure_ascii=False, separators=(",", ":")) + s[j:]
open(dst, "w", encoding="utf-8").write(out)
w = sum(b["result"] == "win" for b in graded.values())
print(f"graded {len(graded)} bets: {w} wins, {len(graded)-w} losses")
for (t, bet), b in sorted(graded.items()): print(f"  {b['result']:4} {b['pnl']:+.2f}  {t} | {bet}")
