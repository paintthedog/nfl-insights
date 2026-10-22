"""Grade paper_wk<W>.csv -> paper_wk<W>_graded.csv. Flat 1u per row (parlays included). Player who doesn't play = push; a parlay drops a void leg."""
import sys, json, re, numpy as np, pandas as pd
W = int(sys.argv[1])
d = pd.read_csv(f"paper_wk{W}.csv")
pbp = pd.read_parquet("data/pbp_2026.parquet"); pbp = pbp[(pbp.week == W) & pbp.two_point_attempt.ne(1)]
done = set(pbp.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
sch = pd.read_parquet("data/sched.parquet"); sch = sch[(sch.season == 2026) & (sch.week == W)].set_index("game_id")
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"])
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
NM = {}
for g, n in zip(pl.gsis_id, pl.display_name): NM.setdefault(norm(n), []).append(g)
def pid(n, gid):
    c = NM.get(norm(n), [])
    if len(c) == 1: return c[0]
    gm = pbp[pbp.game_id == gid]; ids = set(gm.passer_player_id.dropna()) | set(gm.rusher_player_id.dropna()) | set(gm.receiver_player_id.dropna())
    h = [x for x in c if x in ids]; return h[0] if h else (c[0] if c else None)
touched = set(pbp.passer_player_id.dropna()) | set(pbp.rusher_player_id.dropna()) | set(pbp.receiver_player_id.dropna()) | set(pbp.td_player_id.dropna())
tds = pbp[pbp.touchdown.eq(1) & pbp.td_player_id.notna()].groupby("td_player_id").size()
REC = pbp.groupby("receiver_player_id").complete_pass.sum(); RUSH = pbp.groupby("rusher_player_id").rushing_yards.sum()
INTS = pbp.groupby("passer_player_id").interception.sum()
PASS = pbp.groupby("passer_player_id").passing_yards.sum()
def ou(v, side, line): return "void" if v == line else ("win" if (v > line) == (side == "Over") else "loss")
def leg_result(l):
    if l["gid"] not in done: return "pending"
    if l["stat"] == "ml":                                   # same-game stacks: 'this team wins' leg (tie = void)
        g = sch.loc[l["gid"]]; mrg = (g.home_score - g.away_score) * (1 if l["player"] == g.home_team else -1)
        return "void" if mrg == 0 else "win" if mrg > 0 else "loss"
    if l["stat"] == "spread":                               # teaser leg (Oct 21): team margin + teased handicap
        g = sch.loc[l["gid"]]; mrg = (g.home_score - g.away_score) * (1 if l["player"] == g.home_team else -1)
        return "void" if mrg + l["line"] == 0 else "win" if mrg + l["line"] > 0 else "loss"
    p = pid(l["player"], l["gid"])
    if p not in touched: return "void"
    if l["stat"] == "atd": return "win" if tds.get(p, 0) > 0 else "loss"
    v = float(REC.get(p, 0) if l["stat"] == "rec" else PASS.get(p, 0) if l["stat"] == "pass_yds" else RUSH.get(p, 0)); return ou(v, "Over", l["line"])
def grade(r):
    m = r["market"]
    if m == "parlay":
        res = [leg_result(l) for l in json.loads(r["legs"])]
        if "pending" in res: return "pending", None, None
        if "loss" in res: return "loss", None, None
        live = [l for l, x in zip(json.loads(r["legs"]), res) if x == "win"]
        if not live: return "void", None, None
        return "win", None, float(np.prod([l["price"] for l in live]))
    gid = r["game_id"]
    scored = gid in sch.index and pd.notna(sch.loc[gid].home_score)     # final score arrives hours before play-by-play
    if gid not in done and not (scored and m in ("ml", "game_total", "team_total")): return "pending", None, None
    g = sch.loc[gid]
    if m in ("atd", "rec", "ints"):
        p = pid(r["subject"], gid)
        if p not in touched: return "void", None, None
        if m == "atd": n = int(tds.get(p, 0)); return ("win" if n > 0 else "loss"), n, None
        v = float(REC.get(p, 0) if m == "rec" else INTS.get(p, 0)); return ou(v, r["side"], r["line"]), v, None
    if m == "team_total":
        v = float(g.home_score if r["subject"] == g.home_team else g.away_score); return ou(v, r["side"], r["line"]), v, None
    if m == "game_total":
        v = float(g.home_score + g.away_score); return ou(v, r["side"], r["line"]), v, None
    if m == "ml":
        mrg = (g.home_score - g.away_score) * (1 if r["subject"] == g.home_team else -1)
        return ("void" if mrg == 0 else "win" if mrg > 0 else "loss"), float(mrg), None
    return "pending", None, None
res = d.apply(lambda r: pd.Series(grade(r), index=["result", "actual", "paid_price"]), axis=1)
d = pd.concat([d, res], axis=1)
pp = d.paid_price.fillna(d.price)
d["pnl"] = np.select([d.result == "win", d.result == "loss"], [pp - 1, -1.0], 0.0)
# ---- closing-line value (Oct 21), game markets only: nflverse's closing consensus line, free and automatic. Props would need a second
# paid odds pull, which the 500-credit budget can't cover. clv = price taken / closing price - 1 (moneylines); clv_pts = points of
# line value (totals and teaser legs). Positive = the bet got a better number than the market closed at.
dec = lambda a: None if pd.isna(a) else (1 + a / 100 if a > 0 else 1 + 100 / -a)
def clv(r):
    try:
        if r["market"] == "ml" and r["game_id"] in sch.index:
            g = sch.loc[r["game_id"]]; c = dec(g.home_moneyline if r["subject"] == g.home_team else g.away_moneyline)
            return (round(r["price"] / c - 1, 4), None) if c else (None, None)
        if r["market"] == "game_total" and r["game_id"] in sch.index and pd.notna(sch.loc[r["game_id"]].total_line):
            c = float(sch.loc[r["game_id"]].total_line); return None, (r["line"] - c if r["side"] == "Under" else c - r["line"])
        if r["market"] == "parlay" and isinstance(r["legs"], str) and '"spread"' in r["legs"]:
            tot = 0.0
            for l in json.loads(r["legs"]):
                if l["stat"] != "spread" or l["gid"] not in sch.index or pd.isna(sch.loc[l["gid"]].spread_line): return None, None
                g = sch.loc[l["gid"]]; close = -g.spread_line if l["player"] == g.home_team else g.spread_line   # team handicap at the close
                tot += (l["line"] - 6) - close
            return None, round(tot, 1)
    except Exception: pass
    return None, None
fin = d.result.isin(["win", "loss", "void"])
cv = d.apply(lambda r: pd.Series(clv(r) if r["result"] in ("win", "loss", "void") else (None, None), index=["clv", "clv_pts"]), axis=1)
d["clv"], d["clv_pts"] = cv.clv, cv.clv_pts
d.to_csv(f"paper_wk{W}_graded.csv", index=False)
g = d[d.result.isin(["win", "loss"])]
print(f"paper_wk{W}: graded {len(g)}, pending {(d.result=='pending').sum()}, void {(d.result=='void').sum()}")
