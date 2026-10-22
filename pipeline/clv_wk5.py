"""Closing-line check on the Week 5 paper slates: bet price (Thu/Fri) vs the Sunday-morning DK pull (close proxy) for props,
and vs nflverse closing spread/total for game bets."""
import re, glob, pandas as pd, numpy as np
rows = []
for f in glob.glob("odds_raw/wk5/*.txt") + glob.glob("odds_raw/wk5_ou/*.txt"):
    for ln in open(f):
        p = ln.strip().split("|")
        if len(p) == 5 and p[0] != "LAST_UPDATE": rows.append(dict(mkt=p[0], name=p[1], side=p[2], pt=float(p[3]) if p[3] else np.nan, price=float(p[4])))
close = pd.DataFrame(rows).drop_duplicates(["mkt", "name", "side", "pt"], keep="last")
norm = lambda s: re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower()).replace(".", "").replace("'", "").strip()
close["n"] = close.name.map(norm)
sch = pd.read_parquet("data/sched.parquet"); sch = sch[(sch.season == 2026) & (sch.week == 5)]
MK = [("reception yds", "player_reception_yds"), ("rec yds", "player_reception_yds"), ("receptions", "player_receptions"), (" rec", "player_receptions"),
      ("rush yds", "player_rush_yds"), ("rushing yds", "player_rush_yds"), ("pass yds", "player_pass_yds"), ("passing yds", "player_pass_yds"), ("pass att", "player_pass_attempts"), ("rush att", "player_rush_attempts")]
out = []
for f in ["bets_wk5_graded.csv", "bets_wk5_v2_graded.csv"]:
    d = pd.read_csv(f)
    for r in d.itertuples():
        o = dict(slate="v1" if "v2" not in f else "v2", cat=r.cat, bet=r.bet, price=r.price, pnl=r.pnl, news=getattr(r, "news", np.nan))
        if r.cat == "TD":
            nm = norm(r.bet.replace(" anytime TD", "")); c = close[(close.mkt == "player_anytime_td") & (close.n == nm)]
            if len(c): o.update(close_price=c.price.iloc[0], clv=r.price / c.price.iloc[0] - 1, beat=r.price > c.price.iloc[0], same=r.price == c.price.iloc[0])
        elif r.cat == "OU":
            m = re.match(r"(.+?) (Over|Under) ([\d.]+) (.+)", r.bet)
            if m:
                nm, side, pt, kind = norm(m[1]), m[2], float(m[3]), " " + m[4]
                mk = next((v for k, v in MK if k in kind), None)
                c = close[(close.mkt == mk) & (close.n == nm) & (close.side == side)]
                if len(c):
                    cp = c.pt.iloc[0]; pts = (cp - pt) if side == "Over" else (pt - cp)
                    o.update(close_pt=cp, close_price=c.price.iloc[0], clv_pts=pts, beat=(pts > 0) or (pts == 0 and r.price > c.price.iloc[0]), same=(pts == 0 and r.price == c.price.iloc[0]))
        elif r.cat in ("spread", "total", "prime under"):
            m = re.search(r"\((\w+)@(\w+)\)", r.bet) or re.search(r"(\w+) at (\w+)", r.bet)
            g = sch[(sch.away_team == m[1]) & (sch.home_team == m[2])]
            if len(g):
                g = g.iloc[0]
                if r.cat == "spread":
                    tm, pt = re.match(r"(\w+) ([+-][\d.]+)", r.bet).groups(); pt = float(pt)
                    cl = g.spread_line if tm == g.away_team else -g.spread_line    # team's own line, + = getting points
                    o.update(close_pt=cl, clv_pts=pt - cl)
                else:
                    side, pt = re.search(r"(Over|Under) ([\d.]+)", r.bet).groups(); pt = float(pt)
                    o.update(close_pt=g.total_line, clv_pts=(g.total_line - pt) if side == "Over" else (pt - g.total_line))
                o.update(beat=o["clv_pts"] > 0, same=o["clv_pts"] == 0)
        out.append(o)
x = pd.DataFrame(out); x.to_csv("clv_wk5.csv", index=False)
m = x.beat.notna()
print(f"checked {m.sum()} of {len(x)} bets")
for sl in ("v1", "v2"):
    y = x[(x.slate == sl) & m]
    print(f"\n== {sl}: beat close {int(y.beat.sum())}, same {int(y.same.sum())}, worse {int((~y.beat & ~y.same).sum())} of {len(y)}")
    print(y.groupby("cat").agg(n=("beat", "size"), beat=("beat", "sum"), same=("same", "sum"), td_clv=("clv", "mean"), pts=("clv_pts", "mean"), pnl=("pnl", "sum")).round(3).to_string())
    if sl == "v2": print(y.groupby("news").agg(n=("beat", "size"), beat=("beat", "sum"), same=("same", "sum"), td_clv=("clv", "mean"), pnl=("pnl", "sum")).round(3).to_string())
u = x[m].drop_duplicates("bet")
print(f"\nunique bets: beat {int(u.beat.sum())}, same {int(u.same.sum())}, worse {int((~u.beat & ~u.same).sum())} of {len(u)}")
print(u[u.cat != "TD"][["cat", "bet", "price", "close_pt", "clv_pts", "pnl"]].to_string(index=False))
t = u[u.cat == "TD"].sort_values("clv"); print(t[["bet", "price", "close_price", "clv", "pnl"]].round(3).to_string(index=False))
