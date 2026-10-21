"""Anytime TDs in the week's highest-total games: do they beat DK's price? (a) priced weeks; (b) 2016-2025 history: do teams in
top-5-total games score MORE TDs than their implied team total says (i.e. does scoring rise faster than the line), which is the
only way a 'high total' filter could beat prices that already include the total."""
import numpy as np, pandas as pd
exec(open("tests/t_td_band.py").read().split("b = a[")[0])
s = pd.read_parquet("data/sched.parquet"); s = s[s.game_type == "REG"]
s["trk"] = s.groupby(["season", "week"]).total_line.rank(ascending=False, method="first")
a = a.merge(s[["game_id", "total_line", "trk"]], on="game_id", how="left"); a["top5"] = a.trk <= 5
print("(a) priced ATD, weeks", sorted(a.week.unique()))
for lab, m in (("top-5 total games", a.top5), ("other games", ~a.top5)):
    for pl_, mm in (("all prices", m), ("price <= 2.5", m & (a.price <= 2.5))):
        b = a[mm]; print(f"  {lab:18s} {pl_:13s} n={len(b):3d} scored {b.hit.sum():3d}  DK implied {(1/b.price).sum():5.1f}  model {b.model_p.sum():5.1f}  flat 1u {b.pnl.sum():+.2f}")
# (b) history: offensive TDs vs implied team total
h = s[s.season.between(2016, 2025) & s.result.notna() & s.total_line.notna()].copy()
p = pd.concat([pd.read_parquet(f"data/pbp_{y}.parquet", columns=["game_id", "posteam", "touchdown", "td_team", "rush_touchdown", "pass_touchdown"]) for y in range(2016, 2026)])
otd = p[(p.rush_touchdown == 1) | (p.pass_touchdown == 1)].groupby(["game_id", "td_team"]).size()
rows = []
for r in h.itertuples():
    for tm, sgn in ((r.home_team, 1), (r.away_team, -1)):
        imp = r.total_line / 2 + sgn * r.spread_line / 2
        rows.append(dict(season=r.season, top5=r.trk <= 5, total=r.total_line, imp=imp, otd=otd.get((r.game_id, tm), 0)))
d = pd.DataFrame(rows)
print("\n(b) 2016-2025 offensive TDs per team-game vs implied team total")
d["tb"] = pd.cut(d.total, [0, 40, 43, 46, 49, 52, 70])
g = d.groupby("tb", observed=True).agg(n=("otd", "size"), imp_pts=("imp", "mean"), tds=("otd", "mean")); g["tds_per_10pts"] = g.tds / g.imp_pts * 10
print(g.round(3).to_string())
for lab, m in (("top-5 total games", d.top5), ("other games", ~d.top5)):
    x = d[m]; print(f"  {lab:18s} n={len(x)}  implied pts {x.imp.mean():.2f}  off TDs {x.otd.mean():.3f}  TDs per 10 implied pts {x.otd.mean()/x.imp.mean()*10:.3f}")
