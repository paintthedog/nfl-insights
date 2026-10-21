import sys, numpy as np, pandas as pd, statsmodels.api as sm; sys.path.insert(0, ".")
import lines_model as L
s = pd.read_parquet("data/sched.parquet")
s = s[(s.game_type == "REG") & s.result.notna() & s.spread_line.notna() & s.total_line.notna() & (s.season <= 2026)].copy()
s["tot"] = s.home_score + s.away_score
print(f"MARKET, {s.season.min()}-{s.season.max()} regular season, n={len(s)}")
# favorites by spread size (spread_line = home favored by)
s["fav_margin"] = np.where(s.spread_line > 0, s.result, -s.result); s["sz"] = s.spread_line.abs()
s = s[s.sz > 0]
s["fcov"] = np.sign(s.fav_margin - s.sz)
s["bucket"] = pd.cut(s.sz, [0, 3, 6.5, 9.5, 13.5, 30], labels=["1-3", "3.5-6.5", "7-9.5", "10-13.5", "14+"])
t = s[s.fcov != 0].groupby("bucket", observed=True).agg(n=("fcov", "size"), fav_cover=("fcov", lambda x: (x > 0).mean()))
t["z"] = (t.fav_cover - .5) / np.sqrt(.25 / t.n)
print("\nFavorite covers, by spread size:\n" + t.round(3).to_string())
s["ou"] = np.sign(s.tot - s.total_line)
s["tb"] = pd.cut(s.total_line, [0, 38, 41.5, 44.5, 47.5, 50.5, 70], labels=["<=38", "38.5-41.5", "42-44.5", "45-47.5", "48-50.5", "51+"])
t2 = s[s.ou != 0].groupby("tb", observed=True).agg(n=("ou", "size"), over=("ou", lambda x: (x > 0).mean()), avg_line=("total_line", "mean"), avg_pts=("tot", "mean"))
t2["z"] = (t2.over - .5) / np.sqrt(.25 / t2.n)
print("\nOvers, by total size:\n" + t2.round(3).to_string())
# MODEL: walk-forward game model calibration by its own number
pbp, sched = L.load(); g = L.build(pbp, sched); g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)]
o = pd.concat([L.predict(g[g.season == y].copy(), *L.fit(g[g.season < y])) for y in range(2019, 2027)])
print(f"\nMODEL (walk-forward 2019-2026, n={len(o)})")
for nm, a, m in [("total", o.total, o.m_total), ("home margin", o.result, o.m_margin)]:
    sl = sm.OLS(a, sm.add_constant(m)).fit().params.iloc[1]
    q = pd.qcut(m, 5, labels=False)
    tt = pd.DataFrame(dict(model=m, actual=a, q=q)).groupby("q").mean().round(1)
    print(f"{nm}: slope {sl:.2f} (1.00 = calibrated)\n  by model quintile: " + " | ".join(f"model {r.model} -> actual {r.actual}" for r in tt.itertuples()))
