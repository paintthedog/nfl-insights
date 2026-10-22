import numpy as np, pandas as pd
exec(open("tests/t_td_totals.py").read().split("# (b) history")[0].split('print("(a)')[0])
a["imp"] = 1 / a.price
print("cumulative: every priced ATD at or below the price cap (weeks", sorted(a.week.unique()), ")")
print(f"{'cap':>5} | {'all games: n  hit%  DK%  units  ROI':38s} | top-5 totals: n  hit%  DK%  units  ROI")
for cap in [1.5, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 3.0, 3.5, 4.0]:
    s = []
    for m in (a.price <= cap, (a.price <= cap) & a.top5):
        b = a[m]; s.append(f"{len(b):4d} {b.hit.mean()*100:5.1f} {b.imp.mean()*100:5.1f} {b.pnl.sum():+6.2f} {b.pnl.mean()*100:+5.1f}%" if len(b) else "   -")
    print(f"{cap:5.2f} | {s[0]:38s} | {s[1]}")
print("\nby slice (not cumulative):")
a["band"] = pd.cut(a.price, [1, 1.8, 2.1, 2.5, 3.0, 4.0, 100])
for lab, m in (("all", a.price > 0), ("top-5", a.top5)):
    print(lab); print(a[m].groupby("band", observed=True).agg(n=("hit", "size"), hit=("hit", "mean"), dk=("imp", "mean"), units=("pnl", "sum")).round(3).to_string())
