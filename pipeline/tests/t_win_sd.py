"""Nov 14 (Clinton: underdog win chances look too high). Does a closing spread turned into a win chance with the margin's sd (13)
match how often favorites win? Walk-forward: the sd is fit on earlier seasons only. Usage: python tests/t_win_sd.py (from pipeline/)"""
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
s = pd.read_parquet("data/sched.parquet"); s = s[(s.game_type == "REG") & s.result.notna() & s.spread_line.notna() & (s.season >= 2010) & (s.result != 0)]
y = (s.result > 0).astype(int).values; sp = s.spread_line.values
ll = lambda sd, m: -np.mean(np.log(np.clip(np.where(y[m] == 1, norm.cdf(sp[m] / sd), 1 - norm.cdf(sp[m] / sd)), 1e-9, 1)))
rows = []; d = []
for S in range(2016, 2027):
    tr = (s.season < S).values; te = (s.season == S).values
    sd = minimize_scalar(lambda x: ll(x, tr), bounds=(8, 18), method="bounded").x
    for lab, k in (("13", 13.0), ("fit", sd)):
        p = norm.cdf(np.abs(sp[te]) / k); w = (np.sign(sp[te]) == np.sign(s.result.values[te])).astype(float); ok = sp[te] != 0
        rows += [dict(season=S, sd=lab, p=a, w=b) for a, b in zip(p[ok], w[ok])]
    d.append((norm.cdf(sp[te] / 13) - y[te]) ** 2 - (norm.cdf(sp[te] / sd) - y[te]) ** 2); print(S, "fit sd", round(sd, 2), "games", int(te.sum()))
R = pd.DataFrame(rows); d = np.concatenate(d)
for lab in ("13", "fit"):
    x = R[R.sd == lab]; x = x.assign(b=pd.cut(x.p, [.5, .6, .7, .8, 1]))
    print(f"\nfavorite's win chance, sd {lab}: said {x.p.mean():.3f}, won {x.w.mean():.3f}"); print(x.groupby("b", observed=True).agg(n=("w", "size"), said=("p", "mean"), won=("w", "mean")).round(3).to_string())
print(f"\nBrier gain {d.mean():.5f} per game (z {d.mean() / (d.std() / np.sqrt(len(d))):.2f}, {len(d)} games)")
