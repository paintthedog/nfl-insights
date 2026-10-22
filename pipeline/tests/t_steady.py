"""Does a player's steadiness (role and output) tell us anything the projection doesn't? (Oct 22)

Walk-forward rows (tests/_wf_proj.pkl, 2019-2026). For each player-game with 3+ earlier games that season:
  role_cv  = spread of his opportunity share across earlier games (sd / mean)
             WR/TE: share of team targets; RB: share of team carries + targets; QB: share of team dropbacks
  snap_sd  = spread of his snap share
  out_cv   = spread of his PPR points (sd / mean)
Target: the prop-leg question. Take the rung where the model gives the over about 70% (rec for RB/WR/TE,
rush_yds for RB, pass_yds for QB) and ask whether steady players clear it more often than the model says.
Prints hit rate minus model chance by tercile of each measure, fit 2019-2023, checked 2024-2026.
Run: python tests/t_steady.py   (add --all to include fringe players)
"""
import sys, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import proj_model as P
import steady as S

wf = pd.read_pickle("tests/_wf_proj.pkl")
wf = wf[wf.pos.isin(["QB", "RB", "WR", "TE"])].copy()
wf = S.add_features(wf)
tab = pickle.load(open("ratio_table.pkl", "rb"))["tab"]

LEGS = {"WR": ["rec"], "TE": ["rec"], "RB": ["rec", "rush_yds"], "QB": ["pass_yds"]}
rows = []
d = wf[(wf.n_prior >= 3)]
if "--all" not in sys.argv:   # same eligibility as the live rating (default); --all = every player, the first run
    d = d[(d.ppr_mean >= np.where(d.pos == "QB", 10, 6)) & (d.share_mean >= d.pos.map(S.MIN_SHARE))]
d = d[(d.pos != "QB") | d.qb_start]
for r in d.itertuples(index=False):
    for st in LEGS[r.pos]:
        mu = getattr(r, f"p_{st}")
        if not mu or mu <= (1.5 if st == "rec" else 15):
            continue
        qs = P.quantiles_for(tab, st, r.pos, mu)
        if not qs:
            continue
        line = qs[P.QUANTS.index(0.3)]
        line = np.floor(line) - 0.5 if st == "rec" else np.floor(line / 5) * 5 - 0.5   # book-style rung just under it
        if line <= 0:
            continue
        p = P.p_over(tab, st, r.pos, mu, line)
        rows.append(dict(season=r.season, pos=r.pos, st=st, p=p, hit=float(getattr(r, st) > line),
                         role_cv=r.role_cv, snap_sd=r.snap_sd, out_cv=r.out_cv, share=r.share_mean))
L = pd.DataFrame(rows).dropna()
L["gap"] = L.hit - L.p
# combined score: equal-weight average of the role and output percentiles (within pos/stat); higher = steadier
_r = L.groupby(["pos", "st"]).role_cv.rank(pct=True, ascending=False)
_o = L.groupby(["pos", "st"]).out_cv.rank(pct=True, ascending=False)
L["combo"] = -(_r + _o)      # negated so "low = steady" like the spread measures
print(f"{len(L)} legs; overall hit {L.hit.mean():.3f} vs model {L.p.mean():.3f}\n")

def by_tercile(df, col, label):
    out = []
    for (pos, st), g in df.groupby(["pos", "st"]):
        q = g[col].rank(pct=True)
        t = np.where(q <= 1/3, "steady", np.where(q <= 2/3, "middle", "jumpy"))
        for name in ["steady", "middle", "jumpy"]:
            h = g[t == name]
            se = np.sqrt((h.hit * (1 - h.hit)).mean() / max(len(h), 1))
            out.append(dict(pos=pos, stat=st, measure=label, group=name, n=len(h), hit=h.hit.mean(), model=h.p.mean(),
                            gap=h.gap.mean(), se=se))
    return pd.DataFrame(out)

pd.set_option("display.width", 200)
for period, sub in [("2019-2023 (fit)", L[L.season <= 2023]), ("2024-2026 (check)", L[L.season >= 2024])]:
    print("=" * 20, period, len(sub), "legs")
    res = pd.concat([by_tercile(sub, c, c) for c in ["role_cv", "out_cv", "combo"]])
    piv = res.pivot_table(index=["measure", "pos", "stat"], columns="group", values="gap").reindex(columns=["steady", "middle", "jumpy"])
    piv["steady-jumpy"] = piv.steady - piv.jumpy
    print((piv * 100).round(1).to_string()); print()
    # pooled, all legs: does each measure separate hit-minus-model?
    for c in ["role_cv", "snap_sd", "out_cv", "combo"]:
        q = sub.groupby(["pos", "st"])[c].rank(pct=True)
        s_, j_ = sub[q <= 1/3], sub[q > 2/3]
        diff = s_.gap.mean() - j_.gap.mean()
        se = np.sqrt(s_.gap.var() / len(s_) + j_.gap.var() / len(j_))
        print(f"  pooled {c:8s}: steady third {s_.gap.mean()*100:+.1f} pts vs model, jumpy third {j_.gap.mean()*100:+.1f}; "
              f"difference {diff*100:+.1f} (t={diff/se:.1f})")
    print()

# logistic: hit ~ logit(p) + measures (percentiles within pos/stat), fit period vs check period
import statsmodels.api as sm
for c in ["role_cv", "snap_sd", "out_cv"]:
    L[c + "_pc"] = L.groupby(["pos", "st"])[c].rank(pct=True) - 0.5
L["lp"] = np.log(L.p.clip(.01, .99) / (1 - L.p.clip(.01, .99)))
for period, sub in [("fit", L[L.season <= 2023]), ("check", L[L.season >= 2024])]:
    m = sm.Logit(sub.hit, sm.add_constant(sub[["lp", "role_cv_pc", "snap_sd_pc", "out_cv_pc"]])).fit(disp=0)
    print(period, "logit coefs:", m.params.round(3).to_dict(), "\n   t:", m.tvalues.round(1).to_dict())
