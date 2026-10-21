"""Joint probability for same-game parlays (Oct 17). Gaussian copula on historical 'how far above/below his projection' percentiles.
Roles are set from projections (knowable before kickoff): QB, WR1, WR2, TE1, RB1 for each team, plus the opponent's.
  python parlay_joint.py fit          -> data/joint_corr.json (correlations, fit on all walk-forward seasons)
  from parlay_joint import joint; joint([("QB pass_yds", .70), ("WR1 rec_yds", .65)])   -> P(both)
Leg labels: QB pass_yds | WR1/WR2/TE1 rec, rec_yds | RB1 rush_yds, rec ; prefix 'OPP ' for the other team; 'WIN' = this team wins.
The marginal probabilities come from outside (DK no-vig or the site's blended chance); the copula only supplies how they move together."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import norm, multivariate_normal

LEGS = [("QB", "pass_yds"), ("WR1", "rec"), ("WR1", "rec_yds"), ("WR2", "rec"), ("WR2", "rec_yds"), ("TE1", "rec"), ("TE1", "rec_yds"),
        ("RB1", "rush_yds"), ("RB1", "rec")]

def scores(o, sched):
    """per team-game: normal score of each role-stat vs its projection, plus team result"""
    rank = lambda df, pos, col, k: df[df.pos == pos].sort_values(col, ascending=False).groupby(["game_id", "posteam"]).head(k).groupby(["game_id", "posteam"]).tail(1)
    roles = {"QB": o[(o.pos == "QB") & o.qb_start.astype(bool)].drop_duplicates(["game_id", "posteam"]),
             "WR1": rank(o, "WR", "p_tgt", 1), "WR2": rank(o, "WR", "p_tgt", 2), "TE1": rank(o, "TE", "p_tgt", 1), "RB1": rank(o, "RB", "p_car", 1)}
    T = None
    for rl, st in LEGS:
        x = roles[rl][["game_id", "posteam", "season", st, f"p_{st}"]].copy()
        r = x[st] / x[f"p_{st}"].clip(lower=0.1)
        x[f"{rl} {st}"] = norm.ppf((r.rank(method="average") - 0.5) / len(r))          # percentile of 'actual vs projection' -> normal score
        x = x[["game_id", "posteam", "season", f"{rl} {st}"]]
        T = x if T is None else T.merge(x, on=["game_id", "posteam", "season"], how="inner")
    s = sched[["game_id", "home_team", "away_team", "result"]]
    T = T.merge(s, on="game_id"); home = T.posteam == T.home_team
    T["WIN"] = np.where(home, T.result, -T.result)                       # margin; normal score below
    T["WIN"] = norm.ppf((T.WIN.rank(method="average") - 0.5) / len(T))
    T["opp"] = np.where(home, T.away_team, T.home_team)
    cols = [f"{rl} {st}" for rl, st in LEGS] + ["WIN"]
    O = T[["game_id", "posteam"] + cols].rename(columns={**{c: "OPP " + c for c in cols}, "posteam": "opp"})
    return T.merge(O, on=["game_id", "opp"], how="inner"), cols

def fit(T, cols):
    allc = cols + ["OPP " + c for c in cols]
    C = T[allc].corr(method="spearman")
    C = 2 * np.sin(np.pi * C / 6)          # spearman -> pearson for normal scores
    return {"labels": allc, "corr": C.round(4).values.tolist(), "n": int(len(T))}

_C = None
def joint(legs, corr=None):
    """legs: [(label, prob_of_leg_hitting)] ; 'WIN' uses the team-wins probability. Returns P(all hit)."""
    global _C
    if corr is None:
        if _C is None: _C = json.load(open("data/joint_corr.json"))
        corr = _C
    lab = corr["labels"]; M = np.array(corr["corr"])
    idx = [lab.index(l) for l, _ in legs]; S = M[np.ix_(idx, idx)]
    th = np.array([norm.ppf(1 - p) for _, p in legs])     # leg hits when score > threshold
    # P(all z_i > th_i) = P(all -z_i < -th_i)
    return float(multivariate_normal(mean=np.zeros(len(legs)), cov=S, allow_singular=True).cdf(-th))

if __name__ == "__main__" and sys.argv[1:] == ["fit"]:
    o = pd.read_pickle("tests/_wf_proj.pkl"); sched = pd.read_parquet("data/sched.parquet")
    T, cols = scores(o, sched); out = fit(T, cols); json.dump(out, open("data/joint_corr.json", "w"))
    lab = out["labels"]; M = pd.DataFrame(out["corr"], index=lab, columns=lab)
    print(f"fit on {out['n']} team-games"); print(M.loc[cols, cols].round(2).to_string())
