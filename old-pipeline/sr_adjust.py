"""Success-rate + pace adjustment layer for player projections (added Oct 16, tests/t_bundle.py).
Features, all from play-by-play, all from PRIOR games only:
  tgt_sr   receiver's success rate on his targets (EPA > 0), shrunk to league average (K=40 targets)
  qb_sr    the team's quarterback's success rate on dropbacks (K=150), chance-weighted if the starter is uncertain
  pace     team's seconds between snaps in neutral situations (and the opponent's)
Each (position, stat) gets a Poisson GLM multiplier on top of the calibrated projection: p * exp(sum (x - mean) * beta).
Backtest 2022-2026: WR fantasy points -0.34% error (better in 5/5 seasons), QB passing yards -0.4%, targets -0.6 to -0.75%.
Small and real. QB aggressiveness (NGS feed) was left out on purpose: separate feed, separate staleness risk."""
import glob
import numpy as np, pandas as pd, statsmodels.api as sm

B = {
    ("QB", "att"): ["pace", "opp_pace"], ("QB", "cmp"): ["pace", "opp_pace", "qb_sr"], ("QB", "pass_yds"): ["qb_sr", "pace", "opp_pace"],
    ("QB", "pass_1d"): ["qb_sr", "pace"],
    ("WR", "tgt"): ["tgt_sr", "pace"], ("WR", "rec"): ["tgt_sr", "pace", "qb_sr"], ("WR", "rec_yds"): ["tgt_sr", "qb_sr"], ("WR", "rec_1d"): ["tgt_sr", "qb_sr"],
    ("TE", "tgt"): ["tgt_sr", "pace"], ("TE", "rec"): ["tgt_sr", "pace", "qb_sr"], ("TE", "rec_yds"): ["tgt_sr", "qb_sr"], ("TE", "rec_1d"): ["tgt_sr", "qb_sr"],
    ("RB", "car"): ["pace", "opp_pace"], ("RB", "tgt"): ["tgt_sr", "pace"], ("RB", "rec"): ["tgt_sr", "pace"], ("RB", "rec_yds"): ["tgt_sr", "qb_sr"],
}
FEATS = ["tgt_sr", "qb_sr", "pace", "opp_pace"]
MAX_MULT = (0.85, 1.15)       # guard rail: never move a stat more than 15% either way


def _prior(df, key, num, den, K, lg, gd, decay):
    """value going INTO each game row, plus the current value (after the last game) per key for upcoming games"""
    df = df.assign(gameday=df.game_id.map(gd)).sort_values([key, "gameday"]); out, now = [], {}
    for k, x in df.groupby(key):
        a = b = 0.0
        for r in x.itertuples():
            out.append((r.game_id, k, (a + K * lg) / (b + K))); a = a * decay + getattr(r, num); b = b * decay + getattr(r, den)
        now[k] = (a + K * lg) / (b + K)
    return pd.DataFrame(out, columns=["game_id", key, "v"]), now


def features(sched):
    cols = ["game_id", "season_type", "posteam", "play_type", "epa", "receiver_player_id", "passer_player_id", "qb_dropback",
            "two_point_attempt", "wp", "half_seconds_remaining", "game_seconds_remaining", "drive"]
    pb = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
    pb = pb[(pb.season_type == "REG") & pb.play_type.isin(["pass", "run"]) & (pb.two_point_attempt != 1) & pb.posteam.notna()].copy()
    gd = pd.to_datetime(sched.set_index("game_id").gameday)
    e = pb[pb.epa.notna()].copy(); e["s"] = (e.epa > 0).astype(float); e["n"] = 1.0
    c = e[e.play_type == "pass"].dropna(subset=["receiver_player_id"]).groupby(["game_id", "receiver_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"receiver_player_id": "pid"})
    q = e[e.qb_dropback == 1].dropna(subset=["passer_player_id"]).groupby(["game_id", "passer_player_id"])[["s", "n"]].sum().reset_index().rename(columns={"passer_player_id": "pid"})
    pb = pb.sort_values(["game_id", "drive", "game_seconds_remaining"], ascending=[True, True, False])
    pb["dt"] = pb.groupby(["game_id", "drive"]).game_seconds_remaining.diff(-1)
    nz = pb[pb.wp.between(.2, .8) & (pb.half_seconds_remaining > 120) & pb.dt.between(1, 50)]
    pc = nz.groupby(["game_id", "posteam"]).agg(ds=("dt", "sum"), dn=("dt", "size")).reset_index()
    return dict(tgt_sr=_prior(c, "pid", "s", "n", 40, c.s.sum() / c.n.sum(), gd, 0.95),
                qb_sr=_prior(q, "pid", "s", "n", 150, q.s.sum() / q.n.sum(), gd, 0.95),
                pace=_prior(pc, "posteam", "ds", "dn", 60, nz.dt.mean(), gd, 0.9))


def attach(d, F, pred=False):
    """add FEATS to player-game rows d (needs game_id, pid, pos, posteam, opp; p_play for upcoming QBs)"""
    d = d.copy()
    for nm, key, col in (("tgt_sr", "pid", "pid"), ("pace", "posteam", "posteam")):
        tab, now = F[nm]
        if pred: d[nm] = d[col].map(now)
        else: d = d.merge(tab.rename(columns={"v": nm, key: col}), on=["game_id", col], how="left")
    tab, now = F["pace"]
    if pred: d["opp_pace"] = d.opp.map(now)
    else: d = d.merge(tab.rename(columns={"v": "opp_pace", "posteam": "opp"}), on=["game_id", "opp"], how="left")
    tab, now = F["qb_sr"]
    q = d[d.pos == "QB"][["game_id", "posteam", "pid"] + (["p_play"] if pred and "p_play" in d else [])].copy()
    if pred:
        q["v"] = q.pid.map(now); w = q.get("p_play", pd.Series(1.0, index=q.index)).fillna(1.0).clip(lower=0.01)
        qs = (q.v * w).groupby(q.posteam).sum() / w.groupby(q.posteam).sum()
        d["qb_sr"] = d.posteam.map(qs)
    else:
        q = q.merge(tab.rename(columns={"v": "qb_sr"}), on=["game_id", "pid"], how="left").drop_duplicates(["game_id", "posteam"])
        d = d.merge(q[["game_id", "posteam", "qb_sr"]], on=["game_id", "posteam"], how="left")
    return d


def fit(hist):
    """hist: calibrated walk-forward-style projections with actuals and FEATS attached. Returns {(pos, stat): (mu, beta)}"""
    M = {}
    for (pos, st), X in B.items():
        z = hist[(hist.pos == pos) & (hist[f"p_{st}"] > 0.05)].dropna(subset=X + [st])
        if len(z) < 500: continue
        mu = z[X].mean(); m = sm.GLM(z[st], sm.add_constant(z[X] - mu), family=sm.families.Poisson(), offset=np.log(z[f"p_{st}"])).fit()
        M[(pos, st)] = (mu, m.params[X])
    return M


def apply(f, M):
    f = f.copy(); med = {c: f[c].median() for c in FEATS}
    for (pos, st), (mu, beta) in M.items():
        m = f.pos == pos
        X = f.loc[m, list(mu.index)].fillna(med)
        mult = np.exp(((X - mu) * beta).sum(axis=1)).clip(*MAX_MULT)
        f.loc[m, f"p_{st}"] = f.loc[m, f"p_{st}"] * mult
    return f
