"""Steadiness rating (Oct 22): how much a player's role and his output move from game to game this season.

Two parts, each 0-100 and ranked within position among players with 3+ games this season:
  Role   - spread of his opportunity share game to game (WR/TE: share of team targets; RB: share of team carries
           plus targets; QB: share of team dropbacks).
  Output - spread of his PPR points game to game (sd / mean).
100 = the steadiest at his position, 0 = the jumpiest. The overall score blends the two by the weights that
tests/t_steady.py found predictive (see WEIGHTS). It is a label and a filter: it never changes a projection.
Role-change flags say when the past no longer describes the role (teammate out, traded, back from injury, recent shift).
"""
import numpy as np
import pandas as pd

WINDOW = 8        # most recent games this season
MIN_GAMES = 3

# Overall = equal-weight blend of the role and output percentiles. tests/t_steady.py (Oct 22), prop rungs the model gave
# ~70-85%: all players, steadiest third vs jumpiest third cleared +2.0 pts more vs the model in the 2019-23 fit (t=2.8) and
# 2024-26 check (t=2.0); but among eligible players (real role, MIN_SHARE + 6 PPR) only +0.7 (t=0.9) and +1.9 (t=1.7).
# Role spread and output spread traded places between periods, so neither gets more weight. Snap-share spread added nothing.
# Conclusion: a description and a filter, not an edge.
WEIGHTS = dict(role=0.5, output=0.5)
REC = {"WR", "TE"}
MIN_SHARE = {"QB": 0.5, "RB": 0.15, "WR": 0.10, "TE": 0.10}   # eligible to be rated (plus 6+ PPR a game, QB 10+)


def opp_share(df):
    """Per-game opportunity share by position."""
    z = lambda c: df[c].fillna(0) if c in df else 0
    rb = (z("car") + z("tgt")) / (z("tm_car") + z("tm_tgt")).replace(0, np.nan)
    rc = z("tgt") / z("tm_tgt").replace(0, np.nan)
    qb = (z("att") + z("sacks")) / (z("tm_att") + z("tm_sacks")).replace(0, np.nan)
    return np.where(df.pos == "RB", rb, np.where(df.pos == "QB", qb, rc))


def ppr(df):
    z = lambda c: df[c].fillna(0) if c in df else 0
    return (z("pass_yds") / 25 + 4 * z("pass_td") - 2 * z("ints") + z("rush_yds") / 10 + 6 * z("rush_td")
            + z("rec_yds") / 10 + 6 * z("rec_td") + z("rec"))


def add_features(df, include_current=False):
    """Adds share, ppr_act and the rolling spread measures over each player's EARLIER games that season
    (include_current=True: over games up to and including this row, for the live rating)."""
    df = df.sort_values(["pid", "season", "gameday"]).copy()
    df["share"] = opp_share(df)
    df["ppr_act"] = ppr(df)
    sh = 0 if include_current else 1
    def roll(col, fn):
        return df.groupby(["pid", "season"])[col].transform(
            lambda s: getattr(s.shift(sh).rolling(WINDOW, min_periods=2), fn)())
    df["n_prior"] = df.groupby(["pid", "season"]).cumcount() + (1 if include_current else 0)
    df["share_mean"] = roll("share", "mean")
    df["role_cv"] = roll("share", "std") / df.share_mean.replace(0, np.nan)
    df["snap_sd"] = roll("snap", "std") if "snap" in df else np.nan
    m = roll("ppr_act", "mean")
    df["ppr_mean"] = m
    df["out_cv"] = roll("ppr_act", "std") / m.where(m > 1)
    return df


def ratings(latest):
    """latest: one row per player (add_features(include_current=True), last game). Returns pid -> rating dict."""
    d = latest[latest.n_prior >= MIN_GAMES]
    # only real roles get a score: a backup steadily getting 2 targets is "steady" at nothing, and the backtest legs
    # (model mean 1.5+ catches / 15+ yards) never covered him. Others show "too small a role to rate".
    ok = (d.ppr_mean >= np.where(d.pos == "QB", 10, 6)) & (d.share_mean >= d.pos.map(MIN_SHARE))
    d = d[ok]
    out = {}
    for pos, g in d.groupby("pos"):
        role = g.role_cv.rank(pct=True, ascending=False)     # lower spread = steadier = higher percentile
        outp = g.out_cv.rank(pct=True, ascending=False)
        if pos == "QB":   # a starter's dropback share is ~100% every game, so the role part says nothing: output only
            role = role * np.nan
            overall = outp.fillna(0.5)
        else:
            overall = (WEIGHTS["role"] * role.fillna(0.5) + WEIGHTS["output"] * outp.fillna(0.5)).rank(pct=True)
        for i, r in enumerate(g.itertuples(index=False)):
            out[r.pid] = dict(score=int(round(100 * overall.iloc[i])),
                              role=int(round(100 * role.iloc[i])) if pd.notna(role.iloc[i]) else None,
                              output=int(round(100 * outp.iloc[i])) if pd.notna(outp.iloc[i]) else None,
                              n=int(min(r.n_prior, WINDOW)), n_pos=int(len(g)),
                              share=round(float(r.share_mean), 3), role_cv=round(float(r.role_cv), 2) if pos != "QB" else None,
                              out_cv=round(float(r.out_cv), 2) if pd.notna(r.out_cv) else None)
    return out


SHARE_WORD = {"RB": "carries + targets", "QB": "dropbacks", "WR": "targets", "TE": "targets"}


def role_flags(hist, week_rows, ov, names):
    """Reasons the season so far may not describe this week's role. hist: this season's player-games with share,
    team (posteam) and gameday; week_rows: this week's rows (pid, pos, posteam); ov: the week's overrides dict;
    names: pid -> display name. Returns pid -> list of short strings."""
    flags = {}
    hist = hist.sort_values("gameday")
    by_name = {v.lower(): k for k, v in names.items()}
    out_ids = {by_name[n.lower()] for n in ov.get("out", []) if n.lower() in by_name}
    back_ids = {by_name[n.lower()] for n in ov.get("back", []) if n.lower() in by_name}
    lead = ov.get("lead_back", {}) or {}
    season_share = hist.groupby("pid").share.mean()
    for r in week_rows.itertuples(index=False):
        f = []
        h = hist[hist.pid == r.pid].tail(WINDOW)
        if len(h):
            last_team = h.posteam.iloc[-1]
            if last_team != r.posteam:
                f.append(f"Traded: his earlier games were with {last_team}")
            if r.pid in back_ids:
                f.append("Back from an absence this week")
            if r.pos in ("RB", "WR", "TE") and len(h) >= 4:
                recent, before = h.share.tail(2).mean(), h.share.iloc[:-2].mean()
                if before > 0 and max(recent, before) >= 0.12 and abs(recent - before) >= 0.05 and abs(recent / before - 1) >= 0.35:
                    f.append(f"Role {'up' if recent > before else 'down'} lately: {recent:.0%} of {SHARE_WORD[r.pos]} "
                             f"the last 2 games vs {before:.0%} before")
            # a teammate at his position who played in most of this window is out this week
            games = set(h.game_id)
            same = REC if r.pos in REC else {r.pos}
            mates = hist[(hist.posteam == r.posteam) & hist.pid.isin(out_ids) & hist.pos.isin(same) & (hist.pid != r.pid)]
            for mid, mg in mates.groupby("pid"):
                msh = season_share.get(mid, 0)
                if len(games & set(mg.game_id)) >= max(2, len(games) / 2) and msh >= 0.15 and msh >= 0.6 * h.share.mean():
                    f.append(f"Teammate out: {names.get(mid, mid)} ({season_share[mid]:.0%} of {SHARE_WORD[r.pos]})")
        if r.pos == "RB" and r.posteam in lead:
            nm, shv = lead[r.posteam]
            f.append(f"News sets {nm} as lead back ({shv:.0%} of carries)")
        if f:
            flags[r.pid] = f
    return flags
