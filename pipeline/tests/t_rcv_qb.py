"""Receivers: does the quality of THIS game's quarterback, relative to the quarterbacks behind the receiver's recent numbers,
improve receiving projections? Feature qd = starter's shrunk EPA/dropback minus the decayed mix of the team's recent starters
(same construction as the team model). Walk-forward 2021-2026, same harness as proj_backtest.py."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import proj_model as P, team_adj as TA
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); pm = P.position_means(d); d = P.efficiency(d, pm)
# qd per team-game
import glob
full = pd.concat([pd.read_parquet(f, columns=["game_id", "season", "season_type", "posteam", "qb_dropback", "passer_player_id", "epa"]) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
H = TA._qb_hist(full, sched)
def q_at(pid, day):
    if not isinstance(pid, str) or pid not in H: return TA.PRIOR
    days, ce, cn = H[pid]; i = np.searchsorted(days, np.datetime64(day)) - 1
    e, n = (ce[i], cn[i]) if i >= 0 else (0.0, 0.0); return (e + TA.K * TA.PRIOR) / (n + TA.K)
f = full[(full.season_type == "REG") & (full.qb_dropback == 1)]
st = f.groupby(["game_id", "season", "posteam", "passer_player_id"]).size().reset_index(name="n").sort_values("n").drop_duplicates(["game_id", "posteam"], keep="last")
st = st.merge(sched[["game_id", "gameday"]], on="game_id"); st["gameday"] = pd.to_datetime(st.gameday); st = st.sort_values(["posteam", "gameday"])
rows = []
for tm, g in st.groupby("posteam"):
    past = []
    for r in g.itertuples():
        w = num = 0.0
        for i, (ss, qq) in enumerate(reversed(past[-40:])):
            wt = TA.DECAY ** i * (TA.CARRY if ss < r.season else 1.0); w += wt; num += wt * q_at(qq, r.gameday)
        qr = num / w if w else TA.PRIOR; qt = q_at(r.passer_player_id, r.gameday)
        rows.append((r.game_id, tm, qt - qr, past[-1][1] != r.passer_player_id if past else False)); past.append((r.season, r.passer_player_id))
Q = pd.DataFrame(rows, columns=["game_id", "posteam", "qd", "chg"])
d = d.merge(Q, on=["game_id", "posteam"], how="left"); d["qd"] = d.qd.fillna(0); d["chg"] = d.chg.fillna(False)
def run(extra):
    P.EXTRA = {**{s: ["l_snap", "snap_dn", "l_wopr"] + extra for s in ["tgt", "rec", "rec_yds", "rec_1d"]}, **{s: ["l_snap", "snap_dn"] for s in ["car", "rush_yds", "rush_1d"]}}
    out = []
    for s in range(2021, 2027):
        tr = d[d.season < s].copy(); te = d[d.season == s].copy(); teams = tr.drop_duplicates(["game_id", "posteam"])
        va, vc = P.fit_volume(teams); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); out.append(P.apply_calibration(te, cal))
    o = pd.concat(out); return o[(o.pr_n >= 1) & o.implied.notna() & o.pos.isin(["RB", "WR", "TE"])]
A = run([]); B = run(["qd"])
for nm, sel in (("all", lambda x: x.index == x.index), ("QB changed from last game", lambda x: x.chg), ("|qd| > 0.05", lambda x: x.qd.abs() > 0.05)):
    a, b = A[sel(A)], B[sel(B)]
    print(nm, "n", len(a), " ".join(f"{st}: {np.abs(a[st]-a['p_'+st]).mean():.3f}->{np.abs(b[st]-b['p_'+st]).mean():.3f}" for st in ["tgt", "rec", "rec_yds"]))
