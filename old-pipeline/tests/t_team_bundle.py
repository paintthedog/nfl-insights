"""Team-ratings bundle (Oct 16): (1) garbage-time filter on the EPA/success ratings, (2) fumble-luck neutralised,
(3) explosive-play rate (20+ yard gains) for and against. Walk-forward 2019-2026 vs the current model (incl. success rate).
Pace stays computed from all plays in every variant."""
import sys; sys.path.insert(0, ".")
import glob, numpy as np, pandas as pd
import lines_model as L
COLS = L.PBP_COLS + ["fumble", "fumble_lost", "yards_gained"]
pbp0 = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob("data/pbp_*.parquet"))])
sched = pd.read_parquet("data/sched.parquet"); sched = sched[(sched.season >= 2016) & (sched.game_type == "REG")].copy()
base = L.build(pbp0, sched)
PACE = ["h_pace", "a_pace", "pace_sum"]
# explosive-play rates, prior games only, same decay / shrinkage as the ratings
p = pbp0[(pbp0.season_type == "REG") & pbp0.posteam.notna() & pbp0.epa.notna() & pbp0.play_type.isin(["pass", "run"]) & (pbp0.two_point_attempt != 1)].copy()
p["x"] = (p.yards_gained >= 20).astype(float); p["n"] = 1.0
t = p.groupby(["game_id", "posteam", "defteam"])[["x", "n"]].sum().reset_index()
o = t.rename(columns={"posteam": "team"}).drop(columns="defteam"); dd = t.rename(columns={"defteam": "team", "x": "dx", "n": "dn"}).drop(columns="posteam")
s = base[["game_id", "season", "gameday", "home_team", "away_team"]].copy(); s["gameday"] = pd.to_datetime(s.gameday)
lg_ = pd.concat([s.assign(team=s.home_team, home=1), s.assign(team=s.away_team, home=0)])[["game_id", "season", "gameday", "team", "home"]]
lg_ = lg_.merge(o, on=["game_id", "team"], how="left").merge(dd, on=["game_id", "team"], how="left")
lg_ = L.decayed(lg_, "team", ["x", "n", "dx", "dn"], L.DECAY, L.SEASON_CARRY)
lr = p.x.mean(); K = L.K_RATE
lg_["xo"] = (lg_.ds_x + K * lr) / (lg_.ds_n + K); lg_["xd"] = (lg_.ds_dx + K * lr) / (lg_.ds_dn + K)
h = lg_[lg_.home == 1][["game_id", "xo", "xd"]].rename(columns={"xo": "h_xo", "xd": "h_xd"}); a = lg_[lg_.home == 0][["game_id", "xo", "xd"]].rename(columns={"xo": "a_xo", "xd": "a_xd"})
EXPL = h.merge(a, on="game_id")
def variant(garbage=False, fumble=False, explosive=False):
    pb = pbp0.copy()
    if fumble:   # every fumble play gets the league-average EPA of fumble plays: fumbling still costs, the bounce doesn't count
        m = (pb.fumble == 1) & pb.epa.notna() & pb.play_type.isin(["pass", "run"]); pb.loc[m, "epa"] = pb.loc[m, "epa"].mean()
    if garbage:  # rate teams only on plays while the game is in doubt (pace keeps every play, restored below)
        pb = pb[pb.wp.between(0.10, 0.90) | pb.wp.isna()]
    g = L.build(pb, sched)
    g = g.drop(columns=PACE).merge(base[["game_id"] + PACE], on="game_id")
    if explosive:
        g = g.merge(EXPL, on="game_id", how="left")
        g["x_gap"] = (g.h_xo - g.a_xo) + (g.a_xd - g.h_xd); g["x_sum"] = g.h_xo + g.a_xo + g.h_xd + g.a_xd
    return g
base_m, base_t = list(L.MARGIN_X), list(L.TOTAL_X)
V = {"current": {}, "garbage-time filter": dict(garbage=True), "fumble luck": dict(fumble=True), "explosive plays": dict(explosive=True),
     "all three": dict(garbage=True, fumble=True, explosive=True)}
res, rows = {}, []
for lab, kw in V.items():
    g = variant(**kw); g = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & (g.min_games >= 3)].copy()
    L.MARGIN_X = base_m + (["x_gap"] if kw.get("explosive") else []); L.TOTAL_X = base_t + (["x_sum"] if kw.get("explosive") else [])
    outs = []
    for yr in range(2019, 2027):
        mm, mt = L.fit(g[g.season < yr]); outs.append(L.predict(g[g.season == yr].copy(), mm, mt))
    oo = pd.concat(outs).set_index("game_id").sort_index(); res[lab] = oo
    hc, oh = np.sign(oo.result - oo.spread_line), np.sign(oo.total - oo.total_line)
    r = dict(variant=lab, n=len(oo), margin_mae=np.mean(np.abs(oo.result - oo.m_margin)), total_mae=np.mean(np.abs(oo.total - oo.m_total)))
    pk = np.sign(oo.m_margin - oo.spread_line); m = (np.abs(oo.m_margin - oo.spread_line) >= 3) & (hc != 0); r["ATS 3+"] = f"{(pk[m] == hc[m]).mean()*100:.1f}% ({m.sum()})"
    pk = np.sign(oo.m_total - oo.total_line); m = (np.abs(oo.m_total - oo.total_line) >= 3) & (oh != 0); r["OU 3+"] = f"{(pk[m] == oh[m]).mean()*100:.1f}% ({m.sum()})"
    if lab != "current":
        c = res["current"].loc[oo.index]
        for k_, act, mc in (("margin", "result", "m_margin"), ("total", "total", "m_total")):
            dlt = np.abs(c[act] - c[mc]) - np.abs(oo[act] - oo[mc])
            r[f"{k_}_gain"] = f"{dlt.mean():+.3f} ±{dlt.std()/np.sqrt(len(dlt)):.3f} ({int((dlt.groupby(oo.season).mean() > 0).sum())}/8 seasons)"
    rows.append(r); print("done", lab, flush=True)
L.MARGIN_X, L.TOTAL_X = base_m, base_t
pd.set_option("display.width", 250); R = pd.DataFrame(rows).round(3); print(R.to_string(index=False)); R.to_csv("tests/team_bundle_results.csv", index=False)
