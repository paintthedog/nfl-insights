"""Project Week N player stats, fantasy points and outcome ranges. Writes proj_<season>_wk<week>.json."""
import json, pickle
import numpy as np, pandas as pd
import proj_model as P, injuries as J

SEASON, WEEK = 2026, int(__import__("os").environ.get("NFL_WEEK", 5))

pbp = P.load()
sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
td = pd.read_csv(f"td_props_{SEASON}_wk{WEEK}.csv")          # player pool + TD rates (already team-scaled)
ints = pd.read_csv(f"int_props_{SEASON}_wk{WEEK}.csv")[["passer_player_id", "exp_ints"]]

pred = td[["game_id", "posteam", "pid"]].copy()
pred["gameday"] = pd.to_datetime(pred.game_id.map(sched.set_index("game_id").gameday))
pred["season"], pred["week"] = SEASON, WEEK
try:   # this week's regular OL starters ruled Out (ol_missing_wk4.csv, built from the injury report)
    _olm = pd.read_csv(f"ol_missing_wk{WEEK}.csv").set_index("team").ol_missing_pred
    pred["ol_missing"] = pred.posteam.map(_olm).fillna(0.0)
except FileNotFoundError:
    pred["ol_missing"] = 0.0

d = P.build(pbp, sched, players, predict_rows=pred)
# next-man-up backs added by td_project have no usage history; give them their seeded role and position-average efficiency
_seeded = set(td[td.get("seeded", 0) == 1].pid) if "seeded" in td else set()
if _seeded:
    m = d._pred & d.pid.isin(_seeded)
    d.loc[m, "pr_car_sh"] = 0.12; d.loc[m, "pr_tgt_sh"] = 0.04; d.loc[m, "pr_ay_sh"] = 0.0
    d.loc[m, "pr_snap"] = 0.35; d.loc[m, "last_snap"] = 0.35
    for c in [c for c in d.columns if c.startswith("ds_")]: d.loc[m, c] = d.loc[m, c].fillna(0.0)
    print("seeded next-man-up:", d.loc[m, "pid"].tolist())
pm = P.position_means(d)
d = P.efficiency(d, pm)
if _seeded:
    m = d._pred & d.pid.isin(_seeded)
    d.loc[m, "l_snap"] = np.log(0.35); d.loc[m, "snap_dn"] = 0.0
    d.loc[m, "l_wopr"] = np.log(1.5 * 0.04 + 0.01)
hist = d[~d._pred]
va, vc = P.fit_volume(hist.drop_duplicates(["game_id", "posteam"]))
hist = P.raw_projection(hist.copy(), va, vc)
cal = P.fit_calibration(hist[hist.pr_n >= 1])
f = P.raw_projection(d[d._pred].copy(), va, vc)
f = P.apply_calibration(f, cal)

# ---- injuries: returning players are eased back in; Questionable players sit some of the time (the simulation plays that out)
_inj = td.drop_duplicates("pid").set_index("pid")
f["p_play"] = f.pid.map(_inj.p_play).fillna(1.0); f["back_mult"] = f.pid.map(_inj.back_mult).fillna(1.0)
REC_S, RUSH_S = ["p_tgt", "p_rec", "p_rec_yds", "p_rec_1d"], ["p_car", "p_rush_yds", "p_rush_1d"]
nq = f.pos != "QB"
for c in REC_S + RUSH_S:
    f.loc[nq, c] = f.loc[nq, c] * f.loc[nq, "back_mult"]
# ---- make each team add up: the pool gets POOL_FRAC of the team (the rest goes to players who aren't listed), counting
#      a Questionable player at his chance of playing. p_* stay "if he plays" numbers.
_q = f[f.pos == "QB"]     # a team with an uncertain starter has two quarterbacks: the team's passing line is their chance-weighted mix
qbt = _q[["p_att", "p_cmp", "p_pass_yds", "p_pass_1d"]].mul(_q.p_play, axis=0).groupby(_q.posteam).sum()
recv = f[nq].assign(**{c: f.loc[nq, c] * f.loc[nq, "p_play"] for c in REC_S}).groupby("posteam")[REC_S].sum()
for stat, qcol in [("p_tgt", "p_att"), ("p_rec", "p_cmp"), ("p_rec_yds", "p_pass_yds"), ("p_rec_1d", "p_pass_1d")]:
    target = qbt[qcol] * (0.985 if stat == "p_tgt" else 1.0) * J.POOL_FRAC
    scale = (target / recv[stat]).reindex(f.posteam).to_numpy()
    f[stat] = np.where(f.pos != "QB", f[stat] * np.nan_to_num(scale, nan=1.0), f[stat])
# QBs keep their own rushing projection; non-QBs share what's left of the team's carries
qb_car = (_q.p_car * _q.p_play).groupby(_q.posteam).sum()
team_car = f.drop_duplicates("posteam").set_index("posteam").v_car
other = (f.p_car * f.p_play)[nq].groupby(f.posteam[nq]).sum()
car_scale = ((team_car - qb_car.reindex(team_car.index).fillna(0)) * J.POOL_FRAC / other).reindex(f.posteam).to_numpy()
for stat in ["p_car", "p_rush_yds", "p_rush_1d"]:
    f[stat] = np.where(f.pos != "QB", f[stat] * np.nan_to_num(car_scale, nan=1.0), f[stat])
print("team scale ranges: receiving", np.nanmin(scale).round(2), "-", np.nanmax(scale).round(2),
      "| carries", np.nanmin(car_scale).round(2), "-", np.nanmax(car_scale).round(2))
_rs = pd.Series(scale, index=f.posteam).groupby(level=0).first()
_cs = pd.Series(car_scale, index=f.posteam).groupby(level=0).first()
REDIST = [dict(team=t, recv=round(float(_rs.get(t, 1)), 2), car=round(float(_cs.get(t, 1)), 2))
          for t in sorted(set(f.posteam)) if _rs.get(t, 1) >= 1.3 or _cs.get(t, 1) >= 1.5]

f = f.merge(td[["pid", "name", "lam_rush", "lam_rec", "exp_td", "rush_share", "p_td", "flag"]], on="pid", how="left")
f["p_rush_td"] = f.lam_rush.fillna(0)
f["p_rec_td"] = f.lam_rec.fillna(0)
f["p_pass_td"] = np.where(f.pos == "QB", f.exp_td * (1 - f.rush_share) * 0.97, 0)
f = f.merge(ints.rename(columns={"passer_player_id": "pid"}), on="pid", how="left")
f.loc[f.pos == "QB", "p_ints"] = f.loc[f.pos == "QB", "exp_ints"].fillna(f.loc[f.pos == "QB", "p_ints"])
for fmt, v in P.fantasy(f, "p_").items():
    f[f"p_fp_{fmt}"] = v

# ---- consistency, this season: quality starts / duds (PPR vs positional cutoffs) and per-game values for the
# priced stats, so the page can show "cleared this line in X of N games". Descriptive only: past QS rate showed no
# predictive power beyond average points (t < 1.6 at every position, 2022-25 split-half), so the model ignores it.
QS_AT = {"QB": 18.4, "RB": 11.1, "WR": 12.3, "TE": 10.1}
DUD_AT = {"QB": 10.5, "RB": 4.4, "WR": 6.0, "TE": 6.2}
_h = d[(~d._pred) & (d.season == SEASON)].copy()
_h["fp"] = P.fantasy(_h.assign(**{f"a_{c}": _h[c] for c in ["pass_yds","pass_td","ints","sacks","pass_1d","rush_yds","rush_td","rush_1d","rec","rec_yds","rec_td","rec_1d"]}), "a_")["ppr"]
_h = _h.sort_values("gameday")
CONS = {}
for pid, g_ in _h.groupby("pid"):
    pos_ = g_.pos.iloc[0]
    if pos_ not in QS_AT: continue
    CONS[pid] = dict(gp=int(len(g_)), qs=int((g_.fp >= QS_AT[pos_]).sum()), dud=int((g_.fp <= DUD_AT[pos_]).sum()),
                     hist={st: [round(float(x), 1) for x in g_[st]] for st in ["pass_yds", "rush_yds", "rec_yds", "rec"]})
store = pickle.load(open("ratio_table.pkl", "rb"))
tab = store["tab"]
rows = []
for _, r in f.iterrows():
    if r.pos not in ("QB", "RB", "WR", "TE"):
        continue
    q = {}
    for st in P.RANGE_STATS:
        qs = P.quantiles_for(tab, st, r.pos, r[f"p_{st}"])
        if qs:
            q[st] = [round(x, 2) for x in qs]
    rows.append(dict(
        cons=CONS.get(r.pid),
        pid=r.pid, name=r["name"], pos=r.pos, team=r.posteam, opp=r.opp, implied=round(float(r.implied), 1),
        flag=r.flag if isinstance(r.flag, str) else "",
        mean={k: round(float(r[f"p_{k}"]), 2) for k in ["att", "cmp", "pass_yds", "pass_td", "ints", "sacks", "pass_1d",
                                                        "car", "rush_yds", "rush_td", "rush_1d",
                                                        "tgt", "rec", "rec_yds", "rec_td", "rec_1d",
                                                        "fp_legends", "fp_ppr", "fp_std"]},
        p_td=round(float(r.p_td), 3) if pd.notna(r.p_td) else None,
        snap=round(float(r.pr_snap), 3) if pd.notna(r.pr_snap) else None,
        last_snap=round(float(r.last_snap), 3) if pd.notna(r.last_snap) else None,
        wopr=round(float(1.5 * np.nan_to_num(r.pr_tgt_sh) + 0.7 * min(max(np.nan_to_num(r.pr_ay_sh), 0), 1)), 3), q=q))
# ---- game simulation: every player in a game simulated jointly (shared team volume, shared team TDs, role noise,
# hot/cold days, early exits). Shipped per player: simulated mean + quantiles for each stat and for fantasy points,
# which REPLACE the ratio-table ranges where the sim covers the stat, plus floor / ceiling / boom / bust.
import sim_model as S
rng = np.random.default_rng(2026 * 100 + WEEK)
N_SIM = 6000
BOOM_AT = {"QB": 25, "RB": 20, "WR": 20, "TE": 15}     # PPR points that count as a boom game
BUST_AT = {"QB": 12, "RB": 6, "WR": 6, "TE": 4}        # ... and a bust
STD_SCALE = {"QB": 1.0, "RB": 0.75, "WR": 0.7, "TE": 0.7}   # same thresholds scaled for standard scoring
sim_players = {}
for gid, g in f.groupby("game_id"):
    team_rows = {}
    for tm, x in g.groupby("posteam"):
        x = x.copy(); x["name"] = x["name"].fillna(x.pid)
        team_rows[tm] = x[["pid", "name", "pos", "p_tgt", "p_car", "p_rec", "p_rec_yds", "p_rush_yds", "e_catch", "e_ypt", "e_ypc",
                           "lam_rush", "lam_rec", "v_att", "v_car", "exp_td", "rush_share", "p_ints", "p_play", "p_pass_yds"]]
    sim = S.simulate_game(team_rows, n_sim=N_SIM, rng=rng, pool_frac=J.POOL_FRAC)
    for tm, ss in sim.items():
        for i, pid in enumerate(ss["pids"]):
            # a player who might sit: his own numbers are taken from the runs where he dresses ("if he plays");
            # teammates' numbers come from all runs, so they already carry the chance he sits
            act = ss.get("active"); keep = act[:, i] if (act is not None and not act[:, i].all() and act[:, i].any()) else slice(None)
            st = {k: ss[k][keep, i] for k in ["tgt", "rec", "rec_yds", "car", "rush_yds", "rec_td", "rush_td", "pass_yds", "pass_td", "cmp", "ints", "fp_ppr", "fp_std", "fp_dk"]}
            st["td"] = st["rec_td"] + st["rush_td"]
            sd = {k: [round(float(np.quantile(v, q)), 1) for q in P.QUANTS] for k, v in st.items()}
            sd["mean"] = {k: round(float(v.mean()), 2) for k, v in st.items()}
            sd["p_td"] = round(float((st["td"] >= 1).mean()), 3)
            pos = ss["pos"][i]
            for fmt in ("ppr", "std"):
                v = st[f"fp_{fmt}"]; k = 1.0 if fmt == "ppr" else STD_SCALE.get(pos, 1.0)
                sd[f"var_{fmt}"] = dict(floor=round(float(np.quantile(v, .1)), 1), ceil=round(float(np.quantile(v, .9)), 1),
                                       sd=round(float(v.std()), 1),
                                       boom=round(float((v >= BOOM_AT.get(pos, 20) * k).mean()), 3),
                                       bust=round(float((v <= BUST_AT.get(pos, 6) * k).mean()), 3))
            sd["dk"] = dict(pts=round(float(st["fp_dk"].mean()), 2), ceil=round(float(np.quantile(st["fp_dk"], .9)), 1))
            sim_players[pid] = sd
for r in rows:
    sp = sim_players.get(r["pid"])
    if sp is None:
        continue
    r["sim"] = sp
    for st in ["tgt", "rec", "rec_yds", "car", "rush_yds", "pass_yds", "pass_td", "cmp", "ints", "fp_ppr", "fp_std"]:
        r["q"][st] = sp[st]                       # simulated ranges replace the ratio-table ranges
    m0 = dict(r["mean"])
    for st in ["tgt", "rec", "rec_yds", "car", "rush_yds", "rec_td", "rush_td", "pass_yds", "pass_td", "cmp", "ints"]:
        if st in ("pass_yds", "pass_td", "cmp", "ints") and r["pos"] != "QB": continue
        r["mean"][st] = sp["mean"][st]
    for st, drv in [("rec_1d", "rec"), ("rush_1d", "car")]:      # first downs aren't simulated: move them with catches / carries
        if m0[drv] > 0.05: r["mean"][st] = round(m0[st] * sp["mean"][drv] / m0[drv], 2)
    r["p_play"] = round(float(_inj.p_play.get(r["pid"], 1.0)), 2)
    r["mean"]["fp_ppr"], r["mean"]["fp_std"] = sp["mean"]["fp_ppr"], sp["mean"]["fp_std"]   # points from the sim: TDs counted once, INTs included
    r["var"] = {fmt: sp[f"var_{fmt}"] for fmt in ("ppr", "std")}
    r["dk"] = sp["dk"]
out = dict(quants=P.QUANTS, recal=store["recal"], players=rows, redistrib=REDIST, boom_at=BOOM_AT, bust_at=BUST_AT, sim_params=dict(n_sim=N_SIM, share_sd=S.SHARE_SD, eff_sd=S.GAME_EFF_SD, exit_p=S.EXIT_P, td_overdisp=S.TD_OVERDISP))
json.dump(out, open(f"proj_{SEASON}_wk{WEEK}.json", "w"))
print("simulated", f.game_id.nunique(), "games x", N_SIM)

v = pd.DataFrame([dict(name=x["name"], pos=x["pos"], team=x["team"], opp=x["opp"], **x["mean"]) for x in rows])
pd.set_option("display.width", 220)
for grp, cols in [("QB", ["pass_yds", "pass_td", "ints", "rush_yds", "fp_legends", "fp_std"]),
                  ("RB", ["car", "rush_yds", "rec", "rec_yds", "fp_legends", "fp_ppr"]),
                  ("WR", ["tgt", "rec", "rec_yds", "fp_legends", "fp_ppr"]), ("TE", ["tgt", "rec", "rec_yds", "fp_legends"])]:
    print(f"\n{grp}"); print(v[v.pos == grp].sort_values("fp_legends", ascending=False).head(8)[["name", "team", "opp"] + cols].round(1).to_string(index=False))
print(f"\n{len(rows)} players; team check after scaling (QB pass yds vs sum of receivers), 3 teams:")
qbpy = v[v.pos == "QB"].groupby("team").pass_yds.mean()
recy = v[v.pos != "QB"].groupby("team").rec_yds.sum()
print(pd.DataFrame(dict(qb=qbpy, receivers=recy)).dropna().head(3).round(0))
