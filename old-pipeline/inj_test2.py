"""Tests 2 and 3 (and the pool leak):
  leak : share of a team's targets / carries that goes to players who weren't in its last two games (so can't be in the pool)
  Q    : how often Questionable skill players actually play, by final practice status, and their usage when they do
  back : usage in the first game(s) back after missing time"""
import numpy as np, pandas as pd, statsmodels.api as sm, pickle
d = pd.read_parquet("data/inj_test.parquet")
gm = d[["game_id", "posteam", "season", "week", "gameday"]].drop_duplicates().sort_values(["posteam", "gameday"])
gm["k"] = gm.groupby(["posteam", "season"]).cumcount()
d = d.merge(gm[["game_id", "posteam", "k"]], on=["game_id", "posteam"])
snaps = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "offense_pct"]).rename(columns={"gsis_id": "pid"})
snaps = snaps[snaps.offense_pct > 0]
touched = set(zip(d.game_id, d.pid)); snapped = set(zip(snaps.game_id, snaps.pid))
pid_games = d.groupby(["posteam", "season", "k"]).pid.apply(set).to_dict()

# ---- leak
rows = []
for (tm, ssn, k), x in d.groupby(["posteam", "season", "k"]):
    if k < 2: continue
    pool = pid_games.get((tm, ssn, k - 1), set()) | pid_games.get((tm, ssn, k - 2), set())
    out = ~x.pid.isin(pool)
    rows.append(dict(t_out=x.tgt[out].sum(), t=x.tgt.sum(), c_out=x.car[out & (x.pos != "QB")].sum(), c=x.car[x.pos != "QB"].sum()))
lk = pd.DataFrame(rows); LEAK_T, LEAK_C = lk.t_out.sum() / lk.t.sum(), lk.c_out.sum() / lk.c.sum()
print(f"leak: {LEAK_T:.1%} of targets and {LEAK_C:.1%} of non-QB carries go to players outside the last-two-games pool")

# ---- Questionable
inj = pd.read_parquet("data/injuries_all.parquet")
inj = inj[(inj.game_type == "REG") & inj.position.isin(["QB", "RB", "WR", "TE", "FB"]) & inj.gsis_id.notna() & (inj.season >= 2019)].copy()
inj["prac"] = inj.practice_status.map({"Full Participation in Practice": "Full", "Limited Participation in Practice": "Limited", "Did Not Participate In Practice": "DNP"})
inj = inj.merge(gm.rename(columns={"posteam": "team"})[["season", "week", "team", "game_id", "k"]], on=["season", "week", "team"])
inj["played"] = [((g, p) in touched) or ((g, p) in snapped) for g, p in zip(inj.game_id, inj.gsis_id)]
inj["in_pool"] = [p in (pid_games.get((t, s, k - 1), set()) | pid_games.get((t, s, k - 2), set())) for t, s, k, p in zip(inj.team, inj.season, inj.k, inj.gsis_id)]
inj["pos"] = inj.position.replace({"FB": "RB"})
q = inj[inj.in_pool]
print("\nplay rate by final game status (players who were in the team's last two games):")
print(q.groupby(q.report_status.fillna("no status")).played.agg(["size", "mean"]).round(3))
qq = q[q.report_status == "Questionable"]
print("\nQuestionable, by last practice of the week:\n", qq.groupby("prac").played.agg(["size", "mean"]).round(3))
print("\nQuestionable, by position:\n", qq.groupby("pos").played.agg(["size", "mean"]).round(3))
print("\nQuestionable by season:\n", qq.groupby("season").played.mean().round(3).to_dict())
print("Doubtful by practice:\n", q[q.report_status == "Doubtful"].groupby("prac").played.agg(["size", "mean"]).round(3))
PLAY = dict(Questionable=qq.groupby("prac").played.mean().round(3).to_dict(), Questionable_all=round(float(qq.played.mean()), 3),
            Doubtful=round(float(q[q.report_status == "Doubtful"].played.mean()), 3))

# ---- usage when they do play, and first games back
st = inj[["game_id", "gsis_id", "report_status", "prac"]].rename(columns={"gsis_id": "pid"}).drop_duplicates(["game_id", "pid"])
d = d.merge(st, on=["game_id", "pid"], how="left")
d = d.sort_values(["pid", "gameday"])
prev_k = d.groupby(["pid", "posteam", "season"]).k.shift(1)
d["missed"] = np.where(prev_k.notna(), d.k - prev_k - 1, np.where(d.pr_n >= 4, d.k, 0)).clip(0, 8)   # team games missed right before this one
x = d[(d.pr_n >= 2) & d.pos.isin(["RB", "WR", "TE"]) & (d.p_tgt + d.p_car >= 3)].copy()
x["touch"] = x.tgt + x.car; x["p_touch"] = x.p_tgt + x.p_car; x["yds"] = x.rec_yds + x.rush_yds; x["p_yds"] = x.p_rec_yds + x.p_rush_yds
def ratio(g): return pd.Series(dict(n=len(g), usage=g.touch.sum() / g.p_touch.sum(), yards=g.yds.sum() / g.p_yds.sum()))
base = ratio(x[(x.missed == 0) & x.report_status.isna()])
print(f"\nbaseline (not on report, played last game): usage {base.usage:.3f} yards {base.yards:.3f}  n={int(base.n)}")
x["grp_q"] = np.where(x.report_status == "Questionable", "Questionable/" + x.prac.fillna("?"), np.where(x.prac.notna(), "listed, no status/" + x.prac.fillna("?"), "not listed"))
print("\nusage vs projection when they played, by report line (relative to baseline):")
print((x[x.missed == 0].groupby("grp_q").apply(ratio).assign(usage=lambda t: t.usage / base.usage, yards=lambda t: t.yards / base.yards)).round(3))
x["mb"] = pd.cut(x.missed, [-1, 0, 1, 3, 8], labels=["0", "1", "2-3", "4+"])
print("\nfirst game back, by team games missed:")
print(x.groupby("mb", observed=True).apply(ratio).assign(usage=lambda t: t.usage / base.usage, yards=lambda t: t.yards / base.yards).round(3))
print("\n... split by position (missed 1+):")
print(x[x.missed >= 1].groupby("pos").apply(ratio).assign(usage=lambda t: t.usage / base.usage, yards=lambda t: t.yards / base.yards).round(3))
# second game back
d["missed_prev"] = d.groupby("pid").missed.shift(1).fillna(0)
x = x.merge(d[["game_id", "pid", "missed_prev"]], on=["game_id", "pid"])
print("\nsecond game back (missed 2+ before the previous game):", ratio(x[(x.missed == 0) & (x.missed_prev >= 2)]).div(pd.Series(dict(n=1, usage=base.usage, yards=base.yards))).round(3).to_dict())

# ---- walk-forward: do these multipliers reduce error?
x["q_lim"] = ((x.report_status == "Questionable") & (x.prac != "Full")).astype(float); x["q_full"] = ((x.report_status == "Questionable") & (x.prac == "Full")).astype(float)
x["back1"] = (x.missed == 1).astype(float); x["back2"] = (x.missed >= 2).astype(float)
X = ["q_lim", "q_full", "back1", "back2"]
res = []
for s in range(2022, 2027):
    tr, te = x[x.season < s], x[x.season == s].copy()
    m = sm.GLM(tr.touch, sm.add_constant(tr[X]), family=sm.families.Poisson(), offset=np.log(tr.p_touch)).fit()
    te["mult"] = np.exp(m.predict(sm.add_constant(te[X], has_constant="add"), offset=np.zeros(len(te))).pipe(np.log)) / np.exp(m.params["const"])
    res.append(te)
o = pd.concat(res)
for lab, mk in [("all", o.touch >= 0), ("Questionable, not full practice", o.q_lim == 1), ("first game back", o.missed >= 1)]:
    y = o[mk]
    print(f"{lab:34s} n={len(y):5d}  touches MAE {np.abs(y.touch - y.p_touch).mean():.3f} -> {np.abs(y.touch - y.p_touch * y.mult).mean():.3f}   yards MAE {np.abs(y.yds - y.p_yds).mean():.2f} -> {np.abs(y.yds - y.p_yds * y.mult).mean():.2f}   bias {y.p_touch.sum() / y.touch.sum() - 1:+.1%} -> {(y.p_touch * y.mult).sum() / y.touch.sum() - 1:+.1%}")
m = sm.GLM(x.touch, sm.add_constant(x[X]), family=sm.families.Poisson(), offset=np.log(x.p_touch)).fit()
print("\nmultipliers (all seasons):", {k: round(float(np.exp(v)), 3) for k, v in m.params.items() if k != "const"}, "| p-values", {k: round(float(v), 3) for k, v in m.pvalues.items() if k != "const"})
pickle.dump(dict(leak_t=float(LEAK_T), leak_c=float(LEAK_C), play=PLAY, usage={k: float(np.exp(v)) for k, v in m.params.items() if k != "const"}), open("inj_params.pkl", "wb"))
