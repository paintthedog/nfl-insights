"""Test candidate factors walk-forward: defense INT shrinkage, pass breakups, FTN interception-worthy throws."""
import numpy as np, pandas as pd, glob, nflreadpy as nfl
import int_model as M

pbp, sched = M.load(); qb, tg, ctx = M.build_tables(pbp, sched)
gd = ctx[["game_id", "gameday"]].drop_duplicates()

# ---- pass breakups (PDs) from pbp
cols = ["game_id","play_id","posteam","defteam","passer_player_id","complete_pass","incomplete_pass",
        "interception","pass_defense_1_player_id","two_point_attempt"]
p = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_*.parquet"))])
p = p[(p.two_point_attempt != 1) & p.passer_player_id.notna()]
p["att"] = (p.complete_pass + p.incomplete_pass + p.interception).fillna(0)
p["pd_"] = (p.pass_defense_1_player_id.notna() & (p.interception != 1)).astype(int)  # breakups, excluding the INT itself

# ---- FTN interception-worthy throws (2022+)
ftn = pd.concat([nfl.load_ftn_charting([y]).to_pandas() for y in range(2022, 2027)])
ftn = ftn[["nflverse_game_id","nflverse_play_id","is_interception_worthy"]].rename(
    columns={"nflverse_game_id":"game_id","nflverse_play_id":"play_id"})
ftn["play_id"] = ftn.play_id.astype(float)
p["play_id"] = p.play_id.astype(float)
p = p.merge(ftn, on=["game_id","play_id"], how="left")
p["iw"] = (p.is_interception_worthy == True).astype(int)
p["charted"] = p.is_interception_worthy.notna().astype(int) * p.att

qg = p.groupby(["game_id","passer_player_id"], as_index=False).agg(pd_q=("pd_","sum"), iw=("iw","sum"), charted=("charted","sum"))
dg = p.groupby(["game_id","defteam"], as_index=False).agg(pd_d=("pd_","sum"))

lg_int = qb.ints.sum()/qb.att.sum()
# QB-level decayed priors
q = qb.merge(gd, on="game_id").merge(qg, on=["game_id","passer_player_id"], how="left").fillna({"pd_q":0,"iw":0,"charted":0})
q = M.decayed_prior(q, "passer_player_id", "gameday", ["att","pd_q","iw","charted","ints"], M.QB_DECAY)
lg_pd = qg.pd_q.sum()/qb.att.sum()
lg_iw = qg.iw.sum()/qg.charted.sum()
q["qb_pd_rate"] = (q.prior_pd_q + 600*lg_pd)/(q.prior_att + 600)
q["qb_iw_rate"] = (q.prior_iw + 300*lg_iw)/(q.prior_charted + 300)
q["qb_charted"] = q.prior_charted
# defense-level decayed priors
t = tg.merge(gd, on="game_id").merge(dg, on=["game_id","defteam"], how="left").rename(columns={"defteam":"team"})
t = M.decayed_prior(t, "team", "gameday", ["att","ints","pd_d"], M.TEAM_DECAY)
for K in [300, 800, 1500, 3000]:
    t[f"def_int_K{K}"] = np.log((t.prior_ints + K*lg_int)/(t.prior_att + K))
t["def_ballhawk"] = np.log((t.prior_ints + t.prior_pd_d + 800*(lg_int+lg_pd))/(t.prior_att + 800))

df = M.prep(M.build_features(qb, tg, ctx))
df = df[df.att.notna() & (df.att>=10) & df.exp_margin.notna() & (df.prior_games>=1)]
df = df.merge(q[["game_id","passer_player_id","qb_pd_rate","qb_iw_rate","qb_charted"]], on=["game_id","passer_player_id"], how="left")
df = df.merge(t[["game_id","team"]+[c for c in t.columns if c.startswith("def_")]].rename(columns={"team":"defteam"}), on=["game_id","defteam"], how="left")
df["log_qb_pd"] = np.log(df.qb_pd_rate); df["log_qb_iw"] = np.log(df.qb_iw_rate)
df["def_int_Kinf"] = 0.0

BASE = M.INT_X
NODEF = [c for c in BASE if c != "log_def_rate"]
def run(xs, seasons, data, label):
    M.INT_X = xs
    d = data.dropna(subset=M.ATT_X + xs[1:]); preds=[]
    for s in seasons:
        am = M.fit_attempts(d[d.season<s]); im = M.fit_ints(d[d.season<s], am)
        preds.append(M.predict(d[d.season==s], am, im))
    pr = pd.concat(preds); y=(pr.ints>=1).astype(int)
    b = np.mean((pr.p_int-y)**2); c = np.mean((y.mean()-y)**2)
    return label, len(pr), b, 1-b/c, im.params.get(xs[-1], np.nan)

rows=[]
S = range(2019, 2027)
rows.append(run(BASE, S, df, "current model"))
rows.append(run(NODEF, S, df, "no defense INT feature"))
for K in [300, 800, 3000]:
    rows.append(run(NODEF+[f"def_int_K{K}"], S, df, f"defense INT, shrink K={K}"))
rows.append(run(BASE+["def_ballhawk"], S, df, "+ defense INTs+breakups"))
rows.append(run(BASE+["log_qb_pd"], S, df, "+ QB passes-defensed rate"))
print(pd.DataFrame(rows, columns=["variant","n","brier","skill","last_coef"]).round(5).to_string())

# FTN only exists 2022+: evaluate on 2024-2026 with priors from 2022 on, QBs with 150+ charted att
sub = df[(df.season>=2022)].copy()
sub = sub[(sub.season<2023) | (sub.qb_charted>=150)]
S2 = range(2024, 2027)
rows=[run(BASE, S2, sub, "current model (FTN-era subset)"),
      run(BASE+["log_qb_iw"], S2, sub, "+ QB INT-worthy throw rate"),
      run([c for c in BASE if c!="log_qb_rate"]+["log_qb_iw"], S2, sub, "INT-worthy REPLACES QB INT rate")]
print(pd.DataFrame(rows, columns=["variant","n","brier","skill","last_coef"]).round(5).to_string())
print("\ncorr of QB rates with actual INT/att, FTN era:",
      sub[["qb_int_rate","qb_iw_rate","qb_pd_rate"]].corrwith(sub.ints/sub.att).round(3).to_dict())
