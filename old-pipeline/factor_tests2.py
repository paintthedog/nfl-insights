src = open("factor_tests.py").read().split("rows=[]")[0]
exec(src)
# Blended QB risk: INT-worthy rate shrunk toward the QB's OWN INT-rate-implied level,
# so pre-2022 / uncharted QBs fall back to INT history and charted QBs move to INT-worthy.
q["int_implied_iw"] = q.prior_ints.pipe(lambda s: (q.prior_ints + 600*lg_int)/(q.prior_att + 600)) * (lg_iw/lg_int)
for K in [100, 200, 400]:
    q[f"qb_risk_K{K}"] = (q.prior_iw + K*q.int_implied_iw)/(q.prior_charted + K)
df2 = df.merge(q[["game_id","passer_player_id"]+[f"qb_risk_K{K}" for K in [100,200,400]]], on=["game_id","passer_player_id"], how="left")
for K in [100,200,400]: df2[f"log_risk_K{K}"] = np.log(df2[f"qb_risk_K{K}"])
S = range(2019, 2027); S2 = range(2024, 2027)
sub = df[(df.season>=2022)].copy(); sub = sub[(sub.season<2023) | (sub.qb_charted>=150)]
NOQB = [c for c in BASE if c != "log_qb_rate"]
rows=[run(BASE, S, df2, "current model")]
rows.append(run(BASE+["log_qb_pd"], S, df2, "+ QB PD rate"))
for K in [100,200,400]:
    rows.append(run(NOQB+[f"log_risk_K{K}"], S, df2, f"blended risk K={K}"))
    rows.append(run(NOQB+["log_qb_pd", f"log_risk_K{K}"], S, df2, f"blended risk K={K} + PD"))
r = pd.DataFrame(rows, columns=["variant","n","brier","skill","last_coef"])
print("ALL SEASONS 2019-2026 walk-forward"); print(r.round(5).to_string())
rows=[run(BASE, S2, sub.merge(df2[["game_id","passer_player_id","log_risk_K200"]], on=["game_id","passer_player_id"]), "current (2024-26)"),
      run(NOQB+["log_qb_pd","log_risk_K200"], S2, sub.merge(df2[["game_id","passer_player_id","log_risk_K200"]], on=["game_id","passer_player_id"]), "blended K=200 + PD (2024-26)")]
print(pd.DataFrame(rows, columns=["variant","n","brier","skill","last_coef"]).round(5).to_string())
