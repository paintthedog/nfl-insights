"""Grade the coach-quote adjustments (news_q in news_overrides; Nov 8). For every logged use: did the player take an offensive
snap? Compares the model's chance before the quote with the chance after it, by quote category. Usage: python news_q_grade.py
The steps in td_project.py (+1.5 / -1.5 log-odds, gtd halfway to 50%) are provisional until each category has ~40 graded cases."""
import os, pandas as pd, numpy as np
if not os.path.exists("news_q_log.csv"): print("no news_q_log.csv yet"); raise SystemExit
L = pd.read_csv("news_q_log.csv")
s = pd.read_parquet("data/snaps.parquet", columns=["season", "week", "gsis_id", "offense_snaps"])
s = s[s.offense_snaps.fillna(0) > 0][["season", "week", "gsis_id"]].drop_duplicates().assign(played=1)
sched = pd.read_parquet("data/sched.parquet"); done = set(zip(sched[sched.result.notna()].season, sched[sched.result.notna()].week))
L = L.merge(s, left_on=["season", "week", "pid"], right_on=["season", "week", "gsis_id"], how="left").fillna({"played": 0})
L = L[[(a, b) in done for a, b in zip(L.season, L.week)]]
if L.empty: print("no graded quotes yet"); raise SystemExit
L = L.drop_duplicates(["season", "week", "pid"], keep="last")
g = L.groupby("cat").apply(lambda x: pd.Series(dict(n=len(x), model_said=x.p_model.mean(), news_said=x.p_news.mean(), played=x.played.mean(),
        brier_model=((x.p_model - x.played) ** 2).mean(), brier_news=((x.p_news - x.played) ** 2).mean())))
print(g.round(3).to_string())
print(f"all: n={len(L)} Brier model {((L.p_model - L.played) ** 2).mean():.3f} vs with quotes {((L.p_news - L.played) ** 2).mean():.3f}")
