"""Fit on all history through the latest completed week and project the upcoming slate."""
import sys, numpy as np, pandas as pd
import int_model as M

SEASON, WEEK = 2026, int(__import__("os").environ.get("NFL_WEEK", 5))
# Starter overrides verified against injury news (team -> passer_player_id).
# Everyone else defaults to the team's most recent starter.
import json, os
OVERRIDES = json.load(open("qb_overrides.json")).get(str(WEEK), {}) if os.path.exists("qb_overrides.json") else {}   # Jalon Daniels (Mayfield out); CHI defaults to Keenum (last starter, listed on schedule)  # Tyson Bagent (books list him; cleared concussion protocol)

pbp, sched = M.load()
qb, tg, ctx = M.build_tables(pbp, sched)

# ---- upcoming starters
recent = qb[qb.is_starter].merge(ctx[["game_id", "gameday"]].drop_duplicates(), on="game_id")
last = recent.sort_values("gameday").groupby("posteam").tail(1).set_index("posteam")
names = qb.drop_duplicates("passer_player_id").set_index("passer_player_id").name
games = sched[(sched.season == SEASON) & (sched.week == WEEK) & sched.result.isna()]   # unplayed games only
rows = []
for _, g in games.iterrows():
    for tm, opp in [(g.away_team, g.home_team), (g.home_team, g.away_team)]:
        pid = OVERRIDES.get(tm, last.loc[tm, "passer_player_id"])
        rows.append(dict(game_id=g.game_id, season=SEASON, week=WEEK,
                         gameday=pd.to_datetime(g.gameday), posteam=tm, defteam=opp,
                         passer_player_id=pid, name=names[pid]))
pred_rows = pd.DataFrame(rows)
# flag teams whose starter changed in their last game, or who are overridden: the default (last game's starter) may be wrong
_st = recent.sort_values("gameday").groupby("posteam").passer_player_id.apply(lambda s: s.tolist()[-2:])
_notes = {tm: "starter changed last game · confirm" for tm, v in _st.items() if len(v) == 2 and v[0] != v[1] and tm in set(pred_rows.get("posteam", []))}
_notes.update({tm: "expected back from injury · confirm" for tm in OVERRIDES if tm in set(pred_rows.get("posteam", []))})
if os.path.exists(f"news_wk{WEEK}.json"): _notes.update({t: n for t, n in json.load(open(f"news_wk{WEEK}.json")).get("qb_notes", {}).items() if t in set(pred_rows.posteam)})
json.dump(_notes, open(f"starter_notes_wk{WEEK}.json", "w"))

df = M.prep(M.build_features(qb, tg, ctx, predict_rows=pred_rows))
hist = df[~df._pred & (df.prior_games >= 1) & df.att.notna() & (df.att >= 10)]
fut = df[df._pred]

am = M.fit_attempts(hist)
im = M.fit_ints(hist, am)
out = M.predict(fut, am, im)

out["fair_over"] = M.american(out.p_int)
out["fair_under"] = M.american(1 - out.p_int)
out["qb_int_pct"] = out.qb_int_rate * 100
out["qb_iw_pct"] = out.qb_iw_rate * 100
out["qb_pd_pct"] = out.qb_pd_rate * 100
out["matchup"] = np.where(out.home == 1, "vs " + out.defteam, "@ " + out.defteam)

# Historical reference: how often ANY starter threw 1+ INT at this point spread
# (2016 on). The model already uses the spread; this shows how far each QB's
# own profile moves him off a generic QB in the same spot.
from spread_history import bucket
h = df[~df._pred & df.att.notna() & (df.att >= 10) & df.exp_margin.notna()]
hist_by_spread = (h.assign(bucket=bucket(h.exp_margin), hit=(h.ints >= 1))
                  .groupby("bucket", observed=True).hit.mean())
out["spread_bucket"] = bucket(out.exp_margin).astype(str)
out["hist_p_at_spread"] = out.spread_bucket.map(hist_by_spread.rename(index=str))
out["qb_vs_spread_hist"] = out.p_int - out.hist_p_at_spread

cols = ["name", "passer_player_id", "posteam", "matchup", "gameday", "exp_margin", "spread_bucket",
        "hist_p_at_spread", "proj_att", "qb_int_pct", "qb_iw_pct", "qb_pd_pct", "qb_adot", "def_int_rate",
        "exp_ints", "p_int", "qb_vs_spread_hist", "fair_over", "fair_under", "prior_att"]
res = out[cols].sort_values("p_int", ascending=False).reset_index(drop=True)
res.to_csv(f"int_props_{SEASON}_wk{WEEK}.csv", index=False)

pd.set_option("display.width", 200)
print(res.assign(gameday=res.gameday.dt.strftime("%a %m/%d")).round(
    {"exp_margin": 1, "proj_att": 1, "qb_int_pct": 2, "qb_iw_pct": 2, "qb_pd_pct": 2, "qb_adot": 1,
     "def_int_rate": 4, "exp_ints": 2, "p_int": 3, "fair_over": 0, "fair_under": 0,
     "hist_p_at_spread": 3, "qb_vs_spread_hist": 3})
      [["name", "matchup", "qb_int_pct", "qb_iw_pct", "qb_pd_pct", "hist_p_at_spread", "p_int",
        "qb_vs_spread_hist", "fair_over", "fair_under"]].to_string())
print("\nINT model coefficients:\n", im.params.round(4))
