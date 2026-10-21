"""TDs allowed per game by position, season to date -> data/td_dvp.parquet."""
import pandas as pd

pbp = pd.read_parquet("data/pbp_2026.parquet",
                      columns=["defteam", "td_player_id", "rush_touchdown", "pass_touchdown",
                               "game_id", "two_point_attempt"])
td = pbp[(pbp.rush_touchdown.eq(1) | pbp.pass_touchdown.eq(1))
         & pbp.two_point_attempt.ne(1) & pbp.td_player_id.notna() & pbp.defteam.notna()]

pos = pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]).set_index("gsis_id").position
td = td.assign(pos=td.td_player_id.map(pos).replace({"FB": "RB", "HB": "RB"}))
td = td[td.pos.isin(["QB", "RB", "WR", "TE"])]

games = pbp.groupby("defteam").game_id.nunique()
cnt = td.groupby(["defteam", "pos"]).size().unstack(fill_value=0)
for p in ["QB", "RB", "WR", "TE"]:
    if p not in cnt: cnt[p] = 0
out = (cnt[["QB", "RB", "WR", "TE"]].div(games, axis=0)
       .stack().rename("tdpg").reset_index().rename(columns={"level_1": "pos", "defteam": "team"}))
out["rank"] = out.groupby("pos").tdpg.rank(ascending=False, method="min").astype(int)   # 1 = most allowed (softest)
out["lg"] = out.groupby("pos").tdpg.transform("mean")
out.to_parquet("data/td_dvp.parquet", index=False)
print(out.sort_values(["pos", "rank"]).groupby("pos").head(2).to_string(index=False))
