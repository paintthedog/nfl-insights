"""Expected TDs per player-game from where touches happened (league rate by play type x field position),
plus season-to-date actual vs expected. Writes data/xtd.parquet: game_id, pid, td, xtd, and prior-to-this-game season totals."""
import numpy as np, pandas as pd
cols = ["game_id","season","week","season_type","posteam","rusher_player_id","receiver_player_id","rush_attempt","pass_attempt","sack",
        "yardline_100","rush_touchdown","pass_touchdown","two_point_attempt","qb_kneel"]
p = pd.concat([pd.read_parquet(f"data/pbp_{y}.parquet", columns=cols) for y in range(2016, 2027)])
p = p[(p.season_type=="REG")&(p.two_point_attempt!=1)].copy()
car = p[(p.rush_attempt==1)&p.rusher_player_id.notna()&(p.qb_kneel!=1)].copy(); car["pid"]=car.rusher_player_id; car["kind"]="rush"; car["td"]=car.rush_touchdown.fillna(0)
tgt = p[(p.pass_attempt==1)&(p.sack!=1)&p.receiver_player_id.notna()].copy(); tgt["pid"]=tgt.receiver_player_id; tgt["kind"]="tgt"; tgt["td"]=tgt.pass_touchdown.fillna(0)
t = pd.concat([car, tgt])[["game_id","season","week","posteam","pid","kind","yardline_100","td"]]
t["bin"] = pd.cut(t.yardline_100, [0,1,2,3,5,7,10,15,20,30,40,50,100], include_lowest=True)
rate = t.groupby(["kind","bin"], observed=True).td.mean()          # league-wide table, all seasons (stable)
t["xtd"] = t.set_index(["kind","bin"]).index.map(rate).astype(float)
pg = t.groupby(["season","week","game_id","posteam","pid"], as_index=False).agg(td=("td","sum"), xtd=("xtd","sum"))
pg = pg.sort_values(["pid","season","week"])
g = pg.groupby(["pid","season"])
pg["td_std"] = g.td.cumsum() - pg.td; pg["xtd_std"] = g.xtd.cumsum() - pg.xtd; pg["games_std"] = g.cumcount()
# season totals AFTER each game (for display: "so far this season")
pg["td_season"] = g.td.cumsum(); pg["xtd_season"] = g.xtd.cumsum(); pg["games_season"] = g.cumcount() + 1
pg.to_parquet("data/xtd.parquet")
print(len(pg), "player-games;", "rate table:", {k: round(v, 3) for k, v in rate.items() if k[0]=="rush"} )
