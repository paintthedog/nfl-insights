"""Expected TDs from where touches happened (league TD rate by field position & play type), vs actual.
Then: does season-to-date (actual - expected) predict NEXT game's TD, beyond the model? Walk-forward 2019-2025."""
import numpy as np, pandas as pd
cols = ["game_id","season","week","season_type","posteam","rusher_player_id","receiver_player_id","rush_attempt","pass_attempt","sack",
        "yardline_100","rush_touchdown","pass_touchdown","two_point_attempt","qb_kneel","complete_pass"]
p = pd.concat([pd.read_parquet(f"data/pbp_{y}.parquet", columns=cols) for y in range(2016, 2027)])
p = p[(p.season_type=="REG")&(p.two_point_attempt!=1)].copy()
car = p[(p.rush_attempt==1)&p.rusher_player_id.notna()&(p.qb_kneel!=1)].copy(); car["pid"]=car.rusher_player_id; car["kind"]="rush"; car["td"]=car.rush_touchdown.fillna(0)
tgt = p[(p.pass_attempt==1)&(p.sack!=1)&p.receiver_player_id.notna()].copy(); tgt["pid"]=tgt.receiver_player_id; tgt["kind"]="tgt"; tgt["td"]=tgt.pass_touchdown.fillna(0)
t = pd.concat([car, tgt])[["game_id","season","week","posteam","pid","kind","yardline_100","td"]]
t["bin"] = pd.cut(t.yardline_100, [0,1,2,3,5,7,10,15,20,30,40,50,100], include_lowest=True)
# league TD rate by (kind, field-position bin), from 2016-2018 as a fixed table (then test on later seasons)
rate = t[t.season<=2018].groupby(["kind","bin"], observed=True).td.mean()
t["xtd"] = t.set_index(["kind","bin"]).index.map(rate).astype(float)
pg = t.groupby(["season","week","game_id","posteam","pid"], as_index=False).agg(td=("td","sum"), xtd=("xtd","sum"), touches=("td","size"))
pg = pg.sort_values(["pid","season","week"])
# season-to-date before this game
g = pg.groupby(["pid","season"])
pg["td_std"] = g.td.cumsum() - pg.td; pg["xtd_std"] = g.xtd.cumsum() - pg.xtd; pg["games_std"] = g.cumcount()
pg["due"] = pg.xtd_std - pg.td_std                     # positive = fewer TDs than usage suggests
pg["hit"] = (pg.td>=1).astype(int)
test = pg[(pg.season>=2019)&(pg.games_std>=3)&(pg.xtd_std>=0.8)].copy()
print(f"player-games tested: {len(test)}")
test["due_bin"] = pd.cut(test.due, [-9,-2,-1,-0.4,0.4,1,2,9], labels=["way over (-2+)","-2 to -1","-1 to -0.4","neutral","+0.4 to 1","+1 to 2","way due (+2)"])
import statsmodels.api as sm
# control for this game's expected TDs (usage in the game itself = what the TD model knows) + season xTD rate
X = sm.add_constant(pd.DataFrame({"xtd_game": test.xtd, "xtd_rate": test.xtd_std/test.games_std, "due": test.due}))
m = sm.GLM(test.hit, X, family=sm.families.Binomial()).fit()
print("\nLogit of scoring this game ~ this game's expected TDs + season xTD/game + DUE:")
print(f"  due coefficient {m.params['due']:+.4f} (t={m.tvalues['due']:+.2f})  -> each +1 TD 'owed' changes odds by {100*(np.exp(m.params['due'])-1):+.1f}%")
print("\nActual TD rate next game by how due (controls not applied, raw):")
print(test.groupby("due_bin", observed=True).agg(n=("hit","size"), scored=("hit","mean"), xtd_this_game=("xtd","mean")).round(3).to_string())
# and the cleaner version: residual = hit - P(hit | xtd this game)
test["p_base"] = 1-np.exp(-test.xtd)
print("\nScored minus expected-from-this-game's-usage, by due bin (0 = no extra info):")
print(test.groupby("due_bin", observed=True).apply(lambda z: round((z.hit - z.p_base).mean()*100, 1)).to_string())

# ---- Is the "over" effect just red-zone usage the field-position table misses, or real finishing skill? Check within position and vs prior-season.
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id","position"]).drop_duplicates("gsis_id").set_index("gsis_id").position
test["pos"] = test.pid.map(pl)
for pos in ["RB","WR","TE"]:
    z = test[test.pos==pos]
    X = sm.add_constant(pd.DataFrame({"xtd_game": z.xtd, "xtd_rate": z.xtd_std/z.games_std, "due": z.due}))
    m = sm.GLM(z.hit, X, family=sm.families.Binomial()).fit()
    print(f"{pos}: due coef {m.params['due']:+.4f} (t={m.tvalues['due']:+.1f}), n={len(z)}")
# persistence: a player's (actual - expected) one season vs the next
ps = pg.groupby(["pid","season"]).agg(td=("td","sum"), xtd=("xtd","sum"), g=("td","size")).reset_index()
ps = ps[(ps.g>=8)&(ps.xtd>=2)]; ps["over"] = ps.td-ps.xtd
ps["next"] = ps.groupby("pid").over.shift(-1)
c = ps.dropna(subset=["next"])
print(f"\nYear-to-year correlation of (actual - expected TDs): {c.over.corr(c.next):.2f}  (n={len(c)} player-seasons)")
