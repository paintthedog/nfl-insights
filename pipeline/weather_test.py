import numpy as np, pandas as pd, statsmodels.api as sm
s = pd.read_parquet("data/sched.parquet"); s = s[(s.season>=2016)&(s.game_type=="REG")&s.result.notna()]
p = pd.concat([pd.read_parquet(f"data/pbp_{y}.parquet", columns=["game_id","posteam","pass_attempt","sack","complete_pass","passing_yards","rushing_yards","rush_attempt","interception","two_point_attempt","season_type","play_type","air_yards"]) for y in range(2016,2027)])
p = p[(p.season_type=="REG")&(p.two_point_attempt!=1)&p.posteam.notna()].reset_index(drop=True)
p["att"] = ((p.pass_attempt==1)&(p.sack!=1)).astype(int); p["ay"] = np.where(p.att==1, p.air_yards, np.nan); p["isplay"] = p.play_type.isin(["pass","run"]).astype(int)
g = p.groupby(["game_id","posteam"]).agg(att=("att","sum"), sacks=("sack","sum"), cmp=("complete_pass","sum"), pyds=("passing_yards","sum"), ryds=("rushing_yards","sum"),
    car=("rush_attempt","sum"), ints=("interception","sum"), plays=("isplay","sum"), ay=("ay","mean")).reset_index()
g = g.merge(s[["game_id","roof","wind","temp","total_line","spread_line","home_team"]], on="game_id")
g["outdoor"] = ~g.roof.isin(["dome","closed"]); g["wind"] = np.where(g.outdoor, g.wind, 0); g["temp"] = np.where(g.outdoor, g.temp, 70)
g = g[g.wind.notna()&g.temp.notna()&(g.att>0)].copy()
g["ypa"] = g.pyds/g.att; g["cmp_pct"] = g.cmp/g.att; g["int_rate"] = g.ints/g.att; g["sack_rate"] = g.sacks/(g.att+g.sacks); g["ypc"] = g.ryds/g.car.replace(0,np.nan); g["pass_share"] = g.att/(g.att+g.car)
g["wbin"] = pd.cut(g.wind, [-1, 0.5, 9, 14, 19, 60], labels=["dome/0", "1-9", "10-14", "15-19", "20+"])
pd.set_option("display.width", 200)
print("Per team-game by wind, 2016-2026:")
print(g.groupby("wbin", observed=True).agg(n=("att","size"), att=("att","mean"), ypa=("ypa","mean"), cmp=("cmp_pct","mean"), int_rate=("int_rate","mean"),
      sack=("sack_rate","mean"), ypc=("ypc","mean"), pass_share=("pass_share","mean"), adot=("ay","mean"), total_line=("total_line","mean")).round(3).to_string())
g["tbin"] = pd.cut(g.temp, [-20, 25, 40, 55, 75, 120], labels=["<25", "25-40", "40-55", "55-75", "75+"])
print("\nBy temperature (outdoor only):")
print(g[g.outdoor].groupby("tbin", observed=True).agg(n=("att","size"), ypa=("ypa","mean"), int_rate=("int_rate","mean"), ypc=("ypc","mean"), cmp=("cmp_pct","mean"), att=("att","mean")).round(3).to_string())
print("\nControlled for the total line (market already prices some weather):")
for y, lab in [("ypa","yards/att"), ("int_rate","INT rate"), ("cmp_pct","completion %"), ("sack_rate","sack rate"), ("ypc","yards/carry"), ("pass_share","pass share"), ("att","attempts"), ("ay","aDOT")]:
    X = sm.add_constant(pd.DataFrame({"wind": g.wind, "wind15": (g.wind-15).clip(lower=0), "cold": (g.temp<35).astype(int), "total_line": g.total_line.fillna(45)}))
    m = sm.OLS(g[y], X, missing="drop").fit()
    print(f"  {lab:13s} per mph {m.params['wind']:+.4f} (t {m.tvalues['wind']:+.1f}) | extra per mph >15 {m.params['wind15']:+.4f} (t {m.tvalues['wind15']:+.1f}) | cold<35F {m.params['cold']:+.4f} (t {m.tvalues['cold']:+.1f})")
g.to_pickle("/tmp/claude-0/-home-claude/61a3e72b-dbd0-5c34-8049-91e3f7bfb7b5/scratchpad/weather_tg.pkl")
