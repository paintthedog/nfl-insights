"""When a team's lead back (45%+ decayed carry share) misses a game, where do his carries go?
Compare: proportional renormalization (current model) vs giving the top remaining back a bonus slice of the
absent lead's share (alpha), walk-forward 2019-2026."""
import glob, numpy as np, pandas as pd
cols = ["game_id","season","week","season_type","posteam","rusher_player_id","receiver_player_id","play_type","qb_scramble","two_point_attempt"]
p = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
p = p[(p.season_type=="REG") & (p.two_point_attempt!=1)]
car = p[(p.play_type=="run") & (p.qb_scramble!=1) & p.rusher_player_id.notna()].groupby(["game_id","season","week","posteam","rusher_player_id"]).size().reset_index(name="car")
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id","position"]).drop_duplicates("gsis_id").set_index("gsis_id").position
car["pos"] = car.rusher_player_id.map(pl)
rb = car[car.pos.isin(["RB","FB"])].copy()
sched = pd.read_parquet("data/sched.parquet")[["game_id","gameday"]]
rb = rb.merge(sched, on="game_id"); rb["gameday"]=pd.to_datetime(rb.gameday)
rows=[]
D=0.8
for (tm,yr), g in rb.groupby(["posteam","season"]):
    games = g.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()
    hist={}  # pid -> list of (gidx, share)
    tmcar = g.groupby("game_id").car.sum()
    for i,gid in enumerate(games):
        cur = g[g.game_id==gid].set_index("rusher_player_id").car
        shares = {pid: float(c/tmcar[gid]) for pid,c in cur.items()}
        if i>=3:
            # decayed share entering this game
            dec={}
            for pid,h in hist.items():
                w=v=0.0
                for gi,sh in h[-6:]:
                    wt=D**(i-1-gi); w+=wt; v+=wt*sh
                dec[pid]=v/w if w else 0.0
            lead=max(dec, key=dec.get)
            if dec[lead]>=0.45:
                played = set(cur.index)
                lead_played_last = any(gi==i-1 for gi,_ in hist[lead])
                if lead not in played and lead_played_last:
                    others={pid:s for pid,s in dec.items() if pid!=lead and s>=0.03}
                    if others:
                        rows.append((yr,gid,tm,dec[lead],others,shares))
        for pid,c in cur.items(): hist.setdefault(pid,[]).append((i,float(c/tmcar[gid])))
print("lead-back-absent games:", len(rows))
def err(alpha, data):
    es=[]; etop=[]
    for yr,gid,tm,lsh,others,act in data:
        top=max(others,key=others.get); tot=sum(others.values())
        for pid,s in others.items():
            base = s/tot*(1-0) if False else s/ (tot) * (tot+lsh)  # proportional: scale all up by freed share
            pred = s + alpha*lsh*(1 if pid==top else 0) + (1-alpha)*lsh*(s/tot)
            if alpha<0: pred = s/(tot)*(tot+lsh)
            a = act.get(pid,0.0)
            es.append(abs(pred-a))
            if pid==top: etop.append(abs(pred-a))
    return np.mean(es), np.mean(etop)
import collections
data=rows
for alpha in [-1,0,0.25,0.5,0.75,1.0]:
    lab = "proportional" if alpha<0 else f"alpha={alpha}"
    # walk-forward not needed for a structural parameter this coarse; but split halves for honesty
    tr=[r for r in data if r[0]<=2022]; te=[r for r in data if r[0]>2022]
    e_tr=err(alpha,tr); e_te=err(alpha,te)
    print(f"{lab:13} MAE share all/topRB  train {e_tr[0]:.4f}/{e_tr[1]:.4f}   test {e_te[0]:.4f}/{e_te[1]:.4f}")
# what does the top remaining back actually average?
tops=[(max(o,key=o.get),o,l,a) for _,_,_,l,o,a in data]
own=np.mean([o[t] for t,o,l,a in tops]); got=np.mean([a.get(t,0) for t,o,l,a in tops]); lead=np.mean([l for _,_,l,_ in tops])
print(f"top remaining back: prior share {own:.3f}, lead's absent share {lead:.3f}, actual share when lead out {got:.3f}")
# where do the absent lead's carries actually go? split the post-absence team shares into:
# the top remaining known back / other known backs / backs who were NOT in the decayed pool (newcomers)
splits=[]
for yr,gid,tm,lsh,others,act in data:
    top=max(others,key=others.get)
    known=set(others); tot_act=sum(act.values())
    splits.append((act.get(top,0), sum(v for k,v in act.items() if k in known and k!=top),
                   sum(v for k,v in act.items() if k not in known)))
s=np.array(splits)
print(f"\nafter the lead sits, team RB carries split: top known back {s[:,0].mean():.3f}, other known {s[:,1].mean():.3f}, newcomers outside the pool {s[:,2].mean():.3f}")
print(f"newcomers take 10%+ of carries in {np.mean(s[:,2]>=.10):.0%} of these games; 25%+ in {np.mean(s[:,2]>=.25):.0%}")
