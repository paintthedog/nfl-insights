"""Standing rule (Clinton, Oct 18): "ML + top TD scorer, 2 legs". For every game not yet started: the model's projected winner on the
moneyline + the game's single likeliest anytime-TD scorer (either team), as one two-leg same-game parlay.
Usage: python ml_td_slate.py W [--paper]   (paper_slate.py calls build(W) on game mornings)
Leg chances: moneyline = model win chance blended 20/80 with DK's no-vig price; TD = the site's blended chance. Joint chance =
the two multiplied, then nudged by the historical win-vs-lead-scorer link from parlay_joint (RB1 rush_yds ~ WIN) only when the
scorer is on the picked team; a scorer on the other team is left independent. Price = the two DK prices multiplied (a REFERENCE:
DK pays less on a same-game parlay). Graded by hit rate and vs the reference price."""
import sys, json, re
from math import erf, sqrt, log, exp
import numpy as np, pandas as pd

TYPE = "ML + top TD scorer, 2 legs"
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
lg = lambda p: log(p / (1 - p)); ilg = lambda x: 1 / (1 + exp(-x))
blend = lambda pm, pk: ilg(0.2 * lg(min(max(pm, .01), .99)) + 0.8 * lg(min(max(pk, .01), .99)))
STACK_LIFT = 1.08     # P(win & own lead back scores) / (P(win) * P(scores)), from the copula's RB1-WIN link at ~60% legs


def build(W, now=None, picks=None, stamp=None):
    """picks: optional {game_id: (team, scorer)} to log exactly what was chosen (backfill); otherwise the rule picks."""
    D = json.load(open(f"site/data_wk{W}.json")); L = json.load(open(f"lines_2026_wk{W}.json"))
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
    now = now or pd.Timestamp.now(tz="America/New_York")
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")
    M = pd.read_csv("market_log.csv"); M = M[M.week == W].sort_values("pulled_at").drop_duplicates(["game_id", "market", "subject", "side", "point"], keep="last")
    TD = {(r["team"], norm(r["name"])): r for r in D["td"]}
    SDM = L["sd_margin"]; out = []
    for g in L["games"]:
        gid = g["game_id"]
        if gid not in sw.index: continue
        if picks is None and kick(gid) <= now: continue
        h = M[(M.game_id == gid) & (M.market == "game_h2h")]
        ph_m = Phi(g["m_margin"] / SDM)
        if picks and gid in picks: team, scorer = picks[gid]
        elif picks: continue
        else:
            team = g["home"] if ph_m >= 0.5 else g["away"]
            cands = [r for r in D["td"] if r["team"] in (g["home"], g["away"]) and r.get("dk")]
            if not cands: continue
            scorer = max(cands, key=lambda r: r["p"])["name"]
        pr = h[h.subject == team].price; opp = h[h.subject != team].price
        if not len(pr) or not len(opp): print("no moneyline for", gid); continue
        pr, opp = float(pr.iloc[0]), float(opp.iloc[0]); nv = (1 / pr) / (1 / pr + 1 / opp)
        pm = ph_m if team == g["home"] else 1 - ph_m; p_win = blend(pm, nv)
        r = TD.get((g["home"], norm(scorer))) or TD.get((g["away"], norm(scorer)))
        if not r or not r.get("dk"): print("no TD price for", scorer, gid); continue
        p_td = blend(r["p"], 1 / r["dk"] / 1.07)
        joint = p_win * p_td * (STACK_LIFT if r["team"] == team else 1.0)
        legs = [dict(gid=gid, player=team, stat="ml", line=None, price=pr, p=round(p_win, 4), label="WIN"),
                dict(gid=gid, player=r["name"], stat="atd", line=None, price=float(r["dk"]), p=round(p_td, 4), label="ATD")]
        out.append(dict(week=W, type=TYPE, game_id=gid, subject=f"{team} wins + {r['name']} TD", market="parlay", side="All", line=None,
                        price=round(pr * r["dk"], 3), model_p=round(min(joint, 0.99), 4), legs=json.dumps(legs),
                        placed_at=stamp or now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ"), indep_p=round(p_win * p_td, 4)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    W = int(sys.argv[1]); new = build(W)
    print(new[["subject", "price", "model_p"]].to_string(index=False) if len(new) else "no games left to place")
    if "--paper" in sys.argv and len(new):
        f = f"paper_wk{W}.csv"; old = pd.read_csv(f); old = old[~((old.type == TYPE) & old.game_id.isin(set(new.game_id)))]
        pd.concat([old, new], ignore_index=True).to_csv(f, index=False); print(f"{f}: +{len(new)} rows")
