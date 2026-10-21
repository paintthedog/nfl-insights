"""Same-game stacks for the paper record (Oct 18). Usage: python sgp_slate.py W   (appends to paper_wk<W>.csv; paper_slate.py also calls it)
Fixed rules, nothing hand-picked. For every game not yet started, the DraftKings moneyline favorite gets two stacks:
  A  "favorite + lead RB"          favorite wins  +  its lead RB (most projected carries) over a rushing-yards rung
  B  "top two pass catchers"       its top two pass catchers by projected targets (WR or TE), each over a receptions rung
Each player leg uses the DraftKings alternate rung priced closest to 1.40 (about a 70% leg).
Leg chances are the site's blend (20% model, 80% DraftKings, in log-odds); the moneyline leg uses DK's no-vig price blended the same way.
model_p = joint chance from parlay_joint (historical same-game correlations); indep_p = the legs multiplied as if unrelated.
price = the legs' DraftKings prices multiplied. That is a REFERENCE ONLY: DraftKings pays less than this on a same-game parlay,
and the Odds API does not carry same-game prices, so these are graded on hit rate vs the joint chance, not on units."""
import sys, json, re
from math import erf, sqrt, log, exp
import numpy as np, pandas as pd
from parlay_joint import joint

W = int(sys.argv[1]) if __name__ == "__main__" else None
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
lg = lambda p: log(p / (1 - p)); ilg = lambda x: 1 / (1 + exp(-x))
blend = lambda pm, pk: ilg(0.2 * lg(min(max(pm, .01), .99)) + 0.8 * lg(min(max(pk, .01), .99)))
TYPE_A, TYPE_B = "Same-game stacks: favorite + lead RB", "Same-game stacks: top two pass catchers"


def build(W, now=None):
    D = json.load(open(f"site/data_wk{W}.json")); L = json.load(open(f"lines_2026_wk{W}.json")); QS = D["proj"]["quants"]
    sched = pd.read_parquet("data/sched.parquet"); sw = sched[(sched.season == 2026) & (sched.week == W)].set_index("game_id")
    now = now or pd.Timestamp.now(tz="America/New_York")
    kick = lambda gid: pd.Timestamp(f"{sw.loc[gid].gameday} {sw.loc[gid].gametime}").tz_localize("America/New_York")

    def p_over(arr, line):
        arr = np.asarray(arr, float)
        if line < arr[0]: return 1 - QS[0] * (line / max(arr[0], 1e-9))
        for i in range(1, len(arr)):
            if line <= arr[i]:
                t = .5 if arr[i] == arr[i - 1] else (line - arr[i - 1]) / (arr[i] - arr[i - 1]); return 1 - (QS[i - 1] + t * (QS[i] - QS[i - 1]))
        return max(0, (1 - QS[-1]) * .5)

    M = pd.read_csv("market_log.csv"); M = M[M.week == W].sort_values("pulled_at").drop_duplicates(["game_id", "market", "subject", "side", "point"], keep="last")
    P = D["proj"]["players"]; SDM = L["sd_margin"]; rows = []
    for g in L["games"]:
        gid = g["game_id"]
        if gid not in sw.index or kick(gid) <= now: continue
        h = M[(M.game_id == gid) & (M.market == "game_h2h")]
        hp, ap = h[h.subject == g["home"]], h[h.subject == g["away"]]
        if not len(hp) or not len(ap): continue
        hpr, apr = float(hp.price.iloc[0]), float(ap.price.iloc[0])
        fav, fpr = (g["home"], hpr) if hpr <= apr else (g["away"], apr)
        nv = (1 / fpr) / (1 / hpr + 1 / apr); pm = Phi(g["m_margin"] / SDM); pm = pm if fav == g["home"] else 1 - pm
        win = dict(gid=gid, player=fav, stat="ml", line=None, price=fpr, p=round(blend(pm, nv), 4), label="WIN")
        tp = [p for p in P if p["team"] == fav and (p.get("p_play") or 1) >= 0.85]

        def rung(p, market, st):
            r = M[(M.game_id == gid) & (M.market == market) & (M.side == "Over")]
            r = r[r.subject.map(norm) == norm(p["name"])]
            q = (p.get("q") or {}).get(st)
            if not len(r) or not q: return None
            r = r.iloc[(r.price - 1.40).abs().argsort()[:1]].iloc[0]
            if not 1.15 <= r.price <= 1.80: return None
            return dict(gid=gid, player=p["name"], stat=st, line=float(r.point), price=float(r.price), p=round(blend(p_over(q, float(r.point)), 1 / r.price / 1.05), 4))

        rbs = sorted([p for p in tp if p["pos"] == "RB"], key=lambda p: -(p["mean"].get("car") or 0))
        if rbs:
            lg_ = rung(rbs[0], "player_rush_yds_alternate", "rush_yds")
            if lg_: rows.append((TYPE_A, gid, [win, dict(lg_, label="RB1 rush_yds")]))
        pc = sorted([p for p in tp if p["pos"] in ("WR", "TE")], key=lambda p: -(p["mean"].get("tgt") or 0))
        wr = [p for p in pc if p["pos"] == "WR"]; te = [p for p in pc if p["pos"] == "TE"]
        legs = []
        for p in pc[:2]:
            role = "WR1" if wr and p is wr[0] else "TE1" if te and p is te[0] else "WR2"
            l_ = rung(p, "player_receptions_alternate", "rec")
            if l_: legs.append(dict(l_, label=f"{role} rec"))
        if len(legs) == 2 and legs[0]["label"] != legs[1]["label"]: rows.append((TYPE_B, gid, legs))
    out = []
    stamp = now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%MZ")
    for t, gid, legs in rows:
        jp = joint([(l["label"], l["p"]) for l in legs]); ip = float(np.prod([l["p"] for l in legs]))
        subj = " + ".join(f"{l['player']} wins" if l["stat"] == "ml" else f"{l['player']} {l['line']:g}+ {'rec' if l['stat'] == 'rec' else 'rush yds'}" for l in legs)
        out.append(dict(week=W, type=t, game_id=gid, subject=subj, market="parlay", side="All", line=None, price=round(float(np.prod([l["price"] for l in legs])), 3),
                        model_p=round(jp, 4), legs=json.dumps([{k: v for k, v in l.items()} for l in legs]), placed_at=stamp, indep_p=round(ip, 4)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    new = build(W)
    f = f"paper_wk{W}.csv"; old = pd.read_csv(f)
    keep_gids = set(new.game_id) if len(new) else set()
    old = old[~(old.type.isin([TYPE_A, TYPE_B]) & old.game_id.isin(keep_gids))]   # re-running before kickoff replaces only games not yet started
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
    print(f"{f}: +{len(new)} same-game stacks")
    if len(new): print(new[["type", "subject", "price", "model_p", "indep_p"]].to_string())
