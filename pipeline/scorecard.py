"""Scorecard (Oct 18): grade EVERY number the site published against what happened, not just the paper bets.
Usage: python scorecard.py            -> scorecard.json (feeds the top of the Record tab)
For each finished game, uses the last archived snapshot (archive/data_wk<W>_<stamp>.json) saved before that game's kickoff,
so nothing is graded on numbers published after kickoff.
Three kinds of number, three tests:
  averages       projected stat vs actual (players projected to play, p_play >= 0.9, who did play): bias and average miss;
                 where DraftKings posted a line for the same stat, the line's average miss on the same players
  ranges         share of results below the published floor (10th pct, target 10%), above the ceiling (90th, target 10%),
                 inside the middle half (target 50%)
  probabilities  TD, interception, cover and over chances: what the model said on average vs how often it happened, by bucket,
                 plus the Brier score (lower is better) and DraftKings' Brier where DK priced the same thing."""
import glob, json, os, re, sys
from math import erf, sqrt
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import proj_model as PM

SEASON = 2026
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
from statistics import NormalDist
_qn = NormalDist().inv_cdf
T3 = []   # market vs raw model vs blend, per game and per priced TD (Oct 22)
W_PROP = 0.2   # same log-odds weight build_site.py gives the model when it blends TD chances with DraftKings
def _blend(pm, pk):
    pm, pk = min(max(pm, 0.01), 0.99), min(max(pk, 0.01), 0.99); lo = lambda p: np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-(W_PROP * lo(pm) + (1 - W_PROP) * lo(pk)))))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
pbp = pd.read_parquet(f"data/pbp_{SEASON}.parquet")
pbp = pbp[pbp.two_point_attempt.ne(1)]
done = set(pbp.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
sched = pd.read_parquet("data/sched.parquet"); sched = sched[sched.season == SEASON]
pg = PM.player_games(PM.load())[0]
pg = pg[pg.game_id.isin(done)].copy()
pg["fp_ppr"] = (pg.pass_yds * .04 + pg.pass_td * 4 - pg.ints * 2 + pg.rush_yds * .1 + pg.rush_td * 6 + pg.rec + pg.rec_yds * .1 + pg.rec_td * 6)
act = pg.set_index(["game_id", "pid"])
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"])
NM = {}
for g, n in zip(pl.gsis_id, pl.display_name): NM.setdefault(norm(n), []).append(g)
tds = pbp[pbp.touchdown.eq(1) & pbp.td_player_id.notna()].groupby(["game_id", "td_player_id"]).size()
ints = pbp.groupby(["game_id", "passer_player_id"]).interception.sum()
dropbacks = pbp[pbp.qb_dropback.eq(1)].groupby(["game_id", "passer_player_id"]).size()
touched = pbp.groupby("game_id").apply(lambda g: set(g.passer_player_id.dropna()) | set(g.rusher_player_id.dropna()) | set(g.receiver_player_id.dropna()))


def pid_in(name, gid):
    c = NM.get(norm(name), []); ids = touched.get(gid, set())
    h = [x for x in c if x in ids]
    return h[0] if h else None


def game_of(team, sw):
    m = sw[(sw.home_team == team) | (sw.away_team == team)]
    return m.game_id.iloc[0] if len(m) else None


AVG_STATS = ["pass_yds", "rush_yds", "car", "tgt", "rec", "rec_yds", "fp_ppr"]
RANGE_STATS = ["pass_yds", "rush_yds", "car", "tgt", "rec", "rec_yds", "fp_ppr"]
LINE_KEY = {"pass_yds": "pass_yds", "rush_yds": "rush_yds", "rec": "rec", "rec_yds": "rec_yds"}
QI = {q: i for i, q in enumerate(PM.QUANTS)}
A, R, PR, G = [], [], [], []
SNAP_GAMES = set()
_pre = lambda d: d[pd.to_datetime(d.pulled_at, utc=True, format="ISO8601") < pd.to_datetime(d.kickoff, utc=True, format="ISO8601")]
_lt = _pre(pd.read_csv("props_log.csv")); _lt = _lt[_lt.book == "draftkings"].sort_values("pulled_at")
DKTD = {(g, norm(n)): p for g, n, p in zip(_lt.game_id, _lt.player, _lt.price)}   # last pre-kickoff DK price, to compare against
weeks = sorted({int(re.search(r"wk(\d+)_", f).group(1)) for f in glob.glob("archive/data_wk*_*.json")})
for wk in weeks:
    sw = sched[(sched.week == wk) & sched.game_id.isin(done)]
    snaps = sorted(glob.glob(f"archive/data_wk{wk}_*.json"))
    stamp = lambda f: f.split("_")[-2] + f.split("_")[-1][:4]          # archive stamps are Eastern time, like the schedule
    pick = {r.game_id: ([f for f in snaps if stamp(f) < r.gameday.replace("-", "") + r.gametime.replace(":", "")] or [None])[-1] for r in sw.itertuples()}
    for snap in sorted(set(v for v in pick.values() if v)):            # each game graded off the last snapshot saved before ITS kickoff
        gs = sw[sw.game_id.map(pick) == snap]
        D = json.load(open(snap)); gids = set(gs.game_id); SNAP_GAMES.update(gids)
        # ---- players: averages and ranges
        for p in D["proj"]["players"]:
            gid = game_of(p["team"], gs)
            if gid not in gids or (gid, p["pid"]) not in act.index: continue
            a = act.loc[(gid, p["pid"])]; a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
            if (p.get("p_play") or 1) < 0.9: continue
            for st in AVG_STATS:
                if st in ("pass_yds",) and p["pos"] != "QB": continue
                if st in ("tgt", "rec", "rec_yds") and p["pos"] == "QB": continue
                m = (p.get("mean") or {}).get(st)
                if m is None or (st != "fp_ppr" and m < 0.5): continue
                ln = None
                L = (p.get("lines") or {}).get(LINE_KEY.get(st, "_"), {}) if isinstance(p.get("lines"), dict) else {}
                if isinstance(L, dict) and L.get("draftkings"): ln = L["draftkings"].get("pt")
                A.append(dict(wk=wk, pos=p["pos"], stat=st, proj=float(m), act=float(a[st]), line=ln))
            for st in RANGE_STATS:
                q = (p.get("q") or {}).get(st)
                if not q or max(q) <= 0: continue
                if st == "pass_yds" and p["pos"] != "QB": continue
                if st in ("tgt", "rec", "rec_yds") and p["pos"] == "QB": continue
                v = float(a[st]); lo, hi, q25, q75 = q[QI[0.1]], q[QI[0.9]], q[QI[0.25]], q[QI[0.75]]
                R.append(dict(wk=wk, pos=p["pos"], stat=st, below=1.0 if v < lo else (.5 if v == lo else 0.0), above=1.0 if v > hi else (.5 if v == hi else 0.0),
                              mid=float(q25 <= v <= q75)))
        # ---- touchdowns
        for r in (D["td"] if isinstance(D["td"], list) else D["td"].get("rows", [])):
            gid = game_of(r["team"], gs)
            if gid not in gids: continue
            pid = pid_in(r["name"], gid)
            if pid is None: continue                       # didn't play: no grade (same as a push)
            dk = r.get("dk") or DKTD.get((gid, norm(r["name"]))); PR.append(dict(wk=wk, mkt="Anytime TD", p=float(r["p"]), y=float(tds.get((gid, pid), 0) > 0), dk=(1 / dk / 1.07) if dk else None))
        # ---- interceptions (starters who dropped back 15+ times)
        for r in (D["int"] if isinstance(D["int"], list) else []):
            gid = game_of(r["team"], gs)
            if gid not in gids: continue
            pid = pid_in(r["full"], gid)
            if pid is None or dropbacks.get((gid, pid), 0) < 15: continue
            PR.append(dict(wk=wk, mkt="QB throws an interception", p=float(r["p"]), y=float(ints.get((gid, pid), 0) > 0), dk=None))
        # ---- games
        sd = D["lines"].get("sd_margin", 13.5)
        for g in D["lines"]["games"]:
            if g["game_id"] not in gids: continue
            s = sw.set_index("game_id").loc[g["game_id"]]; mg = float(s.home_score - s.away_score); tot = float(s.home_score + s.away_score)
            G.append(dict(wk=wk, m_margin=g["m_margin"], spread=g.get("spread"), mg=mg, m_total=g["m_total"], total=g.get("total"), tot=tot))
            # Oct 22: three numbers per game - market, the model's raw number, and the line-anchored blend the site shows
            if g.get("spread") is not None and g.get("p_home_cover") is not None:
                bm = g["spread"] + sd * _qn(min(max(g["p_home_cover"], 0.01), 0.99))
                T3.append(dict(wk=wk, kind="margin", market=g["spread"], model=g["m_margin"], blend=bm, act=mg, sd=sd))
            if g.get("total") is not None and g.get("p_over") is not None:
                sdt = D["lines"].get("sd_total", 13.0); bt = g["total"] + sdt * _qn(min(max(g["p_over"], 0.01), 0.99))
                T3.append(dict(wk=wk, kind="total", market=g["total"], model=g["m_total"], blend=bt, act=tot, sd=sdt))
            if g.get("spread") is not None and mg != -g["spread"]:
                PR.append(dict(wk=wk, mkt="Spread: home team covers", p=float(g["p_home_cover"]), y=float(mg + g["spread"] > 0), dk=0.5))
            if g.get("total") is not None and tot != g["total"]:
                PR.append(dict(wk=wk, mkt="Game goes over the total", p=float(g["p_over"]), y=float(tot > g["total"]), dk=0.5))
            PR.append(dict(wk=wk, mkt="Model's favorite wins", p=max(Phi(g["m_margin"] / sd), 1 - Phi(g["m_margin"] / sd)), y=float((mg > 0) == (g["m_margin"] > 0)) if mg != 0 else 0.5, dk=None))

# games published before the snapshot archive existed (Weeks 4-5): the odds pulls logged the model's TD chance next to DK's price before kickoff
pre_k = lambda d: d[pd.to_datetime(d.pulled_at, utc=True, format="ISO8601") < pd.to_datetime(d.kickoff, utc=True, format="ISO8601")]
lt = pre_k(pd.read_csv("props_log.csv")); lt = lt[(lt.season == SEASON) & (lt.book == "draftkings") & lt.game_id.isin(done) & ~lt.game_id.isin(SNAP_GAMES)]
lt = lt.sort_values("pulled_at").drop_duplicates(["game_id", "player"], keep="last")
for r in lt.itertuples():
    pid = pid_in(r.player, r.game_id)
    if pid is None or pd.isna(r.model_p): continue
    PR.append(dict(wk=int(r.week), mkt="Anytime TD", p=float(r.model_p), y=float(tds.get((r.game_id, pid), 0) > 0), dk=1 / r.price / 1.07))
    _y = float(tds.get((r.game_id, pid), 0) > 0); T3.append(dict(wk=int(r.week), kind="td", market=1 / r.price / 1.07, model=float(r.model_p), blend=_blend(float(r.model_p), 1 / r.price), act=_y))
# model's average vs DraftKings' line on the same player, same stat (the O/U pulls log the model's mean at pull time)
lo = pre_k(pd.read_csv("props_log_ou.csv")); lo = lo[(lo.season == SEASON) & (lo.book == "draftkings") & lo.game_id.isin(done)]
lo = lo.sort_values("pulled_at").drop_duplicates(["game_id", "player", "market"], keep="last")
OUST = {"player_receptions": "rec", "player_reception_yds": "rec_yds", "player_rush_yds": "rush_yds", "player_pass_yds": "pass_yds"}
LV = []
for r in lo.itertuples():
    pid = pid_in(r.player, r.game_id); st = OUST.get(r.market)
    if pid is None or st is None or (r.game_id, pid) not in act.index or pd.isna(r.model_mean): continue
    a = act.loc[(r.game_id, pid)]; a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
    LV.append(dict(wk=int(r.week), stat=st, model=float(r.model_mean), line=float(r.point), act=float(a[st])))
LV = pd.DataFrame(LV)
A, R, PR, G = pd.DataFrame(A), pd.DataFrame(R), pd.DataFrame(PR), pd.DataFrame(G)
if not len(A) and not os.environ.get("SC_TEST"):   # no pre-kickoff archives yet (archive/ was lost in the Oct 21 export): keep the last scorecard
    print("scorecard: no archived snapshots for finished games; keeping existing scorecard.json"); sys.exit(0)
if not len(A): A = pd.DataFrame(columns=["wk", "stat", "proj", "act", "line"])
if not len(R): R = pd.DataFrame(columns=["wk", "stat", "below", "above", "mid"])
LAB = {"pass_yds": "Passing yards", "rush_yds": "Rushing yards", "car": "Carries", "tgt": "Targets", "rec": "Receptions", "rec_yds": "Receiving yards", "fp_ppr": "Fantasy points (PPR)"}
out = dict(asof=pd.Timestamp.now(tz="America/New_York").strftime("%b %-d, %Y"), weeks=sorted(set(map(int, set(A.wk if len(A) else []) | set(PR.wk if len(PR) else []) | set(LV.wk if len(LV) else [])))),
           snap_weeks=sorted(set(int(g[5:7]) for g in SNAP_GAMES)), n_games=len(G))
vl = []
for st in ["rec", "rec_yds", "rush_yds", "pass_yds"]:
    x = LV[LV.stat == st] if len(LV) else LV
    if len(x) >= 10: vl.append(dict(stat=LAB[st], n=len(x), model=round((x.act - x.model).abs().mean(), 2), dk=round((x.act - x.line).abs().mean(), 2),
                                     closer=round(((x.act - x.model).abs() < (x.act - x.line).abs()).mean(), 3), weeks=sorted(set(map(int, x.wk)))))
out["vs_lines"] = vl
avg = []
for st in AVG_STATS:
    x = A[A.stat == st]
    if not len(x): continue
    y = x[x.line.notna()]
    avg.append(dict(stat=LAB[st], n=len(x), proj=round(x.proj.mean(), 1), act=round(x.act.mean(), 1), bias=round((x.act - x.proj).mean(), 2),
                    mae=round((x.act - x.proj).abs().mean(), 2),
                    n_line=len(y), mae_model_l=round((y.act - y.proj).abs().mean(), 2) if len(y) >= 10 else None, mae_dk=round((y.act - y.line).abs().mean(), 2) if len(y) >= 10 else None))
out["averages"] = avg
rng = []
for st in RANGE_STATS:
    x = R[R.stat == st]
    if len(x): rng.append(dict(stat=LAB[st], n=len(x), below=round(x.below.mean(), 3), above=round(x.above.mean(), 3), mid=round(x.mid.mean(), 3)))
out["ranges"] = rng
probs = []
EDGES = {"Anytime TD": [0, .1, .2, .3, .4, .5, 1.01], "QB throws an interception": [0, .3, .4, .5, .6, 1.01],
         "Spread: home team covers": [0, .45, .5, .55, 1.01], "Game goes over the total": [0, .45, .5, .55, 1.01], "Model's favorite wins": [.5, .6, .7, .8, 1.01]}
for mkt, e in EDGES.items():
    x = PR[PR.mkt == mkt]
    if not len(x): continue
    b = pd.cut(x.p, e, right=False)
    bk = [dict(lo=round(iv.left, 2), hi=round(min(iv.right, 1), 2), n=int(len(g)), said=round(g.p.mean(), 3), hap=round(g.y.mean(), 3)) for iv, g in x.groupby(b, observed=True) if len(g)]
    k = x[x.dk.notna()]
    probs.append(dict(mkt=mkt, n=len(x), said=round(x.p.mean(), 3), hap=round(x.y.mean(), 3), brier=round(((x.p - x.y) ** 2).mean(), 4),
                      brier_dk=round(((k.dk - k.y) ** 2).mean(), 4) if len(k) and mkt == "Anytime TD" else None,
                      brier_coin=round(((0.5 - x.y) ** 2).mean(), 4) if mkt.startswith(("Spread", "Game goes")) else None, buckets=bk))
out["probs"] = probs
# ---- Oct 22: does moving off the market help? Market, raw model and the site's blend, scored on the same games/players.
T3 = pd.DataFrame(T3); three = []
if len(T3):
    for kind, lab in (("margin", "Game margin"), ("total", "Game total"), ("td", "Anytime TD")):
        x = T3[T3.kind == kind]
        if not len(x): continue
        if kind == "td":
            sc = {k: round(float(((x[k] - x.act) ** 2).mean()), 4) for k in ("market", "model", "blend")}; unit = "Brier score, lower is better"
        else:
            sc = {k: round(float((x[k] - x.act).abs().mean()), 2) for k in ("market", "model", "blend")}; unit = "average miss in points"
        three.append(dict(name=lab, n=int(len(x)), unit=unit, **sc, best=min(sc, key=sc.get), weeks=sorted(set(map(int, x.wk)))))
out["three_way"] = three
if len(G):
    gl = G[G.spread.notna()]; gt = G[G.total.notna()]
    out["games"] = dict(n=len(G), margin_model=round((G.mg - G.m_margin).abs().mean(), 2), margin_line=round((gl.mg + gl.spread).abs().mean(), 2) if len(gl) else None,
                        total_model=round((G.tot - G.m_total).abs().mean(), 2), total_line=round((gt.tot - gt.total).abs().mean(), 2) if len(gt) else None)
wk_rows = []
for wk in out["weeks"]:
    a = A[(A.wk == wk) & (A.stat == "fp_ppr")] if len(A) else A; r = R[(R.wk == wk) & (R.stat == "fp_ppr")] if len(R) else R; p = PR[(PR.wk == wk) & (PR.mkt == "Anytime TD")]
    pk = p[p.dk.notna()]; g = G[G.wk == wk] if len(G) else G
    wk_rows.append(dict(week=int(wk), fp_mae=round((a.act - a.proj).abs().mean(), 2) if len(a) else None, fp_bias=round((a.act - a.proj).mean(), 2) if len(a) else None,
                        below=round(r.below.mean(), 3) if len(r) else None, above=round(r.above.mean(), 3) if len(r) else None,
                        td_said=round(p.p.sum(), 1), td_hap=int(p.y.sum()), td_n=len(p),
                        td_brier=round(((p.p - p.y) ** 2).mean(), 4) if len(p) else None, td_brier_dk=round(((pk.dk - pk.y) ** 2).mean(), 4) if len(pk) else None,
                        margin_model=round((g.mg - g.m_margin).abs().mean(), 1) if len(g) else None, margin_line=round((g.mg + g.spread).abs().mean(), 1) if len(g) else None, n_games=len(g),
                        snapshot=int(wk) in out["snap_weeks"]))
out["by_week"] = wk_rows
# ---- report card: fixed rules only, so the wording is generated from the numbers and never hand-written
card = []
def line(name, label, text): card.append(dict(name=name, label=label, text=text))
tdp = next((p for p in probs if p["mkt"] == "Anytime TD"), None)
if tdp and tdp["n"] >= 100:
    hi = [b for b in tdp["buckets"] if b["lo"] >= 0.4]; hn = sum(b["n"] for b in hi); hs = sum(b["said"] * b["n"] for b in hi) / max(hn, 1); hh = sum(b["hap"] * b["n"] for b in hi) / max(hn, 1)
    vs = "about as accurate as DraftKings" if tdp["brier_dk"] is None or abs(tdp["brier"] - tdp["brier_dk"]) < 0.003 else ("more accurate than DraftKings" if tdp["brier"] < tdp["brier_dk"] else "a little behind DraftKings")
    lean = "" if abs(tdp["hap"] - tdp["said"]) < 0.02 else (" Overall it runs low: players scored more often than it said." if tdp["hap"] > tdp["said"] else " Overall it runs high: players scored less often than it said.")
    tail = f" Too cautious on likely scorers: players it put at 40%+ scored {hh:.0%} of the time, not {hs:.0%}." if hn >= 20 and hh - hs > 0.08 else ""
    line("Touchdown chances", "On target" if not tail and not lean else "Lean found", f"{vs[0].upper() + vs[1:]} over {tdp['n']} priced players.{lean}{tail}")
elif tdp: line("Touchdown chances", "Too early", f"{tdp['n']} priced players so far; needs 100+.")
rf = next((r for r in rng if r["stat"] == "Fantasy points (PPR)"), None)
if rf and rf["n"] >= 150:
    tot = rf["below"] + rf["above"]
    lab, txt = ("On target", "about 1 in 5 results land outside the floor-to-ceiling range, as designed.") if 0.14 <= tot <= 0.26 else (("Too narrow", f"{tot:.0%} of results land outside the floor-to-ceiling range; 20% is the design.") if tot > 0.26 else ("Too wide", f"only {tot:.0%} of results land outside the floor-to-ceiling range; 20% is the design."))
    line("Floors and ceilings", lab, f"Over {rf['n']} player-games, {txt}")
elif rf: line("Floors and ceilings", "Too early", f"{rf['n']} player-games so far; needs 150+.")
if "games" in out and out["games"]["n"] >= 30:
    g = out["games"]; better = g["margin_line"] is not None and g["margin_model"] < g["margin_line"]
    line("Game lines", "Ahead of the market" if better else "Market is closer", f"Average miss on the margin {g['margin_model']:.1f} points vs the closing spread's {g['margin_line']:.1f}, over {g['n']} games." + ("" if better else " Read the model's spread as a second opinion, not a bet."))
elif "games" in out: line("Game lines", "Too early", f"{out['games']['n']} game{'s' if out['games']['n'] != 1 else ''} graded; needs 30+.")
if vl:
    parts = []
    for v in vl:
        d = v["model"] - v["dk"]
        parts.append(f"{v['stat'].lower()} " + ("tied" if abs(d) <= 0.03 * v["dk"] else ("ahead" if d < 0 else "behind")))
    behind = sum(1 for v in vl if v["model"] - v["dk"] > 0.03 * v["dk"])
    line("Player lines vs DraftKings", "Market is closer" if behind >= len(vl) / 2 else "Holding up", "Average miss on the same players: " + ", ".join(parts) + f" ({sum(v['n'] for v in vl)} lines). DraftKings' line already includes what the model knows, so a gap is information, not a bet.")
# line movement vs the model (line_moves.py): did the market move toward the model's number after it was published?
import os
if os.path.exists("line_moves.json"):
    LM = json.load(open("line_moves.json")); sp, tt = LM.get("spread", {}), LM.get("total", {}); n = sp.get("n", 0) + tt.get("n", 0)
    tw = sp.get("toward", 0) + tt.get("toward", 0); aw = sp.get("away", 0) + tt.get("away", 0)
    if n >= 20:
        share = tw / max(tw + aw, 1)
        line("Line movement", "Market followed" if share >= 0.6 and tw + aw >= 12 else ("Market went the other way" if share <= 0.4 and tw + aw >= 12 else "Even"),
             f"When the model's spread or total differed from the line by 1+ points at publish, the line later moved toward the model {tw} times, away {aw}, and stayed put {n - tw - aw} ({n} cases). Toward means the market later agreed with what the model saw first.")
    else: line("Line movement", "Too early", f"{n} cases of the model differing from the opening line by 1+ points; needs 20+. Toward-the-model moves would mean the market later agrees with what the model saw first.")
NEED3 = {"Game margin": 40, "Game total": 40, "Anytime TD": 300}
for t in three:
    need = NEED3[t["name"]]; nums = f"market {t['market']}, model {t['model']}, blend {t['blend']} ({t['unit']}, {t['n']} {'players' if t['name'] == 'Anytime TD' else 'games'})"
    if t["n"] < need: line(f"Off the market? {t['name']}", "Too early", f"{t['n']} of {need} needed. So far: {nums}.")
    else:
        lab = {"market": "Market is closer", "blend": "Blend adds a little", "model": "Ahead of the market"}[t["best"]]
        line(f"Off the market? {t['name']}", lab, f"Which number was closest to what happened: {nums}." + (" Moving off DraftKings' number hasn't helped here; the site's number should sit on the market's." if t["best"] == "market" else ""))
out["card"] = card
json.dump(out, open(os.environ.get("SC_TEST") or "scorecard.json", "w"), indent=1, allow_nan=False, default=lambda o: None)
print("weeks", out["weeks"], "snap", out["snap_weeks"]); print(pd.DataFrame(out["by_week"]).to_string()); print(pd.DataFrame(out["vs_lines"]).to_string())
for p in out["probs"]: print(p["mkt"], p["n"], p["said"], p["hap"], p["brier"], p["brier_dk"], p["brier_coin"])
