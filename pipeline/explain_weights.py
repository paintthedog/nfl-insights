"""
How much each input moves this week's numbers, in % terms.

Method (same for every model): for each group of inputs, replace it with the average
(league-wide for team inputs, position average for player inputs), re-run the model,
and measure how far each player's number moves. A group's weight is its average move
divided by the sum of all groups' average moves. Inputs that overlap (e.g. carries and
goal-line carries) each get credit for what they add on their own.
Writes weights.json.
"""
import contextlib, io, json, runpy
import numpy as np, pandas as pd

def quiet(path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return runpy.run_path(path)

def to_weights(base, variants, units):
    rows = []
    for label, val in variants.items():
        rows.append((label, float(np.mean(np.abs(val - base)))))
    tot = sum(v for _, v in rows) or 1
    out = [dict(label=l, pct=round(100 * v / tot, 1), swing=round(v * units, 2)) for l, v in rows]
    return sorted(out, key=lambda r: -r["pct"])

def neutral(df, cols, by=None, value=None):
    df = df.copy()
    for c in cols:
        if value is not None:
            df[c] = value
        elif by is None:
            df[c] = df[c].mean()
        else:
            df[c] = df.groupby(by)[c].transform("mean")
    return df

W = {}

# ------------------------------------------------------------------ interceptions
g = quiet("project_week.py")
M, fut, am, im = g["M"], g["fut"].copy(), g["am"], g["im"]
base = M.predict(fut, am, im)
p0 = base.p_int.to_numpy()
def p_from(df): return M.predict(df, am, im).p_int.to_numpy()
att_mean = base.proj_att.mean()
groups = {
    "Expected pass attempts": None,
    "Interception-worthy throws": ["log_qb_risk"],
    "Passes broken up": ["log_qb_pd"],
    "Throw depth": ["qb_adot"],
    "Game script (spread)": ["exp_margin"],
    "Opposing defense's INT rate": ["log_def_rate"],
    "Pass rush (both sides)": ["off_press_rate", "def_press_rate"],
    "QB inexperience": ["qb_exp_low"],
    "Wind": ["wind"],
}
var = {}
for label, cols in groups.items():
    if cols is None:
        lam = base.int_per_att * att_mean
        var[label] = 1 - np.exp(-lam.to_numpy())
    else:
        var[label] = p_from(neutral(fut, cols))
W["int"] = to_weights(p0, var, 100)          # swing in percentage points

# ------------------------------------------------------------------ anytime TD
g = quiet("td_project.py")
T, d, tmm, mr, mp, f = g["T"], g["d"], g["tmm"], g["mr"], g["mp"], g["f"]
x = d[d._pred & d.pid.isin(f.pid)].drop_duplicates("pid").copy()
b = T.predict(x, tmm, mr, mp)
keep = (b.p_td >= 0.10).to_numpy()
x, b = x[keep], b[keep]
p0 = b.p_td.to_numpy()
pt = lambda df: T.predict(df, tmm, mr, mp).p_td.to_numpy()
groups = {
    "Team's implied points (Vegas)": (["implied"], None),
    "Team's run/pass TD split": (["rush_share"], None),
    "Carry share": (["l_car_sh"], "pos"),
    "Goal-line carries (inside the 10)": (["l_i10_sh"], "pos"),
    "Target share": (["l_tgt_sh"], "pos"),
    "Red-zone targets": (["l_rz_sh"], "pos"),
    "Past touchdown share": (["l_rtd_sh", "l_ptd_sh"], "pos"),
    "Opponent's TDs allowed to position": (["l_dvp_td"], None),
}
W["td"] = to_weights(p0, {l: pt(neutral(x, c, by)) for l, (c, by) in groups.items()}, 100)

# ------------------------------------------------------------------ projections (Legends points)
g = quiet("proj_project.py")
P, d, va, vc, cal = g["P"], g["d"], g["va"], g["vc"], g["cal"]
td = pd.read_csv(f"td_props_2026_wk{__import__('os').environ.get('NFL_WEEK', 5)}.csv")[["pid", "lam_rush", "lam_rec", "exp_td", "rush_share"]]
ints = pd.read_csv(f"int_props_2026_wk{__import__('os').environ.get('NFL_WEEK', 5)}.csv")[["passer_player_id", "exp_ints"]].rename(columns={"passer_player_id": "pid"})
x = d[d._pred].drop_duplicates("pid").merge(td, on="pid", how="inner").merge(ints, on="pid", how="left")
x = x[x.pos.isin(["QB", "RB", "WR", "TE"])].copy()
x["t_rush"], x["t_rec"] = x.lam_rush, x.lam_rec
x["t_pass"] = np.where(x.pos == "QB", x.exp_td * (1 - x.rush_share) * 0.97, 0)

def fp(df):
    o = P.apply_calibration(P.raw_projection(df.copy(), va, vc), cal)
    o["p_rush_td"], o["p_rec_td"], o["p_pass_td"] = o.t_rush, o.t_rec, o.t_pass
    o.loc[o.pos == "QB", "p_ints"] = o.loc[o.pos == "QB", "exp_ints"].fillna(o.loc[o.pos == "QB", "p_ints"])
    return P.fantasy(o, "p_")["ppr"].to_numpy()

f0 = fp(x)
eff = ["e_catch", "e_ypt", "e_rec1d", "e_ypc", "e_rush1d", "e_cmp", "e_ypa", "e_sack", "e_pass1d", "e_int"]
groups = {
    "Vegas total & spread": (["total_line", "margin"], None),
    "Team's recent play volume": (["pg_tm_att", "pg_tm_car"], None),
    "Opponent defense": (["def_att_pg", "def_car_pg", "def_pass_ratio", "def_rush_ratio"], None),
    "Quarterback change": (["qd"], None),
    "Usage (targets, carries, QB attempts)": (["pr_tgt_sh", "pr_car_sh", "pr_start_att"], "pos"),
    "Snap share": (["l_snap", "snap_dn"], "pos"),
    "WOPR": (["l_wopr"], "pos"),
    "Efficiency (yards per play, catch rate)": (eff, "pos"),
    "Touchdown chances": (["t_rush", "t_rec", "t_pass"], "pos"),
    "Interceptions": (["exp_ints"], "pos"),
}
var = {l: fp(neutral(x, c, by)) for l, (c, by) in groups.items()}
W["proj"] = {}
for pos in ["QB", "RB", "WR", "TE"]:
    m = (x.pos == pos).to_numpy() & (f0 >= 6)
    v = {l: arr[m] for l, arr in var.items() if not (l == "Interceptions" and pos != "QB") and not (l == "WOPR" and pos == "QB")
         and not (l == "Snap share" and pos == "QB")}
    W["proj"][pos] = dict(n=int(m.sum()), rows=to_weights(f0[m], v, 1))

# ---- game model: typical swing of each input group in points, over recent games (coef x deviation from average)
import lines_model as LM
_pb, _sc = LM.load(); _g = LM.build(_pb, _sc)
_tr = _g[_g.result.notna() & _g.spread_line.notna() & (_g.min_games >= 3)]
_mm, _mt = LM.fit(_tr)
_rec = _tr[_tr.season >= _tr.season.max() - 2]
def _game_w(model, groups, X):
    out = []
    for lab, cols in groups:
        cols = [c for c in cols if c in model.params.index]
        dev = sum(model.params[c] * (_rec[c].fillna(_tr[c].median()) - _tr[c].mean()) for c in cols)
        out.append((lab, float(np.mean(np.abs(dev)))))
    tot = sum(v for _, v in out)
    return sorted([dict(label=l, pct=round(100 * v / tot, 1), swing=round(v, 2)) for l, v in out], key=lambda r: -r["pct"])
W["margin"] = _game_w(_mm, [("Offense ratings", ["off_gap"]), ("Defense ratings", ["def_gap"]), ("Quarterback quality & changes", ["qd_gap", "qb_gap", "qb_ret_gap", "qb_new_gap"]),
                            ("Short week & rest", ["rest_gap", "h_short", "a_short"]), ("Missing defensive regulars", ["dmiss_gap"]), ("Wind", ["wind_eff"])], LM.MARGIN_X)
W["total"] = _game_w(_mt, [("Offense ratings (both)", ["off_sum"]), ("Defense ratings (both)", ["def_sum"]), ("Quarterback quality & changes", ["qd_sum", "qb_any"]),
                           ("How often both teams pass", ["pass_sum"]), ("Pace", ["pace_sum"]), ("Weather & roof", ["wind_eff", "cold", "dome"]),
                           ("Missing defensive regulars", ["dmiss_sum"])], LM.TOTAL_X)
W["home_field"] = round(float(_mm.params["const"]), 1)
json.dump(W, open("weights.json", "w"), indent=1)
for k in ["margin", "total"]:
    print(k); [print(f"   {r['pct']:5.1f}%  ±{r['swing']:.2f}  {r['label']}") for r in W[k]]
for k in ["int", "td"]:
    print(k); [print(f"   {r['pct']:5.1f}%  ±{r['swing']:.1f}  {r['label']}") for r in W[k]]
for pos, v in W["proj"].items():
    print(pos, f"(n={v['n']})"); [print(f"   {r['pct']:5.1f}%  ±{r['swing']:.2f}  {r['label']}") for r in v["rows"]]
