"""
Game simulator: joint outcomes for every player in a game.

For each simulated game:
  1. Team pass attempts and carries drawn jointly (correlated, sd from history).
  2. Each team's targets split across its players by a Dirichlet draw around their projected shares
     (overdispersed vs pure chance, as measured). Carries likewise, with more dispersion.
  3. Catches: binomial on each player's catch rate. Yards: sum of per-play draws (gamma-ish, heavy right tail).
  4. Touchdowns: team TD count drawn (Poisson around the TD model's team mean), then assigned to players by
     their TD-share weights (from the TD model's lam_rush / lam_rec), so scorers are mutually exclusive-ish.
  5. QB stats are the sums over his receivers, so QB and WR outcomes are consistent by construction.

Inputs are the existing projections (means) per player; the simulator only adds the joint randomness.
Returns per-player samples (N_SIM x players) for: tgt, rec, rec_yds, car, rush_yds, rec_td, rush_td, pass_yds, pass_td, fp_ppr, fp_std.
"""
import numpy as np, pandas as pd

ATT_SD, CAR_SD, ATT_CAR_CORR = 7.8, 6.9, -0.36
TGT_OVERDISP, CAR_OVERDISP = 1.4, 3.2     # variance multipliers vs binomial
YPT_SHAPE = 1.3                           # gamma shape for per-target yards (long right tail)
YPC_SHAPE = 1.6
GAME_EFF_SD = 0.30                        # per-player per-game multiplier on yards/touch (lognormal sd): hot and cold days
SHARE_SD = {"tgt": 0.15, "car": 0.5}        # lognormal sd on each player's expected share: uncertainty in the role itself
EXIT_P = 0.03                             # chance a player leaves early (injury/benching): stats cut at a uniform point
TD_OVERDISP = 1.35                        # team TD count: negative-binomial-ish


def _dirichlet_conc(shares, overdisp, n_trials):
    """Dirichlet concentration that reproduces var = overdisp * binomial var for a share drawn then sampled."""
    # var(share draw) = p(1-p)/(c+1); total var ≈ p(1-p)/(c+1) + p(1-p)/n  -> set (c+1) = n/(overdisp-1)
    return max(n_trials / max(overdisp - 1, 0.05), 2.0)


def simulate_game(team_rows: dict, n_sim=4000, rng=None, pool_frac=1.0):
    """team_rows: {team: DataFrame of that team's players with columns
         pid, name, pos, p_tgt, p_car, p_rec, p_rec_yds, p_rush_yds, e_catch, e_ypt, e_ypc, lam_rec, lam_rush, v_att, v_car, exp_td, rush_share,
         optional: p_ints, qb_mult (QB rows), p_play (chance a Questionable player suits up; his p_* are "if he plays")}
    pool_frac: share of the team's passing that goes to the listed players (the QB's line is their sum / pool_frac).
    Returns {team: {pid: {stat: np.array(n_sim)}}} plus team-level arrays."""
    rng = rng or np.random.default_rng()
    out = {}
    for team, df in team_rows.items():
        df = df.reset_index(drop=True)
        n = len(df)
        v_att, v_car = float(df.v_att.iloc[0]), float(df.v_car.iloc[0])
        # 1. team volume, correlated
        z = rng.multivariate_normal([0, 0], [[1, ATT_CAR_CORR], [ATT_CAR_CORR, 1]], size=n_sim)
        att = np.clip(np.round(v_att + ATT_SD * z[:, 0]), 12, 65).astype(int)
        car = np.clip(np.round(v_car + CAR_SD * z[:, 1]), 8, 55).astype(int)
        # 2. shares (targets normalized to the team's projected attempts; carries to projected carries)
        tsh = df.p_tgt.to_numpy() / max(df.p_tgt.sum(), 1e-9)
        csh = df.p_car.to_numpy() / max(df.p_car.sum(), 1e-9)
        qb_yfac = None
        pp = df.p_play.fillna(1.0).to_numpy().astype(float) if "p_play" in df else np.ones(n)
        active = rng.random((n_sim, n)) < pp[None, :] if (pp < 1).any() else None
        _qi = np.where((df.pos == "QB").to_numpy())[0]
        if active is not None and len(_qi):           # quarterbacks: exactly one plays each run, chosen by their chances
            _w = pp[_qi] / pp[_qi].sum(); _pick = np.minimum(np.searchsorted(np.cumsum(_w), rng.random(n_sim)), len(_qi) - 1)
            active[:, _qi] = False; active[np.arange(n_sim), _qi[_pick]] = True
            if "p_pass_yds" in df and len(_qi) > 1:   # the passing game is better or worse depending on which of them plays
                _py = df.p_pass_yds.fillna(0).to_numpy()[_qi]; _mix = float((_w * _py).sum())
                qb_yfac = (_py[_pick] / _mix) if _mix > 0 else None
        # the sim draws attempts around v_att and gives 98.5% of them a target; align the mean of targets to projected targets
        att_scale = (df.p_tgt.to_numpy() * pp).sum() / max(v_att * 0.985, 1e-9)
        car_scale = (df.p_car.to_numpy() * pp).sum() / max(v_car, 1e-9)
        ct = _dirichlet_conc(tsh, TGT_OVERDISP, v_att); cc = _dirichlet_conc(csh, CAR_OVERDISP, v_car)
        tshare = rng.dirichlet(np.maximum(tsh * ct, 1e-3), size=n_sim)
        cshare = rng.dirichlet(np.maximum(csh * cc, 1e-3), size=n_sim)
        for sh, key in ((tshare, "tgt"), (cshare, "car")):
            sd = SHARE_SD.get(key, 0.0)
            if sd > 0:
                sh *= np.exp(rng.normal(-sd * sd / 2, sd, size=sh.shape)); sh /= sh.sum(1, keepdims=True)
            if active is not None:      # a player who sits gets nothing; his share goes to the teammates who dressed
                sh *= active; sh /= np.maximum(sh.sum(1, keepdims=True), 1e-12)
        att_t = np.clip(np.round(att * 0.985 * att_scale), 5, 70).astype(int)     # targets handed out
        car_t = np.clip(np.round(car * car_scale), 3, 60).astype(int)
        tgt = np.array([rng.multinomial(a, s) for a, s in zip(att_t, tshare)])        # n_sim x n
        carr = np.array([rng.multinomial(c, s) for c, s in zip(car_t, cshare)])
        # 3. catches and yards
        has_t = df.p_tgt > 0.2; has_c = df.p_car > 0.2
        catch = np.clip(np.where(has_t, df.p_rec / df.p_tgt.where(has_t, 1), df.e_catch).astype(float), 0.2, 0.95)
        ypt = np.clip(np.where(has_t, df.p_rec_yds / df.p_tgt.where(has_t, 1), df.e_ypt).astype(float), 0.5, 25)
        ypc = np.clip(np.where(has_c, df.p_rush_yds / df.p_car.where(has_c, 1), df.e_ypc).astype(float), 0.5, 12)
        rec = rng.binomial(tgt, catch)
        # yards: gamma with mean = tgt * ypt, shape scales with tgt (sum of per-target gammas)
        eff = np.exp(rng.normal(-GAME_EFF_SD ** 2 / 2, GAME_EFF_SD, size=(n_sim, n)))   # mean 1
        rec_yds = np.where(tgt > 0, rng.gamma(np.maximum(tgt * YPT_SHAPE, 1e-6), (ypt / YPT_SHAPE)[None, :]) * eff, 0.0)
        rec_yds = np.where(rec > 0, rec_yds, 0.0)
        if qb_yfac is not None: rec_yds = rec_yds * qb_yfac[:, None]
        rush_yds = np.where(carr > 0, rng.gamma(np.maximum(carr * YPC_SHAPE, 1e-6), (ypc / YPC_SHAPE)[None, :]) * eff, 0.0)
        # 4. touchdowns: team count then assignment
        exp_td = float(df.exp_td.iloc[0]); rs = float(df.rush_share.iloc[0])
        team_td = rng.negative_binomial(n=exp_td / (TD_OVERDISP - 1), p=1 / TD_OVERDISP, size=n_sim) if TD_OVERDISP > 1 else rng.poisson(exp_td, n_sim)
        n_rush = rng.binomial(team_td, rs); n_rec = team_td - n_rush
        w_rush = df.lam_rush.to_numpy().clip(min=1e-6); w_rec = df.lam_rec.to_numpy().clip(min=1e-6)
        if "qb_mult" in df: w_rush = w_rush * np.where(is_qb_arr := (df.pos == "QB").to_numpy(), df.qb_mult.fillna(1).to_numpy(), 1.0)
        # weight scorers by projected rate AND by this sim's usage (a back with 25 carries is likelier to score)
        wr_ = w_rush[None, :] * (0.5 + 0.5 * carr / np.maximum(carr.sum(1, keepdims=True), 1) * n)
        wc_ = w_rec[None, :] * (0.5 + 0.5 * tgt / np.maximum(tgt.sum(1, keepdims=True), 1) * n)
        if active is not None: wr_ = wr_ * active + 1e-12; wc_ = wc_ * active + 1e-12
        rush_td = np.array([rng.multinomial(k, w / w.sum()) if k > 0 else np.zeros(n, int) for k, w in zip(n_rush, wr_)])
        rec_td = np.array([rng.multinomial(k, w / w.sum()) if k > 0 else np.zeros(n, int) for k, w in zip(n_rec, wc_)])
        # 5. QB passing = sum over receivers
        is_qb = (df.pos == "QB").to_numpy()
        pass_yds = np.where(is_qb[None, :], rec_yds.sum(1, keepdims=True) / pool_frac, 0.0)
        pass_td = np.where(is_qb[None, :], rec_td.sum(1, keepdims=True), 0)
        cmp_ = np.where(is_qb[None, :], np.round(rec.sum(1, keepdims=True) / pool_frac), 0).astype(int)
        # interceptions for QBs (Poisson around the INT model's expectation)
        p_ints = df.p_ints.fillna(0).to_numpy() if "p_ints" in df else np.zeros(n)
        ints = np.where(is_qb[None, :], rng.poisson(np.maximum(p_ints, 0)[None, :].repeat(n_sim, 0)), 0)
        # early exits: scale every counting stat by a uniform fraction for a few sims per player
        exit_f = np.where(rng.random((n_sim, n)) < EXIT_P, rng.random((n_sim, n)), 1.0)
        for arr in (tgt, rec, carr, rec_td, rush_td, pass_td, cmp_, ints):
            arr[:] = rng.binomial(arr, exit_f)
        rec_yds *= exit_f; rush_yds *= exit_f; pass_yds *= exit_f
        if active is not None:                         # only the quarterback who played gets the passing line
            pass_yds = pass_yds * active; pass_td = pass_td * active; cmp_ = cmp_ * active; ints = ints * active
        fp_std = pass_yds / 25 + 4 * pass_td - 2 * ints + rush_yds / 10 + 6 * rush_td + rec_yds / 10 + 6 * rec_td
        fp_ppr = fp_std + rec
        # DraftKings classic: full PPR, -1 per INT, +3 for 300 passing / 100 rushing / 100 receiving yards (fumbles not modelled)
        fp_dk = fp_ppr + ints + 3 * ((pass_yds >= 300).astype(int) + (rush_yds >= 100) + (rec_yds >= 100))
        out[team] = dict(pids=df.pid.tolist(), names=df.name.tolist(), pos=df.pos.tolist(),
                        tgt=tgt, rec=rec, rec_yds=rec_yds, car=carr, rush_yds=rush_yds, rec_td=rec_td, rush_td=rush_td,
                        pass_yds=pass_yds, pass_td=pass_td, cmp=cmp_, ints=ints, fp_std=fp_std, fp_ppr=fp_ppr, fp_dk=fp_dk,
                        team_att=att, team_car=car, team_td=team_td, active=active)
    return out


def summarize(sim, quants=(0.1, 0.25, 0.5, 0.75, 0.9)):
    rows = []
    for team, s in sim.items():
        for i, pid in enumerate(s["pids"]):
            r = dict(team=team, pid=pid, name=s["names"][i], pos=s["pos"][i])
            for st in ["tgt", "rec", "rec_yds", "car", "rush_yds", "rec_td", "rush_td", "pass_yds", "pass_td", "fp_ppr", "fp_std"]:
                x = s[st][:, i]; r[f"{st}_mean"] = x.mean()
                for q in quants: r[f"{st}_q{int(q*100)}"] = np.quantile(x, q)
            r["p_td"] = ((s["rec_td"][:, i] + s["rush_td"][:, i]) >= 1).mean()
            rows.append(r)
    return pd.DataFrame(rows)
