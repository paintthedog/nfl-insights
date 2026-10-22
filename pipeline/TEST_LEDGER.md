# Test ledger — every model test, kept or not

Why this exists: by mid-October we had run ~45 tests. At a 5% false-positive rate, two or three "wins" are flukes. Rules:
1. Every test gets a row here, including the ones that failed.
2. **Holdout:** 2026 Week 7 onward is untouched evaluation data for everything added in October. Do not tune anything on it.
3. **Season-end re-check (after Week 18):** re-score each KEPT change on Weeks 7–18 only (walk-forward, same metric as when it was adopted). Remove any change that is worse than the version without it. Re-check REJECTED changes that were close (marked "revisit").
4. A change ships only if it beats the current model on seasons it was not fit on, or is a documented judgment call (marked JUDGMENT, with the trade-off).

| Date | Test | Script | Result (walk-forward unless noted) | Decision |
|---|---|---|---|---|
| Sep–Oct | Team ratings: memory (decay), shrinkage, matchup terms, market line as input | lines_variants.py | Current settings best | Kept as is |
| Oct | Strength of schedule (simple + DVOA-style) | tests/t_sos.py, t_sos_retro.py | No gain | REJECTED (switch OPP_ADJ=0) |
| Oct | Hidden player stats (YACOE, RYOE, separation, cushion, box8, TTT, CPOE, aggressiveness), pace, no-huddle, PA, screen, motion | adv_test2.py | All < 0.5% | REJECTED |
| Oct | Snap share, air-yards share, WOPR, PROE | feature_test.py | Rounding errors | REJECTED |
| Oct | Award fit shares / DROY LB tackles | tests/t_award_calib.py, t_droy_lb.py | Tackles feature no help | REJECTED |
| Oct | TD parlay rules (price floor 1.15, cap 3.0, top-5 totals, favored RB1) | tests/t_td_*.py, t_wk5_top5.py | In-sample on Week 5 only | Paper-tracked (not a model change) |
| Oct 16 | Success rate in team ratings — spreads | tests/t_success.py | Margin MAE −0.032 ±0.031, 7/8 seasons better; totals worse | KEPT (spreads only) — revisit at season end |
| Oct 16 | Success rate in player projections (individual) | tests/t_success_player.py | Best −0.6% (WR tgt) | Folded into bundle below |
| Oct 16 | Bundle: QB dropback SR + receiver target SR + pace (+ aggressiveness) | tests/t_bundle.py | WR pts −0.34% (5/5 seasons, ~3 SE); QB pass yds −0.42% | KEPT without aggressiveness (sr_adjust.py) — JUDGMENT: below the 1% bar, kept as small-but-real |
| Oct 16 | Garbage-time filter, fumble luck, explosive plays (team ratings) | tests/t_team_bundle.py | Spreads ±0.03 noise; garbage time hurts totals | REJECTED |
| Oct 16 | Explosive plays & garbage-time usage (player projections + ceilings) | tests/t_proj_expl.py | Means ≤0.2%; ceilings no Brier gain | REJECTED |
| Oct 16 | Game-line probability calibration (logistic on model−line gap) | calib_lines.py | Old probs worse than coin flip (Brier .256); calibrated .2504 | KEPT |
| Oct 16 | Prop probabilities: model vs DK blend | tests/t_prop_calib.py | DK better; best weight 0–20% model (weeks 4–6, n≈730) | KEPT 20% model blend on display — refit as weeks accumulate |
| Oct 16 | Same-game leg correlation (likely legs) | tests/t_combo_lift.py | QB+WR1 stack +4–5 pts (8/8 seasons); fav win + dog offense negative; cross-game independent | Informs parlay rules |
| Oct 16 | Fav RB1 TD + dog WR1 catches | (inline) | Lift ~1.01, unstable by season | No effect |
| Oct 17 | QB change split: new/backup vs starter returning | tests/t_qb_split.py | Returning-starter bias +1.27 → +0.59, but margin MAE +0.010 (2/8 seasons) | REJECTED for now (flags computed, off) — revisit at season end |
| Oct 17 | Receiver team-scaling bias + final recalibration (WR/TE rec, tgt, rec yds) | tests/t_team_scale.py | Top-20% receivers bias +0.23 rec / +3.7 yds → ~0; MAE +0.7% | KEPT — JUDGMENT: bias matters for over/under probabilities; check Week 6+ prop scorecard |
| Oct 17 | Market role check (audit flag, not a model change) | audit_stale.py | Flags White, Allgeier, McLaughlin, Lemon, London on Week 6 | KEPT (process) |

## Oct 18 additions
| Test | Status | Decision rule |
|---|---|---|
| Same-game stacks (favorite + lead RB; top two pass catchers), joint chance from parlay_joint | Started Week 6 (22 stacks) | After Week 18: does the hit rate match model_p better than indep_p? If not, the copula isn't adding anything live. No P&L verdict possible without DK same-game prices. |
| Scorecard: every published number vs results | Started Week 6 (full); Weeks 4–5 from odds logs only | Watch: TD chance buckets (Weeks 4–6 hint the model is underconfident at 40%+), floor/ceiling coverage (target 10/10/50), average lean by stat. Pool weeks; no single-week changes. |
| Board consistency (DK vs DK): bet the side that brings an outlier back toward the rest of DK's board | Started Week 6 (5 bets) | After Week 18: hit rate vs the board's implied chance and vs break-even. If outliers are usually news DK priced in (ROLE), this loses; that is also an answer. |
| Line movement vs model | Started Week 6 | 20+ cases: share of toward-the-model moves. ≥60% means the market later agrees with the model; ≤40% means the market knows things the model doesn't. |
| lead_back override (Oct 18) | Live | Moves carries/yards to the named back (Allgeier's projection landed on DK's 47-yd line). Known gap: the TD chance barely moves (18%->20% vs DK 42%); the TD model's goal-line features don't follow the share override. Look at this in the offseason, or sooner if lead-back TD flags keep appearing. |
