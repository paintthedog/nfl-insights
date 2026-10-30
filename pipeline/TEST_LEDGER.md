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
| Oct 30 | QB change split re-run (same test, data through Week 7) | (scratch, same method as t_qb_split.py) | Margin MAE 10.137 → 10.147 overall; games with a returning starter 9.893 → 9.832 (n=186); new/backup QB games 10.890 → 10.978 (n=324). 2026 fit: returning −1.41 pts, new/backup −2.80 pts (was one −2.24 for any change). Closing line MAE 9.81 | ADOPTED — Clinton's decision (Oct 30): a returning starter must be treated differently from a backup. Accuracy cost is within noise; watch returning-starter games on the scorecard |
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

## Oct 22: steadiness rating (Clinton asked for a role-consistency rating across all players, plus performance variability)
| Test | Script | Result | Decision |
|---|---|---|---|
| Does game-to-game spread in role (opportunity share), snap share, or PPR tell us whether a player clears a prop rung more often than the model's chance? | tests/t_steady.py | Rungs the model put at ~70–85% (rec for RB/WR/TE, rush yds for RB, pass yds for QB), 23,904 legs. Steadiest third vs jumpiest third, hit rate minus model chance: role spread +2.1 pts (t=3.0) in 2019–23, +1.3 (t=1.3) in 2024–26; PPR spread +1.4 (t=2.0) then +3.2 (t=3.1); snap-share spread +0.6 then +0.1 (nothing); equal-weight blend of role + PPR +2.0 (t=2.8) then +2.0 (t=2.0). | (see next row) |
| Same test, players with a real role only | tests/t_steady.py | Restricted to players with a real role (6+ PPR, 10%+ of targets, RB 15%+ of carries + targets: the players who get rated), 18,858 legs: blend +0.7 (t=0.9) fit, +1.9 (t=1.7) check; role +1.0 / +1.1; PPR spread +0.4 / +2.6 (t=2.2). Most of the all-player signal was fringe players missing rungs. | KEPT as a DESCRIPTION and filter (steady.py, Steady column on Projections, ◆ role-change flags), at Clinton's request. Not shown to add an edge among bettable players. Equal weights because the parts traded places; snap spread dropped. Never feeds a projection. Tested against the model's chance, NOT DraftKings' price. Revisit at season end with 2026 Weeks 7–18. |
| Paper test: "Parlays: steady legs, rated 60+" (same rule as steady legs 1.25–1.50, legs limited to Steady 60+ with no role flag) | paper_slate.py | Starts Week 8 game morning. Every steady-leg now carries its score (`stdy`, `stdy_flag`) in the legs JSON. | Exploratory (150 bets, 3 SE). Better read sooner: leg-level hit rate by score band vs the leg's DK no-vig chance once 200+ tagged legs exist. If rated legs don't beat DK's price, the rating is a description, not an edge. |

## Oct 25: more volume for the tests (Clinton: "not testing nearly enough bets")
Week 7 logged 472 paper bets, but the pre-registered main tests fill slowly: QB yards + WR1 catches 0 (the hand-written URL block dropped player_pass_yds_alternate and team_totals), small-underdog teasers 1 a week, matchup receptions ~5 a week. The main tests are NOT changed. Fixes and exploratory side rules (150 bets, 3 SE, like every other exploratory rule):
| Change | Where | Note |
|---|---|---|
| URL block is generated, never hand-written | make_urls.py | ~101 credits for a 14-game slate (7/game + 3). Restores QB yards + WR1 catches. |
| Teaser legs: small underdogs +6 | paper_extra.py | Each leg of the main rule graded alone (ref price 1.80**0.5 = 1.342, break-even 74.5%). Doubles the sample, same rule. |
| Teasers (wide): underdogs +1 to +3, all pairs | paper_extra.py | Comparison rule. Backtest 2016–2025 (below) says the widening DILUTES: expect it to trail the main rule. |
| Same-game stacks (underdog): the three sgp_slate rules on the DK underdog | sgp_slate.build(side="dog") | Oct 16 correlation test: favorite win + dog offense is negatively correlated; dog win + own RB is positively. Graded on hit rate vs joint chance. |
| Matchup props: receptions over, opp top-8 (no TD filter) | paper_extra.py | Clinton's matchup rule without the TD-rate half. Real-season-record rule applied (8+ catches / 25+ carries) to all extras. |
Backtest, 6-point teaser legs, nflverse closing spreads 2016–2025 regular season (pushes dropped), break-even 74.5% at a 1.80 two-teamer:
| Band | Legs | Cover | z vs break-even | 2-leg ROI |
|---|---|---|---|---|
| Underdogs +1.5 to +2.5 (main rule) | 421 | 77.4% ±2.0 | +1.4 | +7.9% |
| Underdogs +1 to +3 (wide) | 967 | 75.2% ±1.4 | +0.5 | +1.7% |
| Underdogs +1 only | 171 | 73.1% | −0.4 | −3.8% |
| Underdogs +3 only | 375 | 73.6% | −0.4 | −2.5% |
| Favorites −1.5 to −2.5 | 423 | 67.6% | −3.0 | −17.7% |
| Favorites −7.5 to −8.5 | 220 | 71.8% | −0.9 | −7.2% |
Read: the main teaser band is the only one above break-even, and even ten seasons only get it to 1.4 SE. Sixty forward bets cannot settle it alone; the backtest plus the forward record together are the evidence. Never tease favorites of −1.5 to −2.5.

## Oct 25 (later): prop pulls ended — the site predicts outcomes, the market is a free benchmark
Clinton stopped the manual Odds API pulls: when he pulled decided what got logged. From Week 8 every paper bet comes from `paper_lines.py`, priced from nflverse's schedule feed at the last daily run before kickoff (fixed timing). Continuing: Teasers: small underdogs +6 (main test, 60), Teaser legs, Teasers (wide), Moneyline model differs 4+, Game totals model differs 3+, Trend unders (wind, prime time), Control: random. Stopped (record.py RETIRED, graded rows kept): ML + top TD scorer, matchup props, all same-game stacks, receptions/INT/team-total gaps, alt-line and steady-leg parlays, TD singles/parlays/bands, board consistency. Note for reading the record: nflverse prices are a market consensus, not DraftKings, so these rules are not comparable bet-for-bet with the Week 6–7 DK rows.
Accuracy questions that still get answered without prices: fantasy-point MAE and bias, TD chance calibration (does 30% score 30%?), floor/ceiling coverage, and game margin/total vs the closing line (scorecard.py).

## Oct 26: accuracy review (Clinton: hone accuracy and confidence; beating the market is not the goal)
| Test | Script | Result | Decision |
|---|---|---|---|
| TD chances under-confident at 40%+? (scorecard: 59% scored vs 50% said, 41 players, Weeks 4–7) | tests/t_td_calib.py (walk-forward 2021–2026, ~25,800 player-games) | 40–45% band said .422 / scored .411; 45–55% .490 / .525; 55%+ .598 / .619. Long-run gap 1–3 pts, not 9. The 2.5–3.9 band (26–40%) runs ~1 pt low every season 2022–2025 | NO CHANGE — this season's gap is mostly a small sample. Recheck the 40%+ bucket at 150+ players |
| Floors and ceilings by stat (scorecard, Week 7, ties count half) | scorecard.py | Receptions 20% outside (on target), targets 17%, PPR 23%; receiving yards 29% (too narrow); rushing yards 39% and carries 37% (n=122, below the 150 floor) | WATCH — if rushing/receiving ranges are still 26%+ outside at 150+ results, widen the yards-per-touch spread (GAME_EFF_SD / YPC_SHAPE) and rerun sim_backtest.py before shipping |
| No-shows: projected to play (p_play 0.9+) for 5+ PPR, never touched the ball | scorecard.py | 2 of 158 (1.3%), Week 7 | WATCH — not a real error source yet |
| Game totals lean | scorecard.py | Model +3.1 pts high vs closing total +1.0 (13 games) | WATCH — needs 30+ games before any shift |
