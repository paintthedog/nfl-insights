# QB INT prop model

Predicts each starting QB's chance of throwing 1+ interception (the over/under 0.5 INT prop) and the fair odds on both sides.

## Weekly use
```
pip install nflreadpy polars pyarrow pandas statsmodels
python get_data.py                  # refresh nflverse play-by-play + schedule
# edit SEASON, WEEK and OVERRIDES in project_week.py
python project_week.py              # writes int_props_<season>_wk<week>.csv
python backtest.py                  # optional: re-check skill vs baselines
```
**Always check starters.** The script assumes each team's most recent starter plays. Injuries and benchings go in `OVERRIDES` (team -> nflverse passer_player_id). Get the ID from `qb` in `int_model.py`, or from `nflreadpy.load_players()`.

## How it works
1. **Attempts model (OLS):** projects pass attempts from the Vegas total, the spread, the QB's recent attempts per start, and the team's recent pass volume. MAE is about 6.4 attempts.
2. **Per-attempt INT rate (Poisson GLM):** trained with actual attempts as exposure. Inputs are the QB's **interception-worthy throw rate** (FTN charting, 2022+, shrunk toward the level implied by his actual INT history, so uncharted QBs fall back to real INTs), the QB's **passes-defensed rate** (breakups on his throws), aDOT, a low-experience flag, sack+hit rates for both the offense and the defense, the defense's INT rate (shrunk hard, 1500 pseudo-attempts), wind, and expected margin.
3. **Output:** expected INTs = rate × projected attempts; P(1+ INT) = 1 − e^(−expected INTs).

Every feature uses only games played *before* the one being predicted.

## Backtest (train 2016–23, test 2024 – 2026 Wk3, 1,226 QB starts)
| model | Brier | log loss |
|---|---|---|
| always guess the hit rate | 0.2498 | 0.6928 |
| projected attempts × league rate | 0.2502 | 0.6935 |
| projected attempts × shrunk QB rate | 0.2485 | 0.6902 |
| **full model** | **0.2446** | **0.6823** |

About 2.1% Brier skill over a constant guess. That's real, and it's small.

## Factors tested (walk-forward, `factor_tests.py` / `factor_tests2.py`)
| factor | verdict |
|---|---|
| QB interception-worthy throws (FTN) | **Kept.** Best single upgrade: 2024–26 skill 1.33% → 1.80% |
| QB passes defensed (breakups on his throws) | **Kept.** Helps in every test |
| Defense INT rate | Kept, but small. Removing it costs a little; how hard it's shrunk doesn't matter |
| Defense INTs + breakups ("ballhawk" rate) | Dropped. Worse than INTs alone |
| Spread buckets / curves | Dropped. Linear spread term is as good |

## INT rate by point spread (2016 – 2026 Wk3, starters with 10+ attempts)
| spot | starts | P(1+ INT) | INT % of attempts |
|---|---|---|---|
| Dog 11+ | 268 | 63.1% | 3.14 |
| Dog 7–10.5 | 655 | 52.1% | 2.35 |
| Dog 4–6.5 | 805 | 55.2% | 2.37 |
| Dog 1–3.5 | 1,021 | 54.7% | 2.47 |
| Fav 1–3.5 | 1,311 | 49.5% | 2.13 |
| Fav 4–6.5 | 663 | 47.4% | 1.99 |
| Fav 7–10.5 | 587 | 42.6% | 1.80 |
| Fav 11+ | 213 | 45.1% | 1.63 |

Attempts barely change with the spread (about 31–34 per start in every bucket). The effect is almost entirely **per-throw risk**: trailing QBs force throws.

`spread_variants.py` tests whether spread buckets, a squared term, or extra slope past ±10 beat the straight-line spread term in walk-forward testing (2019–2026, 3,952 starts). None did; all scored within 0.0001 Brier. The model keeps the linear term. The projections file includes `hist_p_at_spread` (a generic QB at this spread) and `qb_vs_spread_hist` (how far this QB's profile moves him off it).

**Caution:** in walk-forward testing, the model ran about 4 points light on 11+ point underdogs (60% predicted vs 64% actual, n=201). That could be noise, but lean skeptical on unders for huge dogs.

## Betting notes
- Compare `p_int` to the book's **no-vig** probability, not the raw price. At −130/+100 juice you need roughly a 3–4 point edge before anything is worth a bet.
- In the backtest, the lowest-risk quintile hit 39% vs 43% predicted, and the highest hit 60% vs 58%. The model is slightly too timid at the extremes, so its strongest leans are, if anything, understated.
- Known gaps: no true pressure data (sack+hit is a crude stand-in), no weather forecast before game day, no in-game injury risk, and no receiver/OL injuries.

---

# Anytime TD scorer model (`td_model.py`)

## Weekly use
```
python get_data.py
python -c "import nflreadpy as n; n.load_players().write_parquet('data/players.parquet'); n.load_rosters_weekly([2026]).write_parquet('data/rosters_2026.parquet')"
# edit SEASON, WEEK, QB_OVERRIDES, QUESTIONABLE in td_project.py
python td_project.py                   # writes td_props_<season>_wk<week>.csv
# paste prices into td_odds_wk<week>.csv (columns: name,yes,no,book; 'no' optional), then:
python td_edges.py 4
```

## How it works
1. **Team:** expected offensive TDs from the Vegas implied team total (Poisson GLM), split rush/pass by the team's recent rush-TD share.
2. **Player:** share of those TDs from decayed opportunity shares: carries and carries inside the 10 for rushing TDs; targets and red-zone targets for receiving TDs. Position terms cover RBs and TEs. QBs are included for rushing TDs (starters only).
3. **Team rescale:** each team's listed players are scaled so their expected TDs add up to the team total. That redistributes an injured player's share, but only proportionally across everyone.
4. P(anytime TD) = 1 − e^(−λ). Offensive TDs only.

## Backtest (walk-forward 2021 – 2026 Wk3, 27,593 player-games)
| model | Brier | skill vs constant |
|---|---|---|
| constant (21.2%) | 0.1670 | 0% |
| player's own recent TD rate | 0.1582 | 5.3% |
| **model** | **0.1535** | **8.1%** |

Calibrated within about 1 point from 10% to 70%. Known biases: **QBs run about 2 points high** (rushing TDs cluster, and may be trending down), and fullbacks run low (tiny sample).

## Caveats
- **Injuries are the whole game here.** The pool drops IR and already-ruled-out players, but Friday designations and game-day inactives aren't in the data yet. Rerun Saturday.
- **The rescale is crude for big absences.** When a lead back is out, his work goes mostly to the backup RB, not proportionally to the whole team (Miami this week, with Achane on IR).
- **Anytime TD books carry heavy margin**, often 6%+ on the Yes side alone, so edges need to be large. `td_edges.py` strips an assumed 6% when only the Yes price is given.

---

# Player stat projections (`proj_model.py`)

Weekly: run the INT and TD projections first, then
```
python proj_backtest.py        # only when you want to refresh ranges/recalibration (about 2 min)
python proj_project.py         # writes proj_<season>_wk<week>.json
python build_site.py 4 "<odds snapshot time>"
```
- **Volume:** team pass attempts and carries from the Vegas total, spread, team pace, and opponent volume allowed.
- **Share:** each player's decayed target and carry share. Receivers are scaled so they add up to their QB's projection; non-QB carries are scaled to the team's projected carries (this is how an injured back's work gets redistributed).
- **Efficiency:** catch rate, yards per target, yards per carry, first downs per touch, QB completion %, YPA, and sack rate, all shrunk hard to position averages, with a small opponent adjustment. Calibrated per stat and position on past seasons only.
- **Ranges:** empirical actual/projected ratios from out-of-sample games; P(over) uses Platt recalibration fit on held-out games.
- **Backtest (2021 – 2026 Wk3):** fantasy-point error is 11% lower than a season-average baseline for QBs and 3–4% lower for RB/WR/TE (6–9% lower than last-3-games averages).
