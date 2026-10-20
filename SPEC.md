# NFL Insights: rebuild spec

This spec describes what the original pipeline does as of Oct 19, 2026. It was reconstructed from three sources: the live site's data and its own "Model" tab text, the original chat's summaries, and spot checks of that chat. It is the target for the rebuild. Numbers marked *(orig)* are what the original reported. Re-measure them; don't trust them.

> A full code recovery from the old chat's transcript is technically possible, because every Write, Edit and patch is in it. But it would cost roughly 8–10M tokens, so it was declined (Oct 20). A pilot recovered turns 3–20, which hold the first INT and TD models and the first site build. Those ops are kept in `recover/ops/` (raw transcript text with HTML entities; decode with `html.unescape`). A targeted recovery of single files remains an option later.

## 0. Principles (kept from the original)
- Every feature uses only games played **before** the game being predicted.
- Shrink noisy rates toward league averages using pseudo-counts.
- Prefer simple GLM/OLS models plus simulation; avoid fancy ML because of the noise.
- Judge every model against naive baselines (Brier score, log loss, MAE) on held-out seasons, and against DraftKings where prices exist.
- Never hand-edit projection numbers. News enters only through override files, and narrative appears on game cards as text only.
- Market framing: DraftKings' line usually already knows what the model knows, so a gap is information, not a bet. Calls blend model and market.

## 1. Data
**nflverse via `nflreadpy`**, saved as parquet in `data/`:
- Play-by-play, 2016 onward. Refresh the current season on every run; earlier seasons once.
- Schedules (`sched.parquet`): spreads, totals, moneylines, roof, wind, temperature, QBs, coaches, referees and rest days.
- Rosters (weekly), players, injuries (official report), snap counts.
- FTN charting (2022 onward), which provides `is_interception_worthy` and other charting fields.
- Weekly player stats.

**Other sources:**
- **Odds:** The Odds API, DraftKings only, decimal. Two kinds of call:
  - Events list (0 credits).
  - Per-game calls: anytime TD (1 credit per game), game lines (1 per call), and over/under props (about 4 credits per game, only for games Clinton names).
  - Clinton must paste the URLs; Claude can't fetch any it builds itself. Raw responses go to `odds_raw/wk{W}/`, and the prior pull is archived first. `ingest_odds.py {W} "<timestamp>"` converts them. The account has 500 credits per season, so keep a ledger.
- **DraftKings DFS salaries:** pulled early in the week.
- **Weather:** within 48h of kickoff only. MySportsWeather is primary and Covers.com is the cross-check, written to `forecasts_wk{W}.csv`. Open-Meteo is blocked.
- **News / roles:** read every item on two feeds daily:
  - https://www.thefantasyfootballers.com/fantasy-football-news/
  - https://fantasydata.com/nfl/fantasy-football-player-news

  Supplement with web search for injuries and QB starters.

## 2. News overrides: `news_overrides_wk{W}.json`
```
asof, source,
out: [names]                       # removed; volume redistributed
p_play: {name: prob}               # questionable / expected
qb_start: {TEAM: passer_id/name}   # starter overrides
next_man: {...}                    # an unknown back promoted
lead_back: {TEAM: [name, share]}   # role among known backs
back: [names]                      # activated / returning
ol_out: [...]                      # missing O-line starters
team_notes: {TEAM: text}           # display only
```
**Rules:**
- **48-hour freshness:** re-verify or drop any entry older than 48 hours.
- A new week's file starts as a **copy** of last week's, never blank.
- For every team with a back marked `out`, confirm the lead back and set `lead_back`, because the model's default even committee split is usually wrong.

**Staleness audit** (`audit_stale.py W` → `audit_wk{W}.txt`) checks:
- QB starter mismatches
- roster and injury conflicts
- carryovers (prior absences that need confirming)
- ROLE flags on running backs
- OL continuity
- weather

It also runs a **market role check**: flag a player when DK's TD price implies a role change the model missed.

## 3. Models

### 3a. Game lines (`lines_project.py`)
- Two OLS regressions on every game since 2016, one for **margin** and one for **total**.
- Inputs: team ratings (offense and defense, opponent-adjusted), QB adjustments (including `qb_change`), weather (wind, dome), rest, divisional games, injuries, and success-rate/pace features (added Oct 16–18).
- **Main job: each team's expected points**, which drive TD chances and player volume. When market lines exist, team totals blend model and market.
- Outputs per game: `m_margin`, `m_total`, `p_home_cover`, `p_over`, `sd_margin`, `sd_total`. Probabilities were **calibrated** after the raw ones proved overconfident.
- **Team totals:** for each team, expected TDs and rush share; projected passing (attempts, completions, yards, INTs, sacks), rushing (carries, yards), plays and yards.
- *(orig)* On 1,959 games from 2019–26, DK beats the model on both margin MAE (9.81 vs 10.13) and total MAE (10.37 vs 10.54). At 3+ point edges the model's side covers 51.7% on spreads and 53.2% on totals, against 52.4% to break even.

### 3b. QB interceptions (`int_model.py`; pilot-recovered v1)
- **Attempts:** OLS on Vegas total, spread, the QB's decayed attempts per start, team pass volume and home field.
- **Interception rate:** Poisson GLM per attempt, with actual attempts as the exposure term.
- **Inputs, by current weight:**

  | Input | Weight |
  |---|---|
  | Expected pass attempts | 20% |
  | Passes broken up | 19% |
  | Game script (spread) | 15% |
  | Throw depth (aDOT) | 14% |
  | INT-worthy throws (FTN) | 12% |
  | Defense INT rate | 11% |
  | QB inexperience | 5% |
  | Pass rush (sacks + QB hits for both teams) | 3% |
  | Wind | 1% |

- **Shrinkage pseudo-counts:** QB INT rate 600 attempts, defense INT rate 1,500, pressure 150, aDOT 150.
- **Decay:** 0.97 per game for QBs, 0.90 for teams.
- **Output:** P(1+ INT) = 1 − e^(−expected INTs).
- *(orig)* Backtest Brier on 2024 through 2026 Week 3: 0.2446, against 0.2498 for a constant guess (about 2% skill).

### 3c. Anytime TD (`td_model.py`, `td_project.py`)
- Expected team TDs come from the implied team total (with `rush_share`), then are split among players by:

  | Input | Weight |
  |---|---|
  | Target share | 39% |
  | Implied points | 23% |
  | Carry share | 17% |
  | Team run/pass TD split | 8% |
  | Carries inside the 10 | 6% |
  | Past TD share | 5% |
  | Red-zone targets | 1% |
  | Opponent TDs allowed to the position | 1% |

- Each team's players are scaled so they add up to the team total.
- Per-player fields: `p`, `xtd`, `fair`, `due`, `gtd`, `stds`, `otdg`/`otdrk` (opponent TDs allowed and rank), `i10`, `rz`, plus DK and FanDuel prices.
- *(orig)* Brier 0.1535 against 0.1670 for a constant guess. QBs run about 2 points high.
- **Known weakness:** longshot-tail "edges." Only report edges in the 2x–4x range. Clinton bets 2.5–3.9.

### 3d. Player projections (`proj_model.py`, `proj_project.py`, `sim_model.py`)
- **Per-player means:** usage (targets, carries, QB attempts), snap share and its trend (`snap_dn`), WOPR, efficiency (yards per play, catch rate), opponent defense (DvP-style ratio), team recent play volume, Vegas total and spread, QB change, interceptions, and TD chances. Success-rate and pace adjustments were added Oct 16–18.
- **Team volume scaling:** listed players get 97% of team volume and depth players get the rest. A receiver team-scaling bias was found and fixed. `redistrib` warnings fire when a receiving scale is ≥1.3 or a carry scale is ≥1.5.
- **Injury handling:**
  - Out, Doubtful, IR and inactive players are removed and their volume goes to teammates.
  - **Questionable players** play in a share of sims based on their last practice (`p_play`):

    | Last practice | Game status set | Midweek |
    |---|---|---|
    | Did not practice | 48% | 18% |
    | Limited | 72% | 66% |
    | Full | 85% | 88% |

  - **Early week:** players who were Out last week return 31% of the time; Doubtful, 45%.
  - **Back from injury:** 77–87% of usual volume for one game.
  - **Next man up:** if the lead back is out and no backup has 25%+ of carries, add the top healthy back at a 12% carry share.
  - **QB change:** affects receivers' volume and efficiency, not target distribution.
- **Simulation:** 6,000 sims per game, with teammates sharing team volume and team TDs.
  - `share_sd`: targets 0.15, carries 0.5
  - `eff_sd` 0.3, `exit_p` (early exit) 0.03, `td_overdisp` 1.35
  - **Outputs:** mean, 21 quantiles (0.02 through 0.98), floor/ceiling at p10/p90, boom/bust probabilities (cutoffs below), and `var.ppr`/`var.std`

    | Position | Boom at (PPR) | Bust at |
    |---|---|---|
    | QB | 25 | 12 |
    | RB | 20 | 6 |
    | WR | 20 | 6 |
    | TE | 15 | 4 |

- **Recalibration** (`recal`): linear `[a, b]` per stat (`fp_legends`, `pass_yds`, `rec`, `rec_yds`, `rush_yds`, `_pooled`).
- **Scoring:**
  - **PPR:** 1 per catch, 0.1 per yard, 6 per TD, 0.04 per passing yard, 4 per passing TD, −2 per INT.
  - **Std:** the same without catches.
  - **DK:** DraftKings DFS scoring, including the 3-point yardage bonuses.
  - **`fp_legends`:** Clinton's dynasty league scoring.
- **DFS fields:** `dk_sal`, `dk_pts`, `dk_val` (points per $1,000 of salary) and `dk_cval` (the same at the ceiling).
- **Consistency fields:** quality starts and duds (description only), last-3 history per stat.
- *(orig)* Held-out fantasy-point miss beats the player's season average by 12% (QB), 4% (RB) and 5% (WR, TE). Ranges land inside p10–p90 78–84% of the time.

### 3e. Same-game parlay joint probability (`parlay_joint.py`)
Correlation-aware joint probability built from historical correlation, validated on held-out 2024–26 data.

### 3f. Board consistency (DK vs DK, no model)
Compare DK's own lines with each other:
- moneyline vs spread
- team TD total vs implied points
- player TD price vs usage ladders

Also: a line-movement scorer (how often the line moves toward or away from the model after publishing).

## 4. Bets tracked (paper), graded every week
**Families:**
- TD singles: the skip list (not bet), the 2.5–3.9 band the model likes, and the likeliest scorer per game
- TD parlays across games: likeliest per game at 2 and 3 legs; top-5 totals at 2 and 3 legs; the top-5 favorites' lead RB; both teams' shortest in the top-total game
- **ML + top TD scorer, 2 legs:** started Week 6, and Clinton dictates the picks
- Same-game stacks: favorite + lead RB; top two pass catchers
- **Matchup props:** the opponent is top-8 in fantasy points allowed to the position AND allows at least the league-average TD rate. Enforced in `paper_slate.py` as its own graded group.
- Receptions (model differs by 5+ points) and alt-line parlays
- Game lines: totals and moneylines where the model differs by 3–4+ points
- Board consistency
- Interceptions: 0.5 line, model differs by 5+ points
- Every TD price by band (tracking only): under 2.5, 2.5–3.9, 4.0–6.0, over 6.0
- **Control:** random picks

**Per bet:** `week, type, bet, price, model_p, result, actual, pnl`, flat 1-unit stakes.

**Summaries:**
- `strategy`: n, wins, pending, hit rate, model rate, breakeven, units and a read ("Too early", "Earning trust", "No edge yet", "Don't bet")
- `verdicts` per market
- `weekly` accuracy (MAE vs DK, Brier vs DK, ATS and O/U)

**Scorecard** (`scorecard.py`): grades every published number, not just bets — averages and bias, range coverage, probability calibration buckets, games, and a by-week view. A test ledger records about 45 tests run this season, with a season-end holdout plan.

**Private bet log:** `bets/my_bets.csv` holds Clinton's real bets and DFS lineups. It is NEVER published.

## 5. Site (single self-contained HTML artifact)
- **Six tabs:** Games, Players, Trends, Season, Record, Model. The home page has 4 cards. The page was cut from 3.8MB to about 1.5MB, and is now about 1.3MB.
- **Payload:** `const ALL = {weeks, default, data:{W:{...}}, trends, awards, season, record, scorecard}`.
- **Per week:**
  - `int`, `td`, `dvp` (cur/prev), `lines` (`by_spread`, `by_total`, `combos`, `games`, `teams` ATS/OU, `situations`, `league_ou`, `summary`, `sd_*`, `team_totals`)
  - `weather`, `line_hist`, `proj`, `how` (Model tab HTML), `news_notes`, `board` (`value`, `fades`, `rec`, `info`), `ats_rec`, `weights` (input-importance bars), `games_in`, `through`
- **Trends:** about 25 reference tables from 2016–2025, each with a verdict line:
  - Lines: key numbers, push rates, totals
  - Teasers and moneylines by spread
  - Teams: luck and its reversal, home field, quarters, decisions
  - People: coaches ATS, QB starts, first three career starts, referees
  - Spots: situational records, wind, time slot, surface, temperature
  - Players: TD rate by role and implied points, 100-yard rates
- **Season:** win-total distributions, division, playoff, #1-seed, conference and Super Bowl odds, strength of schedule and market line.
- **Awards:** MVP, OPOY, DPOY, OROY, DROY and COY, using historical winner traits.
- **Design:**
  - Minimalist with a pink accent.
  - Team-color win bars, with OKLab color distance used to keep team pairs distinguishable.
  - Decimal odds everywhere. DK prices appear in small orange type.
  - Mobile-first and also good on desktop.
- **Checks:** a headless page check (`nav.js`) clicks through the tabs and reports console errors before publishing.

## 6. Schedule (target, all run from this repo)
- **Daily 8:52 a.m. ET:** a fresh session clones the repo, then:
  1. Refreshes data.
  2. Diffs injuries against the overrides.
  3. Runs the news and role feeds.
  4. Pulls weather if within 48h.
  5. Rebuilds the current week (and next week if it has lines).
  6. Runs the audit and page checks.
  7. Publishes, commits data and overrides, and sends a digest: new outs and returns, role changes, projection moves >1.5 points or ATD swings >15%, forecast changes, unresolved audit items. If nothing changed, two sentences.
- **Game mornings:** the odds pull. The session gives Clinton the URL block, he pastes it, then ingest, rebuild, publish. Report edges in the 2x–4x range only.
- **Tracker:** snapshots before each slate; Tuesday grade and recap.

## 7. Rebuild order (proposed)
1. Data layer (`pipeline/data.py`) plus game lines and team totals. Everything downstream depends on implied points.
2. TD model.
3. Player projections, simulation and injury logic.
4. Site v1: Games and Players tabs, built from the published HTML's markup and CSS as the design reference.
5. Overrides, audit and daily scheduled task. **Cutover point:** the new daily run takes over the URL and the old chain is retired.
6. Record and paper bets, then Trends, Season and Awards. INT model last, since INT props are no longer bet.

Each step ships with a backtest against the *(orig)* numbers above.
