# NFL Insights — project notes

Read this first in every chat. This repo is the single source of truth. Chats come and go; anything worth keeping is committed here.

## What this is
NFL Insights (also called "Paydirt") is Clinton's NFL projection and betting-research site, published as a claude.ai artifact: https://claude.ai/artifact/MVrDv9jyZBGoEnfbcnYSZH. Other people see it, so his personal bet log never goes on the site.

**Goal:** the most informed, likelihood-based bets possible, plus visibility into trends. Beating DraftKings is NOT the goal. Don't frame the work as edge-hunting, but stay honest when the market is more accurate than the model.

## Status (Oct 21, 2026)
- **The original pipeline is back.** Exported from the old chat on Oct 21 into `pipeline/` (code, config, small data). Large data and model files were left out of the export; `pipeline/bootstrap_data.py` rebuilds them from nflverse.
- **One command updates the site from a fresh workspace:** `bash pipeline/daily.sh` (picks the current week, adds next week once its lines exist). Output: `pipeline/site/paydirt.html`. Takes ~25 min from scratch, ~10 once data is cached.
- Oct 21 evening: Week 7 and 8 rebuilt and published as v152.
- Lost in the export: pre-kickoff archives before Oct 21 (the scorecard keeps its last numbers until new weeks are graded) and line-movement history (restarts Oct 21).
- Run state that must persist between runs is committed: `pipeline/archive/`, `pipeline/data/line_history.csv`, record and bet CSVs, override JSONs. Commit after every run.
- `SPEC.md` and `recover/` are now reference only.
- `tracker/` is superseded by the pipeline's own scorecard; not scheduled.
- Oct 22 fixes: traded and news-activated players now enter the projections; the overrides file is dated per entry and housekept by `auto_overrides.py`; four pre-registered main tests with fixed sample sizes on the Record tab; the scorecard grades market vs raw model vs blend; bootstrap rebuilds `tests/_wf_proj.pkl` again; superseded script copies moved to `pipeline/legacy/` (never edit those).
- Oct 22 (Clinton's request): **Steady rating** (`pipeline/steady.py`, computed in `proj_project.py`). 0–100 within position: half role spread (game-to-game share of team targets, or carries + targets for backs), half PPR spread; QBs on output only; 3+ games. ◆ flags a role change (teammate out, trade, return, recent share jump/drop, news lead back). Label and filter only, never a projection input. Backtest: among players with a real role it adds ~1–2 pts on prop rungs vs the model, not reliable (the bigger all-player effect came from fringe players). Treat it as a description, not an edge (TEST_LEDGER). New exploratory paper rule "Parlays: steady legs, rated 60+"; every steady leg is tagged with its score.

- Oct 25 (Clinton: not enough bets tested): `pipeline/make_urls.py W` now writes the game-morning Odds API block (never hand-write it; Week 7's dropped the pass-yds ladder and team totals, so QB yards + WR1 had 0 bets). Exploratory side rules in `pipeline/paper_extra.py` (teaser legs graded alone, wide teasers, underdog same-game stacks, matchup receptions without the TD filter); main tests unchanged. Teaser backtest 2016–2025 in TEST_LEDGER: only the +1.5 to +2.5 underdog band clears break-even (z +1.4).

- **Oct 25 (Clinton's decision): no more manual Odds API pulls.** The timing was his call, which made the record anecdotal. The site's job is predicting outcomes accurately and grading those predictions. Game spreads, totals and moneylines come free from nflverse (refreshed each daily run; closing lines after games) and stay as the benchmark for game predictions. Player numbers show the model's own figures (no DK prop blend from Week 8 on). Paper bets: `pipeline/paper_lines.py` logs game-line rules automatically at the last daily run before kickoff (teasers main + legs + wide, moneyline and total gaps, trend unders, control). Rules that needed prop or same-game prices are stopped (record.py RETIRED); their graded rows stay. `paper_slate.py`, `paper_extra.py`, `make_urls.py` are no longer run.

## Daily update (what the scheduled task does)
Follow `pipeline/UPDATE_CHECKLIST.md` steps 1–7 (never the odds pull, which is manual). In short: run `bash pipeline/daily.sh`; read the role feeds and fix `news_overrides_wk{W}.json`; resolve `audit_wk{W}.txt`; rebuild; publish `pipeline/site/paydirt.html` to the site URL; commit and push; send a short digest.

## Clinton's standing rules
- **Odds:** decimal only. **DraftKings** is the reference book (he also bets FanDuel; FanDuel won't allow INT props in a same-game parlay). Never scrape DK or FanDuel sites.
- **Odds pulls: stopped Oct 25.** Don't suggest or schedule them. (If Clinton ever restarts one: The Odds API, 500-credit account; Claude can only fetch URLs he pastes as plain text; `pipeline/make_urls.py` writes the block.)
- **Weather:** MySportsWeather is the primary source and Covers is the cross-check. Open-Meteo is blocked, so don't suggest it.
- **Props:** used mainly as conservative legs in 2–3-leg parlays targeting a 2x+ payout. Favor consistent players with the best matchups: the opponent must be top-8 in fantasy points allowed to that position AND allow at least the league-average TD rate.
- **Anytime TD:** focus on the 2.5–3.9 decimal range. 4.5+ is too random to bet.
- **INT props:** no longer bet. Still tracked.
- **Never** put a skill player in a lineup or bet without a real season record: 8+ catches for pass catchers, 25+ carries for backs.
- **DFS (DraftKings):** pull salaries early in the week.
  - No zero-catch players.
  - Stack = QB + exactly 2 of his pass catchers, and no more from that team.
  - At most one Eagle. No Saints while they're bad.
  - Follow established tournament practice; don't hand-tune an optimizer.
- **Test strategies tracked every week** (pre-registered, mechanical; see the checklist for exact rules):
  - "ML + top TD scorer, 2 legs" (started Week 6; picks the winner by the line-anchored chance since Oct 21)
  - matchup props, same-game stacks (incl. QB yards + WR1 catches), TD fades, a random-pick control
  - added Oct 21: small-underdog teasers, wind and prime-time unders, steady-leg parlays
  - cut Oct 21: top-5-total TD parlays, favorites' lead-RB TD parlays, the TD 4.0–6.0 band
- **What Clinton wants from the data:** find which predictions the model is genuinely good at, and the bets closest to structural "hacks" of the book: boosts and promos, DK vs FanDuel price shopping, key-number teasers, same-game parlays priced below their fair (correlated) price, and DK's board disagreeing with itself. The model has not beaten DK's lines on any market yet; say so when it matters.
- **Market data:** nflverse game lines only (free, automatic). Closing-line value for game-line paper bets comes from the same feed.
- When asked a raw question (e.g. "most TDs allowed to which position"), answer the raw count, not a relative-to-average number.
- Clinton bets throughout the week and may send screenshots of bets to judge against the projections.

## Recurring schedule (ET)
- **Daily 8:52 a.m.:** "NFL Insights daily update" scheduled task. Starts a fresh session from this repo each run; nothing chains or reschedules itself. Its prompt is the "Daily update" section above.
- **Tuesday:** the same run also sends the weekly recap (what worked, what didn't, what we learned, season accuracy).
- Oct 21: the old self-rescheduling tasks (daily update chain, tracker snapshots) were disabled. Never rebuild schedules as send_later chains tied to one chat.

## Repo layout
- `pipeline/` — the whole site pipeline: data pulls, models, site build (`daily.sh` is the entry point)
- `pipeline/legacy/` — superseded copies of scripts, reference only
- `pipeline/archive/` — pre-kickoff copies of every published week (feeds the scorecard; commit them)
- `tracker/`, `recover/`, `SPEC.md` — reference only
- Clinton's private bet log is NOT in this repo (the repo is public). Never add it.

## How to work here
- Keep chats short and focused on one topic. Commit at the end of any change.
- Put notes and decisions in this file, not in chat.
