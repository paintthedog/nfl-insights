# NFL tracker tick — standing instructions

Runs in Clinton's chat on four days a week. All times are America/New_York.
- Thursday 10:41: snapshot, before Thursday night's game.
- Sunday 12:44: snapshot, before the 1 p.m. games and after the morning site update.
- Monday 10:41: snapshot, before Monday night's game.
- Tuesday 10:41: snapshot, then grade the week and send the recap.

## Steps

1. **Schedule the next tick first**, so the chain never breaks. Call `send_later` with the message "NFL tracker tick — follow /home/claude/nfl-tracker/TICK.md". Pick the time from this sequence:
   - Thursday → Sunday 12:44
   - Sunday → Monday 10:41
   - Monday → Tuesday 10:41
   - Tuesday → Thursday 10:41

   Get the current time with `current_time`, then convert the next tick to UTC.

2. **Snapshot the site.** Read it with the Artifact tool: action "read", path "index.html", url https://claude.ai/artifact/MVrDv9jyZBGoEnfbcnYSZH (the pinned "NFL Insights" artifact the daily update publishes to). If that read fails, use https://claude.ai/code/artifact/a5fdeeee-5678-41f4-90b1-b438d179addc instead. Then run `python3 /home/claude/nfl-tracker/tracker.py snap <saved index.html path>`.
   - Look at the site's `asof` timestamp. If it's more than about 30 hours old, the daily update has probably stopped running, and you must tell Clinton (see the note on the daily update below).

3. **Thursday, Sunday or Monday: stop here silently.** Don't send a message unless something broke.

4. **Tuesday only.** Run `python3 /home/claude/nfl-tracker/tracker.py grade`. Then send the recap with `SendUserMessage`, sized to fit about two phone screens:
   - Start with a one-line headline verdict for the week.
   - "What worked" and "What didn't": bullets by bet family with W–L and units, taken from the site's `record.thisweek`. Add a sample-size caveat wherever n is small.
   - "What we learned": 3–5 numbered takeaways covering model vs DK accuracy, calibration misses by probability bucket, projection bias, how the fade/skip lists did, how the random-pick control did, and how the newest test strategies did against expectation.
   - "Season accuracy": draw this from `out/summary.json`.
     - Rank the stats by relative miss.
     - Name the 3 most reliable and 3 least reliable player projections (3+ games).
     - Say whether the model is getting better week over week.
     - Report how often PPR results land outside the floor-to-ceiling range.
   - Call longshot TD hits (over 6.0) variance, not strategy.
   - End with at most one concrete adjustment worth considering.

   Tone: blunt and realistic, with some encouragement where it's earned. Use decimal odds only. Never include Clinton's private bet log.

   If Monday night's game isn't in the nflverse stats yet, say so in one line, and grade it on the following Thursday tick.

## Notes

- Grading uses each player's latest snapshot taken before his game's kickoff.
- Tracking started in Week 6, with only Monday night's game; Week 7 is the first full week. Earlier weeks can't be backfilled because the site drops old projections.
- The season log lives in `out/season_log.csv` and is rebuilt from the snapshots on every grade.
- **The daily update** runs in a separate conversation that hit its length limit. Its instructions can't be changed. It was still publishing as of Oct 19. If it stops, tell Clinton right away. Its pipeline code exists only in that conversation's workspace.
