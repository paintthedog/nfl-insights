"""Injury adjustments. Every number here was measured on 2019-2026 games by inj_test.py / inj_test2.py; rerun those to refresh.

  P_PLAY     chance a Questionable RB/WR/TE plays, by his last practice of the week (2,027 cases; 69% overall)
  BACK_MULT  usage in the first game back relative to projection, by team games missed (second game back is ~normal)
  POOL_FRAC  share of a team's targets / non-QB carries that goes to the listed pool. With returning players listed, 2.2%
             still goes to players with no earlier game for the team this season, and a little more to fringe players
             below the listing cutoff, so the pool is scaled to 97% of the team, not 100%
Tested and rejected: role-aware redistribution of an absent player's volume (no better than scaling the pool), and a
usage discount for Questionable players who do play (0.97, not significant)."""
P_PLAY = {"DNP": 0.48, "Limited": 0.72, "Full": 0.85, None: 0.69}
PRACTICE = {"Full Participation in Practice": "Full", "Limited Participation in Practice": "Limited", "Did Not Participate In Practice": "DNP"}
POOL_FRAC = 0.97


def back_mult(missed):
    return 1.0 if missed < 1 else 0.87 if missed < 2 else 0.855 if missed < 4 else 0.775


def p_play(practice_status):
    return P_PLAY.get(PRACTICE.get(practice_status), P_PLAY[None])


# Early in the week (before the new report): chance a skill player plays his NEXT game given last week's final status
# (2019-25: Out 1,930 cases, Doubtful 299, Questionable-and-sat 918). Used until this week's statuses are posted.
CARRY = {"Out": 0.31, "Doubtful": 0.45, "Q_sat": 0.52}
CARRY2 = {"Out": 0.49, "Doubtful": 0.64, "Q_sat": 0.60}       # ... and the game after that (look-ahead week)
CARRY_LBL = {"Out": "out last wk", "Doubtful": "doubtful last wk", "Q_sat": "sat last wk"}

# Midweek, before game statuses exist: chance a QB/RB/WR/TE listed on the practice report plays, by practice status
# (2019-25, 11,800 player-weeks, final practice of the week). Thursday's status is earlier than the final one, so a
# Thursday DNP tends to understate his chances slightly. "Not injury related - resting player" is treated as healthy.
PRAC_MID = {"DNP": 0.18, "Limited": 0.66, "Full": 0.88}
PRAC_LBL = {"DNP": "no practice", "Limited": "limited practice", "Full": "full practice"}
def p_mid(practice_status):
    return PRAC_MID.get(PRACTICE.get(practice_status))

