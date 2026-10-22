"""Defense vs position: what each defense allows per game to QBs, RBs, WRs and TEs.

Fantasy points are PPR: 1 per catch, 0.1 per rush/rec yard, 6 per rush/rec TD,
0.04 per passing yard, 4 per passing TD, -2 per interception.
Regular season only; two-point tries excluded.
"""
import numpy as np, pandas as pd

COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "two_point_attempt",
        "passer_player_id", "rusher_player_id", "receiver_player_id", "pass_attempt", "sack",
        "complete_pass", "passing_yards", "rushing_yards", "receiving_yards",
        "pass_touchdown", "rush_touchdown", "interception", "rush_attempt", "qb_kneel"]
POS = ["QB", "RB", "WR", "TE"]


def dvp(seasons, players):
    pbp = pd.concat([pd.read_parquet(f"data/pbp_{s}.parquet", columns=COLS) for s in seasons])
    p = pbp[(pbp.season_type == "REG") & (pbp.two_point_attempt != 1) & pbp.defteam.notna()].copy()
    pos = players.drop_duplicates("gsis_id").set_index("gsis_id").position.replace({"FB": "RB", "HB": "RB"})
    f = lambda c: p[c].fillna(0)

    # passing (credited to the passer)
    pa = p[p.passer_player_id.notna() & (p.pass_attempt == 1) & (p.sack != 1)]
    passing = pd.DataFrame({"game_id": pa.game_id, "season": pa.season, "defteam": pa.defteam,
                            "pos": pa.passer_player_id.map(pos),
                            "pass_att": 1, "pass_yds": pa.passing_yards.fillna(0),
                            "pass_td": pa.pass_touchdown.fillna(0), "ints": pa.interception.fillna(0)})
    # rushing
    ru = p[p.rusher_player_id.notna() & (p.rush_attempt == 1) & (p.qb_kneel != 1)]
    rushing = pd.DataFrame({"game_id": ru.game_id, "season": ru.season, "defteam": ru.defteam,
                            "pos": ru.rusher_player_id.map(pos), "car": 1,
                            "rush_yds": ru.rushing_yards.fillna(0), "rush_td": ru.rush_touchdown.fillna(0)})
    # receiving
    rc = p[p.receiver_player_id.notna() & (p.pass_attempt == 1) & (p.sack != 1)]
    receiving = pd.DataFrame({"game_id": rc.game_id, "season": rc.season, "defteam": rc.defteam,
                              "pos": rc.receiver_player_id.map(pos), "tgt": 1,
                              "rec": rc.complete_pass.fillna(0), "rec_yds": rc.receiving_yards.fillna(0),
                              "rec_td": rc.pass_touchdown.fillna(0)})
    allp = pd.concat([passing, rushing, receiving]).fillna(0)
    allp = allp[allp.pos.isin(POS)]
    by = allp.groupby(["season", "defteam", "pos", "game_id"]).sum(numeric_only=True).reset_index()
    games = p.groupby(["season", "defteam"]).game_id.nunique().rename("games")
    tot = by.groupby(["season", "defteam", "pos"]).sum(numeric_only=True).reset_index().merge(games, on=["season", "defteam"])
    stats = ["pass_att", "pass_yds", "pass_td", "ints", "car", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"]
    for c in stats:
        tot[c] = tot[c] / tot.games
    tot["fp"] = (tot.pass_yds * 0.04 + tot.pass_td * 4 - tot.ints * 2 + tot.rush_yds * 0.1 + tot.rush_td * 6
                 + tot.rec * 1 + tot.rec_yds * 0.1 + tot.rec_td * 6)
    tot["td"] = tot.rush_td + tot.rec_td + np.where(tot.pos == "QB", tot.pass_td, 0)
    # rank 1 = allows the MOST (softest matchup for that position)
    tot["fp_rank"] = tot.groupby(["season", "pos"]).fp.rank(ascending=False, method="min").astype(int)
    tot["lg_fp"] = tot.groupby(["season", "pos"]).fp.transform("mean")
    return tot[["season", "defteam", "pos", "games", "fp", "fp_rank", "lg_fp", "td"] + stats]


if __name__ == "__main__":
    pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"])
    t = dvp([2025, 2026], pl)
    pd.set_option("display.width", 200)
    print(t[(t.season == 2026) & (t.pos == "RB")].sort_values("fp_rank").head(6).round(1).to_string(index=False))
    print(t.groupby(["season", "pos"]).fp.mean().round(1))
