"""
Compare TD model to anytime-TD prices you paste in.

odds file (td_odds_wk<week>.csv), one row per player:
    name,yes,no,book
    Jahmyr Gibbs,-290,,bet365
    James Cook,-105,-125,DraftKings
'no' is optional; most books only post the Yes side. When it's missing, the
market's fair probability is estimated by stripping an assumed Yes-side margin.
"""
import sys, numpy as np, pandas as pd

week = sys.argv[1] if len(sys.argv) > 1 else "4"
YES_ONLY_MARGIN = 0.06   # typical overround baked into a Yes-only anytime TD price
BLEND = 0.5              # trust the model halfway until it's proven against closing lines

dec = lambda a: np.where(a > 0, 1 + a / 100, 1 + 100 / -a)
proj = pd.read_csv(f"td_props_2026_wk{week}.csv")
odds = pd.read_csv(f"td_odds_wk{week}.csv")
m = odds.merge(proj, on="name", how="left", indicator=True)
unmatched = m[m._merge == "left_only"].name.tolist()
m = m[m._merge == "both"].copy()

imp_yes = pd.Series(1 / dec(m.yes.astype(float)), index=m.index)
if "no" in m and m.no.notna().any():
    no = m.no.astype(float)
    imp_no = pd.Series(np.where(no.notna(), 1 / dec(no.fillna(-110)), np.nan), index=m.index)
    two_way = imp_yes / (imp_yes + imp_no)
else:
    two_way = pd.Series(np.nan, index=m.index)
m["mkt_p"] = np.where(two_way.notna(), two_way, imp_yes / (1 + YES_ONLY_MARGIN))
m["blend_p"] = BLEND * m.p_td + (1 - BLEND) * m.mkt_p
m["edge_pts"] = (m.p_td - m.mkt_p) * 100
m["ev_yes"] = m.blend_p * dec(m.yes.astype(float)) - 1
m["call"] = np.select([m.ev_yes >= 0.05, m.ev_yes >= 0.02], ["bet", "lean"], "pass")
out = m.sort_values("ev_yes", ascending=False)[
    ["name", "pos", "posteam", "opp", "p_td", "mkt_p", "blend_p", "edge_pts", "yes", "book", "ev_yes", "call", "flag"]]
out.to_csv(f"td_edges_wk{week}.csv", index=False)
pd.set_option("display.width", 200)
print(out.round({"p_td": 3, "mkt_p": 3, "blend_p": 3, "edge_pts": 1, "ev_yes": 3}).to_string(index=False))
if unmatched:
    print("\nNot in projections (check spelling or the player wasn't projected):", unmatched)
