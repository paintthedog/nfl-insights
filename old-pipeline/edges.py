"""Compare model P(1+ INT) with book prices. Prices = best available per side (ScoresAndOdds, Sep 30 2026)."""
import sys, numpy as np, pandas as pd
week = sys.argv[1] if len(sys.argv) > 1 else "4"
proj = pd.read_csv(f"int_props_2026_wk{week}.csv")
odds = pd.read_csv(f"odds_wk{week}.csv")
dec = lambda a: np.where(a > 0, 1 + a / 100, 1 + 100 / -a)
m = proj.merge(odds, on="name", how="outer", indicator=True)
missing = m[m._merge != "both"][["name", "_merge"]]
m = m[m._merge == "both"].copy()
m["imp_over"], m["imp_under"] = 1 / dec(m.over), 1 / dec(m.under)
m["mkt_p"] = m.imp_over / (m.imp_over + m.imp_under)          # no-vig market probability
m["edge_pts"] = (m.p_int - m.mkt_p) * 100                     # + = model likes the OVER
# The model has never been tested against market prices, so trust it only halfway:
# blend 50/50 with the no-vig market number before computing EV.
BLEND = 0.5
m["blend_p"] = BLEND * m.p_int + (1 - BLEND) * m.mkt_p
m["ev_over"] = m.blend_p * dec(m.over) - 1                    # EV per $1 at best price
m["ev_under"] = (1 - m.blend_p) * dec(m.under) - 1
m["side"] = np.where(m.ev_over > m.ev_under, "OVER", "UNDER")
m["ev"] = m[["ev_over", "ev_under"]].max(axis=1)
m["price"] = np.where(m.side == "OVER", m.over, m.under)
m["book"] = np.where(m.side == "OVER", m.over_book, m.under_book)
m["call"] = np.select([m.ev >= 0.05, m.ev >= 0.02], ["bet", "lean"], "pass")
out = m.sort_values("ev", ascending=False)[["name", "posteam", "matchup", "p_int", "mkt_p", "blend_p", "edge_pts",
                                            "side", "price", "book", "ev", "call"]]
out.to_csv(f"edges_wk{week}.csv", index=False)
pd.set_option("display.width", 200)
print(out.round({"p_int": 3, "mkt_p": 3, "blend_p": 3, "edge_pts": 1, "ev": 3}).to_string(index=False))
if len(missing): print("\nUnmatched:\n", missing.to_string(index=False))
