"""
lookup.py — helper for human annotation
Shows the full fact sheet of one match from StatsBomb data, so claims can be checked without remembering the game.

Usage:
  python lookup.py "Spain vs England"          # part of the match name is enough
  python lookup.py "Spain vs England" Kane     # only lines that mention a word (player, team, 'card'...)
  python lookup.py "Argentina vs Canada semi"  # add the stage when two teams met twice
"""
import json
import sys
from pathlib import Path

import pandas as pd

GT_DIR = Path("data/ground_truth")


def find_match(query):
    idx = pd.read_csv(GT_DIR / "matches_index.csv")
    idx["name"] = idx.home + " vs " + idx.away
    q = query.lower()
    full = (idx.name + " " + idx.competition + " " + idx.stage).str.lower()
    words = q.replace("(", " ").replace(")", " ").replace(",", " ").split()
    hits = idx[full.apply(lambda f: all(w in f for w in words if w != "vs"))
               & full.apply(lambda f: any(t.lower() in f for t in [q.split(" vs ")[0].strip()]))]
    if hits.empty:
        sys.exit(f"No match found for '{query}'. Try e.g. \"Spain vs England\".")
    if len(hits) > 1:
        print("Several matches found, be more specific:")
        for r in hits.itertuples():
            print(f"  {r.name} ({r.competition}, {r.stage})")
        sys.exit()
    return hits.iloc[0]


def fact_sheet(m):
    gt = json.loads((GT_DIR / f"gt_{m.match_id}.json").read_text(encoding="utf-8"))
    ev_all = pd.read_csv(GT_DIR / f"events_{m.match_id}.csv", low_memory=False)
    ev = ev_all[ev_all.period < 5]
    lines = (GT_DIR / f"summary_{m.match_id}.txt").read_text(encoding="utf-8").splitlines() + ["", "Cards:"]
    for c in ["foul_committed_card", "bad_behaviour_card"]:
        if c in ev:
            for r in ev[ev[c].notna()].itertuples():
                lines.append(f"  {int(r.minute) + 1}' {r.player} ({r.team}) - {getattr(r, c)}")
    lines += ["", "Substitutions:"]
    subs = ev[ev.type == "Substitution"]
    for r in subs.itertuples():
        lines.append(f"  {int(r.minute) + 1}' {r.team}: {r.substitution_replacement} on for {r.player}")
    lines += ["", "Shots (minute, player, outcome):"]
    for r in ev[ev.type == "Shot"].itertuples():
        pen = " (penalty)" if getattr(r, "shot_type", None) == "Penalty" else ""
        lines.append(f"  {int(r.minute) + 1}' {r.player} ({r.team}) - {r.shot_outcome}{pen}")
    so = ev_all[(ev_all.period == 5) & (ev_all.type == "Shot")]
    if len(so):
        lines += ["", "Penalty shootout (in kick order):"]
        for i, r in enumerate(so.itertuples(), 1):
            lines.append(f"  kick {i}: {r.player} ({r.team}) - {r.shot_outcome}")
    lines += ["", "All players (goals, assists, shots, key passes, passes):"]
    for p in sorted(gt["players"], key=lambda p: (p["team"], p["player"])):
        lines.append(f"  {p['player']} ({p['team']}): goals {p['goals']}, assists {p['assists']}, "
                     f"shots {p['shots']}, key passes {p['key_passes']}, passes {p['passes']}")
    return lines


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    match = find_match(sys.argv[1])
    out = fact_sheet(match)
    if len(sys.argv) > 2:
        words = [w.lower() for w in sys.argv[2:]]
        out = [l for l in out if any(w in l.lower() for w in words)]
    print(f"=== {match['name']} ({match.competition}, {match.stage}, {match.date}) ===")
    print("\n".join(out))