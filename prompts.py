"""
prompts.py — Phase 2a (M3 Prompt builder)
Builds the prompt for each input condition. Every condition shares the same
system prompt and task wording, so the ONLY difference is the data supplied.

  A  no data            -> match name only (memory baseline)
  B  raw events         -> key match events, no totals computed for the model
  C  stats summary      -> compact summary from ground_truth.py
  D  summary + rule     -> C plus an explicit grounding instruction

Usage (preview a prompt):
    python prompts.py 3869685 C
"""
import json
import sys
from pathlib import Path

import pandas as pd

GT_DIR = Path("data/ground_truth")
CONDITIONS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = "You are a football journalist writing post-match reports for a sports news website."

TASK = (
    "Write a match report of about 250 words for the match below. "
    "Cover the result, the goals, the key players and how the match was played. "
    "Write in plain prose with no headings or bullet points."
)

GROUNDING_RULE = (
    "IMPORTANT: Use only facts that appear in the match data above. "
    "Do not add any player, number, minute or event that is not in the data. "
    "When you mention a statistic, use the exact number from the data. "
    "If something is not in the data, leave it out."
)

# Event types kept for condition B (everything else, e.g. thousands of passes, is dropped)
KEY_EVENTS = {"Shot", "Own Goal Against", "Foul Committed", "Bad Behaviour", "Substitution",
              "Half Start", "Half End"}


def load_gt(match_id):
    return json.loads((GT_DIR / f"gt_{match_id}.json").read_text(encoding="utf-8"))


def match_header(gt):
    return (f"Match: {gt['home_team']} vs {gt['away_team']}\n"
            f"Competition: {gt['competition']}, {gt['stage']}\n"
            f"Date: {gt['date']}")


def raw_events_text(match_id):
    """Condition B: key events only, one line each, in match order. No totals are given."""
    ev = pd.read_csv(GT_DIR / f"events_{match_id}.csv")
    ev = ev[ev["type"].isin(KEY_EVENTS)
            | (ev.get("pass_goal_assist") == True)  # noqa: E712
            | (ev.get("pass_shot_assist") == True)  # noqa: E712
            | (ev.get("pass_type") == "Corner")]
    ev = ev[ev["period"] < 5]  # drop penalty shootout kicks, they are listed separately
    ev = ev.sort_values(["period", "minute", "second"], kind="stable")
    lines = ["period,minute,second,team,player,event,detail"]
    for r in ev.itertuples():
        detail = ""
        if r.type == "Shot":
            detail = f"{getattr(r, 'shot_type', '')} shot, outcome: {r.shot_outcome}"
        elif r.type == "Pass":
            kind = "goal assist" if getattr(r, "pass_goal_assist", False) == True else (  # noqa: E712
                   "shot assist" if getattr(r, "pass_shot_assist", False) == True else "")  # noqa: E712
            detail = ", ".join(x for x in [str(r.pass_type) if isinstance(r.pass_type, str) else "", kind] if x)
        elif r.type == "Foul Committed" and isinstance(getattr(r, "foul_committed_card", None), str):
            detail = r.foul_committed_card
        elif r.type == "Bad Behaviour" and isinstance(getattr(r, "bad_behaviour_card", None), str):
            detail = r.bad_behaviour_card
        elif r.type == "Substitution" and isinstance(getattr(r, "substitution_replacement", None), str):
            detail = f"replaced by {r.substitution_replacement}"
        player = r.player if isinstance(r.player, str) else ""
        lines.append(f"{r.period},{r.minute},{r.second},{r.team},{player},{r.type},{detail}")
    return "\n".join(lines)


def shootout_line(gt):
    so = gt.get("penalty_shootout")
    if not so:
        return ""
    h, a = gt["home_team"], gt["away_team"]
    return f"\nPenalty shootout result: {h} {so[h]} - {so[a]} {a}"


def build_prompt(match_id, condition):
    """Return {'system': ..., 'user': ...} for one match and condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    gt = load_gt(match_id)

    if condition == "A":
        data = match_header(gt)
    elif condition == "B":
        data = (match_header(gt) + shootout_line(gt)
                + "\n\nMatch events (key events only, in time order):\n" + raw_events_text(match_id))
    else:  # C and D
        data = (GT_DIR / f"summary_{match_id}.txt").read_text(encoding="utf-8")

    user = f"{TASK}\n\n--- MATCH DATA ---\n{data}\n--- END OF MATCH DATA ---"
    if condition == "D":
        user += "\n\n" + GROUNDING_RULE
    return {"system": SYSTEM_PROMPT, "user": user}


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "3869685"
    cond = sys.argv[2] if len(sys.argv) > 2 else "C"
    p = build_prompt(mid, cond)
    print("SYSTEM:", p["system"], "\n\nUSER:\n" + p["user"])
    print(f"\n[{len(p['user'].split())} words in user prompt]")