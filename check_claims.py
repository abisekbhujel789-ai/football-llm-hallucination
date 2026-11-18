"""
check_claims.py — Phase 3b (M6 Fact checker, rule-based part)
Checks every extracted claim against the ground truth. No LLM is used here.

Verdicts:
  correct       claim agrees with the ground truth
  incorrect     claim contradicts the ground truth  (error_type: numeric / player / timing / result)
  unverifiable  claim cannot be checked from the data (e.g. a stat we don't have)
  needs_human   tactical or other claims -> labelled by a human later (M7)

Usage:
  python check_claims.py --claims data/claims/gemini_gemini-3.1-flash-lite
Output: data/results/<folder>_claims.csv  (+ a short summary printed per condition)
"""
import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

GT_DIR = Path("data/ground_truth")
RESULTS_DIR = Path("data/results")

MINUTE_TOLERANCE = 2          # goal minutes can differ by 1-2 between sources (e.g. 45+2)
TEAM_STAT_KEYS = {            # claim stat -> (ground-truth key, tolerance)
    "shots": ("shots", 0), "shots_on_target": ("shots_on_target", 0), "xg": ("xg", 0.05),
    "passes": ("passes", 0), "pass_accuracy": ("pass_accuracy_pct", 1.0),
    "possession": ("pass_share_pct", 3.0),   # possession approximated by pass share (state in thesis)
    "corners": ("corners", 0), "fouls": ("fouls", 0),
    "yellow_cards": ("yellow_cards", 0), "red_cards": ("red_cards", 0),
}
PLAYER_STAT_KEYS = {"goals": ("goals", 0), "shots": ("shots", 0), "xg": ("xg", 0.05),
                    "assists": ("assists", 0), "key_passes": ("key_passes", 0), "passes": ("passes", 0)}


# ---------------- matching helpers ----------------
def norm(s):
    """Lowercase, strip accents and punctuation: 'Mbappé' -> 'mbappe'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def same_team(a, b):
    return bool(a and b) and (norm(a) in norm(b) or norm(b) in norm(a))


AMBIGUOUS = "__ambiguous__"


def find_player(name, gt):
    """Return the ground-truth common name for a player mentioned in a claim.
    None = no such player in this match; AMBIGUOUS = name fits several players (e.g. just 'Martinez')."""
    if not name:
        return None
    n = norm(name)
    # pass 1: exact match on any alias; pass 2: partial match (e.g. 'Kolo Muani' inside 'Randal Kolo Muani')
    for exact in (True, False):
        hits = set()
        for common, names in gt["player_aliases"].items():
            for alias in names:
                a = norm(alias)
                if (n == a) if exact else (len(n) > 3 and len(a) > 3 and (n in a or a in n)):
                    hits.add(common)
        if len(hits) == 1:
            return hits.pop()
        if len(hits) > 1:
            # prefer the player whose full common name is written in the claim ('Lautaro Martinez')
            full = [h for h in hits if norm(h) == n or norm(h) in n]
            return full[0] if len(full) == 1 else AMBIGUOUS
    # pass 3: every word of the claimed name appears in a full name
    # ('Eduardo Guerrero' -> 'Eduardo Antonio Guerrero Lozcano')
    words = set(n.split())
    if len(words) >= 2:
        hits = {common for common, names in gt["player_aliases"].items()
                if any(words <= set(norm(alias).split()) for alias in names)}
        if len(hits) == 1:
            return hits.pop()
        if len(hits) > 1:
            return AMBIGUOUS
    return None


PERIOD_ENDS = (45, 90, 105, 120)   # "90th minute" / "stoppage time" can mean any minute of added time
OUT_OF_SCOPE = ("tournament", "competition", "campaign", "career", "season", "international goal",
                "so far", "eliminat", "progress", "advance", "qualif", "knockout", "through to", "group winner")


def minute_ok(goal_min, claimed):
    if abs(goal_min - claimed) <= MINUTE_TOLERANCE:
        return True
    return claimed in PERIOD_ENDS and claimed <= goal_min <= claimed + 10


def is_existence(text):
    """'X provided an assist', 'Y scored a goal' -> at least one, not exactly one."""
    t = " " + norm(text) + " "
    return any(k in t for k in (" an assist ", " a goal ", " one assist ", " assisted ")) and " only " not in t


def out_of_scope(text):
    """Claims about the wider tournament or career cannot be checked from one match."""
    t = norm(text)
    return any(k in t for k in OUT_OF_SCOPE)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# ---------------- time / in-match helpers ----------------
TIME_WORDS = ("minute", "half", "stoppage", "injury", "added time", "extra time", "break", "interval",
              "late", "early", "start", "end")
IN_MATCH = ("after ", "made it", "make it", "level", "equalis", "equaliz", "half time", "halftime",
            "at the break", "interval", "lead ", "led ", "leading", "trailing", "went ", "put ")


def mentions_time(text):
    """Only trust a minute the extractor filled in if the claim itself talks about time."""
    t = " " + norm(text) + " "
    return any(ch.isdigit() for ch in t) or any(k in t for k in TIME_WORDS)


def half_window(text):
    """'before half-time' / 'in the first half' -> any minute of that half (incl. added time)."""
    t = norm(text)
    if any(k in t for k in ("first half", "before half", "before the break", "before the interval", "opening half")):
        return (1, 55)
    if any(k in t for k in ("second half", "after the break", "after half", "after the interval")):
        return (46, 100)
    return None


def in_match_score(text):
    """'2-2 after Mbappe's second goal' describes a moment in the match, not the final score."""
    t = " " + norm(text) + " "
    return any(k in t for k in IN_MATCH) and not any(k in t for k in (" final ", " full time ", " ended ", " finished "))


def running_scores(gt):
    """Every score the match passed through: 0-0, 1-0, 1-1, ..."""
    h, a = gt["home_team"], gt["away_team"]
    s, out = {h: 0, a: 0}, [(0, 0)]
    for g in gt["goals"]:
        s[g["team"]] += 1
        out.append((s[h], s[a]))
    return out


def goal_time_ok(goal_min, minute, window):
    if window:
        return window[0] <= goal_min <= window[1]
    return minute is None or minute_ok(goal_min, minute)

# ---------------- checks per claim type ----------------
def check_result(c, gt):
    h, a = gt["home_team"], gt["away_team"]
    sh, sa = gt["score"][h], gt["score"][a]
    so = gt.get("penalty_shootout")
    if sh != sa:
        winner, method = (h if sh > sa else a), ("extra_time" if gt.get("went_to_extra_time") else "normal_time")
    elif so:
        winner, method = (h if so[h] > so[a] else a), "penalties"
    else:
        winner, method = "draw", "normal_time"
    claimed = c.get("winner")
    if claimed:
        ok = (claimed == "draw" and winner == "draw") or same_team(claimed, winner)
        if not ok:
            return "incorrect", "result", f"winner was {winner}"
    if c.get("method") and c["method"] != method:
        return "incorrect", "result", f"decided in {method}"
    return ("correct", None, f"{winner} ({method})") if claimed or c.get("method") else ("unverifiable", None, "no winner stated")


def check_score(c, gt):
    h, a = gt["home_team"], gt["away_team"]
    true = gt.get("penalty_shootout") if c.get("shootout") else gt["score"]
    if not true:
        return "incorrect", "numeric", "there was no penalty shootout"
    hg, ag = num(c.get("home_goals")), num(c.get("away_goals"))
    if hg is None or ag is None:
        return "unverifiable", None, "score numbers missing"
    # lenient on order: extractors sometimes swap home/away
    if sorted([hg, ag]) == sorted([true[h], true[a]]):
        return "correct", None, ""
    if not c.get("shootout") and in_match_score(c.get("text", "")):
        if any(sorted([hg, ag]) == sorted(s) for s in running_scores(gt)):
            return "correct", None, "score at a moment during the match"
        return "incorrect", "numeric", f"match never stood at {int(hg)}-{int(ag)}"
    return "incorrect", "numeric", f"was {true[h]}-{true[a]}"


def check_goal(c, gt):
    goals = gt["goals"]
    text = c.get("text", "")
    player = find_player(c.get("player"), gt) if c.get("player") else None
    window = half_window(text)
    minute = num(c.get("minute")) if mentions_time(text) and not window else None
    if c.get("player"):
        if player == AMBIGUOUS:
            return "unverifiable", None, f"name '{c['player']}' fits several players"
        if player is None:
            return "incorrect", "player", f"{c['player']} did not play / not in data"
        mine = [g for g in goals if g["player"] == player]
        if not mine:
            return "incorrect", "player", f"{player} did not score"
        if not any(goal_time_ok(g["minute"], minute, window) for g in mine):
            return "incorrect", "timing", f"{player} scored at {[g['minute'] for g in mine]}"
        return "correct", None, ""
    if minute is not None or window:  # "a goal in the 81st minute" without a player
        near = [g for g in goals if goal_time_ok(g["minute"], minute, window)]
        if not near:
            return "incorrect", "timing", "no goal at that time"
        if c.get("team") and not any(same_team(c["team"], g["team"]) for g in near):
            return "incorrect", "player", f"goal was by {near[0]['team']}"
        return "correct", None, ""
    return "unverifiable", None, "no player or minute"


def check_team_stat(c, gt):
    key = TEAM_STAT_KEYS.get(c.get("stat"))
    value = num(c.get("value"))
    if not key or value is None:
        return "unverifiable", None, "stat/value not checkable"
    gt_key, tol = key
    team_claim = c.get("team")
    if not team_claim or norm(team_claim) in ("both", "total", "match", "both teams", "combined"):
        # whole-match total, e.g. "there were 30 shots in the match"
        if gt_key in ("pass_share_pct", "pass_accuracy_pct"):
            return "unverifiable", None, "percentage has no match total"
        true = round(sum(s[gt_key] for s in gt["team_stats"].values()), 2)
    else:
        team = next((t for t in gt["team_stats"] if same_team(team_claim, t)), None)
        if team is None:
            return "unverifiable", None, f"team '{team_claim}' not in this match"
        true = gt["team_stats"][team][gt_key]
    ok = abs(true - value) <= tol
    return ("correct", None, "") if ok else ("incorrect", "numeric", f"{gt_key}={true}")


def unrecorded_assists(gt, team):
    """Open-play goals of a team that have no assist recorded in StatsBomb (solo goals or missing links)."""
    goals = [g for g in gt["goals"] if same_team(team, g["team"]) and g["type"] == "goal"]
    recorded = sum(p["assists"] for p in gt["players"] if same_team(team, p["team"]))
    return max(len(goals) - recorded, 0)


def check_player_stat(c, gt):
    if not c.get("player"):
        # e.g. "Japan had 0.77 expected goals" mislabelled as a player stat
                names_team = c.get("team") and norm(c["team"]) in norm(c.get("text", ""))
                return check_team_stat(c, gt) if names_team else ("unverifiable", None, "no single player named")
    key = PLAYER_STAT_KEYS.get(c.get("stat"))
    value = num(c.get("value"))
    player = find_player(c.get("player"), gt)
    if not key or value is None:
        return "unverifiable", None, "stat/value not checkable"
    if player == AMBIGUOUS:
        return "unverifiable", None, f"name '{c.get('player')}' fits several players"
    if player is None:
        return "incorrect", "player", f"{c.get('player')} not in this match"
    row = next((p for p in gt["players"] if p["player"] == player), None)
    if row is None:
        return "unverifiable", None, "player had no recorded actions"
    gt_key, tol = key
    true = row[gt_key]
    if c.get("stat") in ("goals", "assists") and is_existence(c.get("text", "")):
        if true >= 1:
            return "correct", None, f"{player} {gt_key}={true} (at least one)"
        value = 1
    if abs(true - value) <= tol:
        return "correct", None, ""
    if gt_key == "assists" and value > true and value - true <= unrecorded_assists(gt, row["team"]):
        return "unverifiable", None, "assist not recorded in StatsBomb data"
    return "incorrect", "numeric", f"{player} {gt_key}={true}"


CHECKS = {"result": check_result, "score": check_score, "goal": check_goal,
          "team_stat": check_team_stat, "player_stat": check_player_stat}


def check_claim(c, gt):
    if out_of_scope(c.get("text", "")):
        return "needs_human", None, "about the tournament or career, not this match"
    fn = CHECKS.get(c.get("type"))
    if fn is None:
        return "needs_human", None, ""
    return fn(c, gt)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Verify extracted claims against ground truth")
    ap.add_argument("--claims", required=True, help="folder created by extract_claims.py")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.claims).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        gt = json.loads((GT_DIR / f"gt_{d['match_id']}.json").read_text(encoding="utf-8"))
        for i, c in enumerate(d["claims"]):
            verdict, err, note = check_claim(c, gt)
            rows.append({"match_id": d["match_id"], "competition": gt["competition"], "condition": d["condition"],
                         "model": d["report_model"], "claim_id": i, "type": c.get("type"), "text": c.get("text"),
                         "verdict": verdict, "error_type": err, "note": note})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{Path(args.claims).name}_claims.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows accents correctly

    # quick summary: accuracy over automatically checkable claims, by condition and match type
    fame = pd.read_csv(GT_DIR / "matches_index.csv")[["match_id", "fame"]]
    auto = df[df.verdict.isin(["correct", "incorrect"])].merge(fame, on="match_id", how="left")
    summary = auto.groupby(["condition", "fame"]).agg(checkable=("verdict", "size"),
                                                      wrong=("verdict", lambda v: (v == "incorrect").sum()))
    summary["accuracy_%"] = (100 * (1 - summary.wrong / summary.checkable)).round(1)
    print(summary.to_string())
    print("\nVerdict counts:\n" + df.verdict.value_counts().to_string())
    print(f"\nSaved {len(df)} claims -> {out}")


if __name__ == "__main__":
    main()