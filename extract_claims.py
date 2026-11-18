"""
extract_claims.py — Phase 3a (M5 Claim extractor)
Uses an LLM to split each generated report into atomic claims with a fixed JSON structure,
so that check_claims.py can verify them automatically.

Usage:
  python extract_claims.py --reports data/reports/gemini_gemini-3.1-flash-lite --provider gemini --model gemini-3.1-flash-lite --sleep 10
Output: data/claims/<same folder name>/<match>_<cond>.json
"""
import argparse
import json
import re
import time
from pathlib import Path

from generate import PROVIDERS, call_with_retry

CLAIMS_DIR = Path("data/claims")

EXTRACT_SYSTEM = "You are a careful annotator who extracts factual claims from football match reports."

EXTRACT_PROMPT = """Split the match report below into ATOMIC claims: each claim states exactly one checkable fact.
Return ONLY a JSON list (no other text). Each item has these keys:

  "text"    : the claim in a short sentence, e.g. "Messi scored in the 23rd minute"
  "type"    : one of
              "result"      - who won / draw / how (e.g. "Argentina won on penalties")
              "score"       - a final score or shootout score
              "goal"        - a specific goal by a player (with minute if given)
              "team_stat"   - a team statistic (shots, possession, corners, cards...)
              "player_stat" - a player statistic (goals, shots, assists, xG, key passes)
              "tactical"    - how the match was played (pressing, dominance, momentum...)
              "other"       - anything else (history, records, context, quotes)
  "team"    : team name, "both" for a combined match total (e.g. "30 shots in the match"), or null
  "player"  : player name or null
  "minute"  : integer minute or null
  "stat"    : for team_stat/player_stat use one of: shots, shots_on_target, xg, passes,
              pass_accuracy, possession, corners, fouls, yellow_cards, red_cards,
              goals, assists, key_passes; otherwise null
  "value"   : the number stated, or null
  "winner"  : for "result": the winning team, or "draw"; otherwise null
  "method"  : for "result": "normal_time", "extra_time", "penalties" or null
  "home_goals"/"away_goals" : for "score" only, else null
  "shootout": for "score" only: true if the score is a penalty shootout score

Rules:
- Split compound sentences: "Mbappe scored twice in two minutes" -> separate goal claims.
- A hat-trick is a player_stat claim with stat "goals" and value 3.
- Use team_stat/player_stat ONLY for whole-match counts. A claim about part of the match
  ("no shots in the first half", "until the 70th minute") is "tactical".
- Single events that are not a count - hitting the post, a missed chance, a save, a substitution,
  an injury, a card shown to one player - are "other".
- Do not invent claims that are not in the report. Do not judge whether claims are true.

MATCH: {home} (home) vs {away} (away)

REPORT:
{report}
"""


def parse_json_list(text):
    """Pull the first JSON list out of a model reply (models sometimes add ``` fences)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON list in reply")
    claims = json.loads(text[start:end + 1])
    if not isinstance(claims, list):
        raise ValueError("reply is not a list")
    return [c for c in claims if isinstance(c, dict) and c.get("text")]


def main():
    ap = argparse.ArgumentParser(description="Extract atomic claims from generated reports")
    ap.add_argument("--reports", required=True, help="folder of report JSON files from generate.py")
    ap.add_argument("--provider", required=True, choices=PROVIDERS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    rep_dir = Path(args.reports)
    out_dir = CLAIMS_DIR / rep_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = PROVIDERS[args.provider]
    done = skipped = failed = 0

    for f in sorted(rep_dir.glob("*.json")):
        out = out_dir / f.name
        if out.exists():
            skipped += 1
            continue
        rep = json.loads(f.read_text(encoding="utf-8"))
        prompt = EXTRACT_PROMPT.format(home=rep["home"], away=rep["away"], report=rep["report"])
        try:
            text, _ = call_with_retry(fn, args.model, EXTRACT_SYSTEM, prompt)
            claims = parse_json_list(text)
        except Exception as e:
            failed += 1
            print(f"  [fail] {f.name}: {str(e)[:150]}")
            if "429" in str(e):
                print("\nFree quota used up. Run the same command again later - finished files are kept.")
                return
            continue
        out.write_text(json.dumps({
            "match_id": rep["match_id"], "condition": rep["condition"], "report_model": rep["model"],
            "extractor_provider": args.provider, "extractor_model": args.model,
            "report": rep["report"], "claims": claims,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        print(f"  [ok] {f.name}: {len(claims)} claims")
        time.sleep(args.sleep)

    print(f"\nDone: {done} new, {skipped} existed, {failed} failed -> {out_dir}/")


if __name__ == "__main__":
    main()