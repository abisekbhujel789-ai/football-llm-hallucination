"""
ground_truth.py — Phase 1 (M1 Data loader + M2 Ground truth builder)
Loads StatsBomb open data, computes real match stats (the answer key),
and writes a compact text summary that LLMs receive in condition C.

Usage:
    python ground_truth.py            # test: only the final of each tournament
    python ground_truth.py all        # every match of every tournament
Data: StatsBomb Open Data (credit StatsBomb in your thesis).
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")  # statsbombpy prints a credentials warning for free data

OUT = Path("data/ground_truth")
OUT.mkdir(parents=True, exist_ok=True)

TOURNAMENTS = [
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2024"),
    ("Copa America", "2024"),
]

ON_TARGET = {"Goal", "Saved", "Saved To Post"}
CARD_COLS = ["foul_committed_card", "bad_behaviour_card"]


# ---------------- helpers ----------------
def col(df, name):
    """Return a column, or an empty column if this match doesn't have it."""
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def get_matches(competition_name, season_name):
    comps = sb.competitions()
    row = comps[(comps.competition_name == competition_name) & (comps.season_name == season_name)]
    if row.empty:
        raise ValueError(f"Not found: {competition_name} {season_name}")
    r = row.iloc[0]
    m = sb.matches(competition_id=int(r.competition_id), season_id=int(r.season_id))
    m["competition"] = f"{competition_name} {season_name}"
    return m.sort_values("match_date")


# ---------------- ground truth ----------------
def goal_timeline(ev, teams):
    """All goals incl. own goals, excluding penalty shootouts (period 5)."""
    ev = ev[ev.period < 5]
    goals = []
    shots = ev[ev.type == "Shot"]
    for r in shots[col(shots, "shot_outcome") == "Goal"].itertuples():
        goals.append({"minute": int(r.minute) + 1, "team": r.team, "player": r.player,
                      "type": "penalty" if getattr(r, "shot_type", None) == "Penalty" else "goal"})
    # 'Own Goal Against' = the team that conceded; player = who scored the own goal
    for r in ev[ev.type == "Own Goal Against"].itertuples():
        other = [t for t in teams if t != r.team][0]
        goals.append({"minute": int(r.minute) + 1, "team": other, "player": r.player, "type": "own goal"})
    return sorted(goals, key=lambda g: g["minute"])


def team_stats(ev, team, total_passes):
    t = ev[(ev.team == team) & (ev.period < 5)]
    shots = t[t.type == "Shot"]
    passes = t[t.type == "Pass"]
    completed = passes[col(passes, "pass_outcome").isna()]
    cards = pd.concat([col(t, c) for c in CARD_COLS])
    return {
        "shots": len(shots),
        "shots_on_target": int(col(shots, "shot_outcome").isin(ON_TARGET).sum()),
        "xg": round(float(pd.to_numeric(col(shots, "shot_statsbomb_xg")).fillna(0).sum()), 2),
        "passes": len(passes),
        "pass_accuracy_pct": round(100 * len(completed) / max(len(passes), 1), 1),
        "pass_share_pct": round(100 * len(passes) / max(total_passes, 1), 1),  # possession proxy
        "corners": int((col(passes, "pass_type") == "Corner").sum()),
        "fouls": int((t.type == "Foul Committed").sum()),
        "yellow_cards": int(cards.isin(["Yellow Card", "Second Yellow"]).sum()),
        "red_cards": int(cards.isin(["Red Card", "Second Yellow"]).sum()),
    }


def player_stats(ev):
    ev = ev[(ev.period < 5) & ev.player.notna()]
    rows = []
    for (team, player), g in ev.groupby(["team", "player"]):
        shots = g[g.type == "Shot"]
        passes = g[g.type == "Pass"]
        goal_assist = col(passes, "pass_goal_assist") == True  # noqa: E712
        shot_assist = col(passes, "pass_shot_assist") == True  # noqa: E712
        rows.append({
            "team": team,
            "player": player,
            "goals": int((col(shots, "shot_outcome") == "Goal").sum()),
            "shots": len(shots),
            "xg": round(float(pd.to_numeric(col(shots, "shot_statsbomb_xg")).fillna(0).sum()), 2),
            "assists": int(goal_assist.sum()),
            "key_passes": int((goal_assist | shot_assist).sum()),
            "passes": len(passes),
            "passes_completed": int(col(passes, "pass_outcome").isna().sum()),
        })
    return pd.DataFrame(rows)


def shootout(ev, teams):
    """Penalty shootout result (period 5), or None."""
    so = ev[(ev.period == 5) & (ev.type == "Shot")]
    if so.empty:
        return None
    scored = so[col(so, "shot_outcome") == "Goal"].groupby("team").size()
    return {t: int(scored.get(t, 0)) for t in teams}


def name_map(match_id):
    """Full StatsBomb name -> common name (nickname if StatsBomb has one)."""
    names = {}
    for _, df in sb.lineups(match_id=match_id).items():
        for r in df.itertuples():
            nick = r.player_nickname if isinstance(r.player_nickname, str) and r.player_nickname else r.player_name
            names[r.player_name] = nick
    return names


def aliases(full, common):
    """Names a report might use for one player: full, common, and surname."""
    out = {full, common, common.split()[-1]}
    return sorted(out)


def build_ground_truth(match):
    ev = sb.events(match_id=int(match.match_id))
    names = name_map(int(match.match_id))
    ev["player"] = ev["player"].map(lambda p: names.get(p, p) if isinstance(p, str) else p)
    home, away = match.home_team, match.away_team
    total_passes = int((ev[ev.period < 5].type == "Pass").sum())
    goals = goal_timeline(ev, [home, away])

    gt = {
        "match_id": int(match.match_id),
        "competition": match.competition,
        "date": str(match.match_date),
        "stage": str(match.competition_stage),
        "home_team": home,
        "away_team": away,
        "score": {home: sum(g["team"] == home for g in goals), away: sum(g["team"] == away for g in goals)},
        "went_to_extra_time": bool((ev.period >= 3).any()),
        "penalty_shootout": shootout(ev, [home, away]),
        "goals": goals,
        "team_stats": {home: team_stats(ev, home, total_passes), away: team_stats(ev, away, total_passes)},
        "players": player_stats(ev).to_dict(orient="records"),
        "player_aliases": {common: aliases(full, common) for full, common in names.items()},
        "source": "StatsBomb Open Data",
    }

    official = {home: int(match.home_score), away: int(match.away_score)}
    if gt["score"] != official:
        print(f"  [warn] score mismatch {home} vs {away}: computed {gt['score']} vs official {official}")
    return gt, ev


# ---------------- summary for the LLM (condition C) ----------------
def make_summary(gt):
    h, a = gt["home_team"], gt["away_team"]
    lines = [
        f"Match: {h} vs {a} ({gt['competition']}, {gt['stage']}, {gt['date']})",
        f"Final score (after extra time if played, excluding shootout): {h} {gt['score'][h]} - {gt['score'][a]} {a}",
    ]
    if gt["penalty_shootout"]:
        so = gt["penalty_shootout"]
        lines.append(f"Penalty shootout: {h} {so[h]} - {so[a]} {a}")
    lines += ["", "Goals:"]
    for g in gt["goals"]:
        tag = {"own goal": " (own goal)", "penalty": " (penalty)"}.get(g["type"], "")
        lines.append(f"  {g['minute']}' {g['player']} ({g['team']}){tag}")
    if not gt["goals"]:
        lines.append("  None")

    lines += ["", "Team stats:"]
    for team, s in gt["team_stats"].items():
        lines.append(
            f"  {team}: shots {s['shots']} (on target {s['shots_on_target']}), xG {s['xg']}, "
            f"passes {s['passes']} ({s['pass_accuracy_pct']}% accurate), pass share {s['pass_share_pct']}%, "
            f"corners {s['corners']}, fouls {s['fouls']}, yellow cards {s['yellow_cards']}, red cards {s['red_cards']}"
        )

    players = pd.DataFrame(gt["players"])
    players["impact"] = players["xg"] + players["goals"] + 0.1 * players["key_passes"]
    lines += ["", "Key players:"]
    for p in players.sort_values("impact", ascending=False).head(6).itertuples():
        lines.append(f"  {p.player} ({p.team}): goals {p.goals}, assists {p.assists}, shots {p.shots}, "
                     f"xG {p.xg}, key passes {p.key_passes}")
    return "\n".join(lines)


def save_match(match):
    gt, ev = build_ground_truth(match)
    mid = gt["match_id"]
    (OUT / f"gt_{mid}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / f"summary_{mid}.txt").write_text(make_summary(gt), encoding="utf-8")
    # raw events kept for condition B (key columns only, to save space)
    keep = [c for c in ["period", "minute", "second", "team", "player", "type", "shot_outcome", "shot_type",
                        "pass_type", "pass_goal_assist", "pass_shot_assist", "foul_committed_card",
                        "bad_behaviour_card", "substitution_replacement"] if c in ev.columns]
    ev[keep].to_csv(OUT / f"events_{mid}.csv", index=False)
    return gt


# ---------------- match selection ----------------
FAMOUS_STAGES = {"Final", "Semi-finals"}
GROUP_PER_TOURNAMENT = 9      # lesser-known group games per tournament
SEED = 42                     # fixed seed -> the same matches are picked every time (reproducible)


def select_matches(matches, mode):
    """test = the final only; sample = final + semi-finals + 9 random group games; all = everything."""
    if mode == "all":
        return matches
    if mode == "sample":
        famous = matches[matches.competition_stage.isin(FAMOUS_STAGES)]
        group = matches[matches.competition_stage == "Group Stage"]
        group = group.sample(n=min(GROUP_PER_TOURNAMENT, len(group)), random_state=SEED)
        return pd.concat([famous, group]).sort_values("match_date")
    return matches.tail(1)


# ---------------- run ----------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode not in ("test", "sample", "all"):
        sys.exit("usage: python ground_truth.py [test|sample|all]")
    index = []
    for comp, season in TOURNAMENTS:
        matches = get_matches(comp, season)
        chosen = select_matches(matches, mode)
        print(f"{comp} {season}: {len(chosen)} of {len(matches)} matches")
        for _, m in chosen.iterrows():
            row = {"match_id": int(m.match_id), "competition": m.competition, "date": str(m.match_date),
                   "stage": str(m.competition_stage), "home": m.home_team, "away": m.away_team,
                   "score": f"{int(m.home_score)}-{int(m.away_score)}",
                   "fame": "famous" if m.competition_stage in FAMOUS_STAGES else "ordinary"}
            if (OUT / f"gt_{row['match_id']}.json").exists():
                index.append(row)            # already built earlier -> skip the download
                print(f"  kept  {m.home_team} vs {m.away_team}")
                continue
            try:
                save_match(m)
                index.append(row)
                print(f"  saved {m.home_team} vs {m.away_team} ({m.competition_stage})")
            except Exception as e:  # keep going if one match fails
                print(f"  [error] {m.home_team} vs {m.away_team}: {e}")
    pd.DataFrame(index).to_csv(OUT / "matches_index.csv", index=False)
    print(f"\nDone. {len(index)} matches in matches_index.csv")
    print(pd.DataFrame(index).groupby(["competition", "fame"]).size().to_string())