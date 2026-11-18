"""
annotate_export.py — Phase 5a (M8 Human annotation, export)
Builds blind Excel sheets for human labelling.

Rows to label:
  1. every claim the checker marked 'needs_human' (tactical / tournament claims)
  2. a random sample of automatically checked claims (correct + incorrect), to measure checker-human agreement
Annotator A labels every row; annotator B labels a random overlap subset (for Cohen's kappa).
The sheets hide the model, condition and checker verdict, so the labelling is blind.

Usage:
  python annotate_export.py                      # 20% auto sample, 20% overlap for annotator B
  python annotate_export.py --auto-frac 0.1      # smaller auto sample if time is short
Output (data/annotation/):
  annotation_A.xlsx   all rows   -> you
  annotation_B.xlsx   overlap    -> second annotator
  key.csv             hidden answer key (do NOT give to annotators)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

GT_DIR = Path("data/ground_truth")
CLAIMS_DIR = Path("data/claims")
REPORTS_DIR = Path("data/reports")
RESULTS_DIR = Path("data/results")
OUT = Path("data/annotation")
SEED = 42
LABELS = ["correct", "incorrect", "not_checkable"]

INSTRUCTIONS = [
    "HOW TO LABEL",
    "",
    "Each row is one claim taken from an AI-written football match report.",
    "Use the 'matches' sheet (real match data) and, if needed, the 'reports' sheet (the full report) as evidence.",
    "Choose ONE label in the 'label' column (drop-down):",
    "",
    "  correct        the claim agrees with the match data or with well-known facts about the match",
    "  incorrect      the claim contradicts the match data (wrong number, player, minute, result, event)",
    "  not_checkable  opinion, tactics or tournament context that the data cannot confirm or deny",
    "",
    "Rules:",
    "  - Judge the claim only, not the writing style.",
    "  - Statistics in the data are the truth (e.g. 'possession' is approximated by pass share; allow +/-3 points).",
    "  - Goal minutes may differ by 1-2 minutes; '45+2' counts as the 45th-47th minute.",
    "  - Tactical claims ('Spain dominated possession', 'England sat deep') are 'correct' if the numbers clearly",
    "    support them, 'incorrect' if the numbers clearly contradict them, otherwise 'not_checkable'.",
    "  - Use 'comment' for anything unsure. Do not discuss labels with the other annotator until both are done.",
]


def find_folders():
    return sorted(p.name[:-len("_claims.csv")] for p in RESULTS_DIR.glob("*_claims.csv")
                  if (REPORTS_DIR / p.name[:-len("_claims.csv")]).is_dir())


def load_claims(folders):
    frames = []
    for f in folders:
        df = pd.read_csv(RESULTS_DIR / f"{f}_claims.csv")
        df["folder"] = f
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_sheet(path, rows, reports, matches):
    wb = Workbook()
    ws = wb.active
    ws.title = "claims"
    head = ["row_id", "report", "match", "claim", "label", "comment"]
    ws.append(head)
    for r in rows.itertuples():
        ws.append([r.row_id, r.report_code, r.match, r.text, "", ""])
    for col, width in zip("ABCDEF", [9, 9, 34, 80, 16, 40]):
        ws.column_dimensions[col].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDE7DF")
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=4):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(rows) + 1}")
    ws.freeze_panes = "A2"

    wr = wb.create_sheet("reports")
    wr.append(["report", "match", "text"])
    for r in reports.itertuples():
        wr.append([r.report_code, r.match, r.text])
    wr.column_dimensions["A"].width, wr.column_dimensions["B"].width, wr.column_dimensions["C"].width = 9, 34, 120
    for row in wr.iter_rows(min_row=2, min_col=3, max_col=3):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")

    wm = wb.create_sheet("matches")
    wm.append(["match", "real match data (StatsBomb)"])
    for m in matches.itertuples():
        wm.append([m.match, m.summary])
    wm.column_dimensions["A"].width, wm.column_dimensions["B"].width = 34, 120
    for row in wm.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")

    wi = wb.create_sheet("instructions", 0)
    for line in INSTRUCTIONS:
        wi.append([line])
    wi["A1"].font = Font(bold=True, size=13)
    wi.column_dimensions["A"].width = 120
    wb.active = 1
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="Export blind annotation sheets")
    ap.add_argument("--folders", nargs="*", help="model folders (default: all found)")
    ap.add_argument("--auto-frac", type=float, default=0.20, help="share of auto-checked claims to re-label")
    ap.add_argument("--overlap", type=float, default=0.20, help="share of rows also given to annotator B")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "key.csv").exists():
        raise SystemExit(f"{OUT}/key.csv already exists. Delete the data/annotation folder first if you "
                         "really want new sheets (this would throw away any labels already entered).")

    folders = args.folders or find_folders()
    claims = load_claims(folders)
    index = pd.read_csv(GT_DIR / "matches_index.csv")
    index["match"] = index.home + " vs " + index.away + " (" + index.competition + ")"
    claims = claims.merge(index[["match_id", "match"]], on="match_id", how="left")

    human = claims[claims.verdict == "needs_human"].assign(task="needs_human")
    auto = claims[claims.verdict.isin(["correct", "incorrect"])]
    sample = (auto.groupby(["folder", "condition"]).sample(frac=args.auto_frac, random_state=SEED)
              .assign(task="auto_sample"))
    rows = pd.concat([human, sample], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # anonymous report codes, so annotators cannot tell model or condition
    keys = rows[["folder", "match_id", "condition"]].drop_duplicates().sample(frac=1, random_state=SEED)
    keys["report_code"] = [f"R{i:03d}" for i in range(1, len(keys) + 1)]
    rows = rows.merge(keys, on=["folder", "match_id", "condition"], how="left")
    rows["row_id"] = range(1, len(rows) + 1)

    report_rows = []
    for k in keys.itertuples():
        d = json.loads((CLAIMS_DIR / k.folder / f"{k.match_id}_{k.condition}.json").read_text(encoding="utf-8"))
        report_rows.append({"report_code": k.report_code, "match_id": k.match_id, "text": d["report"]})
    reports = pd.DataFrame(report_rows).merge(index[["match_id", "match"]], on="match_id").sort_values("report_code")

    used = index[index.match_id.isin(rows.match_id)]
    matches = pd.DataFrame({"match": used.match,
                            "summary": [(GT_DIR / f"summary_{m}.txt").read_text(encoding="utf-8") for m in used.match_id]})

    overlap_ids = rows.sample(frac=args.overlap, random_state=SEED + 1).row_id
    rows["in_B"] = rows.row_id.isin(overlap_ids)

    write_sheet(OUT / "annotation_A.xlsx", rows, reports, matches)
    rows_b = rows[rows.in_B]
    write_sheet(OUT / "annotation_B.xlsx", rows_b, reports[reports.report_code.isin(rows_b.report_code)],
                matches[matches.match.isin(rows_b.match)])
    rows[["row_id", "report_code", "folder", "match_id", "condition", "claim_id", "type", "text",
          "verdict", "error_type", "task", "in_B"]].to_csv(OUT / "key.csv", index=False, encoding="utf-8-sig")

    print(f"Rows for annotator A: {len(rows)}  (needs_human {len(human)}, auto sample {len(sample)})")
    print(f"Rows for annotator B: {len(rows_b)}")
    print(rows.groupby(["folder", "task"]).size().to_string())
    print(f"\nSaved -> {OUT}/annotation_A.xlsx, annotation_B.xlsx, key.csv (keep key.csv to yourself)")


if __name__ == "__main__":
    main()