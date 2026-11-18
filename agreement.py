"""
agreement.py — Phase 5b (M8 Human annotation, analysis)
Reads the filled annotation sheets and reports:
  1. labelling consistency, A vs B (Cohen's kappa). With one annotator, B is the same person
     re-labelling the shared rows a week later (intra-annotator agreement).
  2. checker vs human agreement on the auto-checked sample (is the rule-based checker trustworthy?)
  3. human verdicts for the 'needs_human' claims, per model and condition
  4. final accuracy per model and condition = checker verdicts + human labels combined

Usage:
  python agreement.py
Input : data/annotation/key.csv, annotation_A.xlsx, annotation_B.xlsx (optional)
Output: data/annotation/results/  (csv tables, disagreements.csv, summary.txt)
"""
import math
from pathlib import Path

import pandas as pd

ANN = Path("data/annotation")
OUT = ANN / "results"
GT_DIR = Path("data/ground_truth")
LABELS = ["correct", "incorrect", "not_checkable"]
NICE = {"gemini_gemini-3.1-flash-lite": "Flash-Lite", "gemini_gemma-4-26b-a4b-it": "Gemma 4"}
LOG = []


def say(text=""):
    print(text)
    LOG.append(str(text))


def show(title, df):
    say(f"\n=== {title} ===")
    say(df.to_string(index=False) if len(df) else "(nothing yet)")


def cohen_kappa(a, b):
    """Cohen's kappa for two lists of labels."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def landis_koch(k):
    if math.isnan(k):
        return "n/a"
    for limit, word in [(0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                        (0.80, "substantial"), (1.01, "almost perfect")]:
        if k <= limit:
            return word
    return "almost perfect"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * (c - h), 1), round(100 * min(1, c + h), 1))


def read_labels(path, who):
    df = pd.read_excel(path, sheet_name="claims")
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})
    bad = df[(df.label != "") & ~df.label.isin(LABELS)]
    if len(bad):
        say(f"[warn] {who}: {len(bad)} rows have an unknown label (allowed: {', '.join(LABELS)}): "
            f"row_id {bad.row_id.tolist()[:10]}")
    df.loc[~df.label.isin(LABELS), "label"] = ""
    return df[["row_id", "label", "comment"]].rename(columns={"label": f"label_{who}", "comment": f"comment_{who}"})


def main():
    if not (ANN / "key.csv").exists():
        raise SystemExit("data/annotation/key.csv not found. Run annotate_export.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    key = pd.read_csv(ANN / "key.csv")
    key["model"] = key.folder.map(lambda f: NICE.get(f, f))
    df = key.merge(read_labels(ANN / "annotation_A.xlsx", "A"), on="row_id", how="left")
    has_b = (ANN / "annotation_B.xlsx").exists()
    if has_b:
        df = df.merge(read_labels(ANN / "annotation_B.xlsx", "B"), on="row_id", how="left")
    df["label_A"] = df["label_A"].fillna("")

    done = (df.label_A != "").sum()
    say(f"Annotator A progress: {done}/{len(df)} rows labelled")
    if has_b:
        b_rows = df[df.in_B]
        say(f"Annotator B progress: {(b_rows.label_B.fillna('') != '').sum()}/{len(b_rows)} rows labelled")

    # 1. inter-annotator agreement
    if has_b:
        both = df[(df.label_A != "") & (df.label_B.fillna("") != "")]
        k = cohen_kappa(both.label_A, both.label_B)
        agree = (both.label_A == both.label_B).mean() * 100 if len(both) else float("nan")
        say(f"\n=== 1. Labelling consistency: A vs B (intra-annotator if B = same person later) ===")
        say(f"  rows labelled by both: {len(both)}")
        if len(both) == 0:
            say("  (file B not labelled yet - re-label it about a week after file A)")
        say(f"  raw agreement: {agree:.1f}%   Cohen's kappa: {k:.3f} ({landis_koch(k)})")
        if len(both):
            say(pd.crosstab(both.label_A, both.label_B, rownames=["A"], colnames=["B"]).to_string())
        dis = both[both.label_A != both.label_B][["row_id", "model", "condition", "text", "label_A", "label_B",
                                                   "comment_A", "comment_B"]]
        dis.to_csv(OUT / "disagreements.csv", index=False, encoding="utf-8-sig")
        say(f"  {len(dis)} disagreements -> {OUT}/disagreements.csv (discuss them, then fix annotation_A.xlsx)")

    # 2. checker vs human on the auto-checked sample
    s = df[(df.task == "auto_sample") & (df.label_A != "")]
    say("\n=== 2. Rule-based checker vs human (auto-checked sample) ===")
    if len(s):
        say(pd.crosstab(s.verdict, s.label_A, rownames=["checker"], colnames=["human"]).to_string())
        both = s[s.label_A.isin(["correct", "incorrect"])]
        k = cohen_kappa(both.verdict, both.label_A)
        agree = (both.verdict == both.label_A).mean() * 100 if len(both) else float("nan")
        say(f"  on {len(both)} claims the human could judge: agreement {agree:.1f}%, "
            f"kappa {k:.3f} ({landis_koch(k)})")
        fa = ((both.verdict == "incorrect") & (both.label_A == "correct")).sum()
        miss = ((both.verdict == "correct") & (both.label_A == "incorrect")).sum()
        say(f"  checker false alarms (said incorrect, human: correct): {fa}")
        say(f"  checker misses       (said correct, human: incorrect): {miss}")
        s[s.verdict != s.label_A][["row_id", "model", "condition", "text", "verdict", "label_A", "comment_A"]] \
            .to_csv(OUT / "checker_vs_human_differences.csv", index=False, encoding="utf-8-sig")
    else:
        say("  (no labelled sample rows yet)")

    # 3. human verdicts on needs_human claims
    h = df[(df.task == "needs_human") & (df.label_A != "")]
    t3 = (h.groupby(["model", "condition"]).label_A.value_counts().unstack(fill_value=0)
          .reindex(columns=LABELS, fill_value=0).reset_index())
    if len(t3):
        t3["accuracy_pct"] = (100 * t3.correct / (t3.correct + t3.incorrect).where(lambda x: x > 0)).round(1)
    show("3. Human labels for 'needs_human' claims", t3)

    # 4. final accuracy: checker verdicts for auto claims + human labels for needs_human claims
    all_claims = []
    for folder in key.folder.unique():
        c = pd.read_csv(Path("data/results") / f"{folder}_claims.csv")
        c["model"] = NICE.get(folder, folder)
        c["folder"] = folder
        all_claims.append(c)
    allc = pd.concat(all_claims, ignore_index=True)
    labels = df[df.task == "needs_human"][["folder", "match_id", "condition", "claim_id", "label_A"]]
    allc = allc.merge(labels, on=["folder", "match_id", "condition", "claim_id"], how="left")
    allc["final"] = allc.verdict
    nh = allc.verdict == "needs_human"
    allc.loc[nh, "final"] = allc.loc[nh, "label_A"].fillna("").replace({"": "needs_human (unlabelled)"})
    rows = []
    for (m, cond), g in allc.groupby(["model", "condition"]):
        n_ok, n_bad = (g.final == "correct").sum(), (g.final == "incorrect").sum()
        n = n_ok + n_bad
        lo, hi = wilson(n_ok, n)
        rows.append({"model": m, "condition": cond, "checkable": n, "incorrect": n_bad,
                     "accuracy_pct": round(100 * n_ok / n, 1) if n else float("nan"), "ci_low": lo, "ci_high": hi,
                     "still_unlabelled": (g.final == "needs_human (unlabelled)").sum()})
    t4 = pd.DataFrame(rows)
    show("4. Final accuracy (checker + human labels)", t4)

    t3.to_csv(OUT / "needs_human_labels.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(OUT / "final_accuracy.csv", index=False, encoding="utf-8-sig")
    allc.to_csv(OUT / "all_claims_final.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nSaved -> {OUT}/")


if __name__ == "__main__":
    main()