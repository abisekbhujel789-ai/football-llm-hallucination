"""
analyze.py — Phase 4 (M7 Analysis)
Reads the checked claims of every model, computes all numbers for Chapter 4 and draws the charts.

Usage:
  python analyze.py                       # every model that has data/results/<folder>_claims.csv
  python analyze.py --folders gemini_gemini-3.1-flash-lite gemini_gemma-4-26b-a4b-it

Output (data/results/analysis/):
  summary.txt               everything printed on screen
  t1_generation.csv         reports generated / failed per model and condition
  t2_accuracy.csv           accuracy + 95% Wilson CI per model and condition
  t3_accuracy_fame.csv      same, split famous / ordinary
  t4_density.csv            claims, checkable claims and words per report
  t5_report_level.csv       % reports with >=1 wrong claim, % stating the correct score
  t6_verdicts.csv           share of correct / incorrect / needs_human / unverifiable
  t7_error_types.csv        error type counts
  incorrect_claims.csv      every incorrect claim (for the appendix)
  per_report.csv            one row per report (all metrics)
  fig4_1_accuracy.png, fig4_2_density.png, fig4_3_verdicts.png
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

GT_DIR = Path("data/ground_truth")
REPORTS_DIR = Path("data/reports")
RESULTS_DIR = Path("data/results")
OUT = RESULTS_DIR / "analysis"
CONDS = ["A", "B", "C", "D"]
NICE = {"gemini_gemini-3.1-flash-lite": "Flash-Lite", "gemini_gemma-4-26b-a4b-it": "Gemma 4"}
COLORS = ["#1F5FA8", "#D9822B", "#2F7D4F", "#7A4FA0"]

LOG = []


def say(text=""):
    print(text)
    LOG.append(str(text))


def label(folder):
    return NICE.get(folder, folder.split("_", 1)[-1])


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n (in %)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * (centre - half), 1), round(100 * min(1, centre + half), 1))


# ---------------- load ----------------
def find_folders():
    """Every results CSV that has a matching report folder (skips the pilot 'test')."""
    return sorted(p.name[:-len("_claims.csv")] for p in RESULTS_DIR.glob("*_claims.csv")
                  if (REPORTS_DIR / p.name[:-len("_claims.csv")]).is_dir())


def load(folders):
    index = pd.read_csv(GT_DIR / "matches_index.csv")
    claims, reports = [], []
    for f in folders:
        df = pd.read_csv(RESULTS_DIR / f"{f}_claims.csv")
        df["model"] = label(f)
        claims.append(df)
        for m in index.itertuples():
            for c in CONDS:
                p = REPORTS_DIR / f / f"{m.match_id}_{c}.json"
                words = len(json.loads(p.read_text(encoding="utf-8"))["report"].split()) if p.exists() else np.nan
                reports.append({"model": label(f), "match_id": m.match_id, "condition": c, "fame": m.fame,
                                "generated": p.exists(), "words": words})
    claims = pd.concat(claims, ignore_index=True)
    claims = claims.drop(columns=[c for c in ["fame"] if c in claims]).merge(
        index[["match_id", "fame"]], on="match_id", how="left")
    return claims, pd.DataFrame(reports)


# ---------------- per-report metrics ----------------
def per_report(claims, reports):
    c = claims.assign(correct=claims.verdict.eq("correct"), incorrect=claims.verdict.eq("incorrect"),
                      needs_human=claims.verdict.eq("needs_human"), unverifiable=claims.verdict.eq("unverifiable"),
                      score_ok=claims.type.eq("score") & claims.verdict.eq("correct"))
    agg = c.groupby(["model", "match_id", "condition"]).agg(
        claims=("verdict", "size"), correct=("correct", "sum"), incorrect=("incorrect", "sum"),
        needs_human=("needs_human", "sum"), unverifiable=("unverifiable", "sum"),
        score_ok=("score_ok", "any")).reset_index()
    r = reports[reports.generated].merge(agg, on=["model", "match_id", "condition"], how="left")
    for col in ["claims", "correct", "incorrect", "needs_human", "unverifiable"]:
        r[col] = r[col].fillna(0).astype(int)
    r["score_ok"] = r["score_ok"].fillna(False).astype(bool)
    r["checkable"] = r.correct + r.incorrect
    r["hallucinated"] = r.incorrect > 0
    return r


# ---------------- tables ----------------
def accuracy_table(claims, by):
    auto = claims[claims.verdict.isin(["correct", "incorrect"])]
    rows = []
    for key, g in auto.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        n, wrong = len(g), int((g.verdict == "incorrect").sum())
        lo, hi = wilson(n - wrong, n)
        rows.append(dict(zip(by, key), checkable=n, incorrect=wrong,
                         accuracy_pct=round(100 * (n - wrong) / n, 1), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def show(title, df):
    say(f"\n=== {title} ===")
    say(df.to_string(index=False))


# ---------------- statistics ----------------
def safe_wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y)
    if len(d) == 0 or np.all(d == 0):
        return np.nan, 1.0
    res = stats.wilcoxon(x, y)
    return float(res.statistic), float(res.pvalue)


def run_tests(claims, rep):
    say("\n=== Statistical tests (p < 0.05 = significant) ===")
    auto = claims[claims.verdict.isin(["correct", "incorrect"])]
    for model, g in auto.groupby("model"):
        say(f"\n[{model}]")
        table = pd.crosstab(g.condition, g.verdict).reindex(columns=["correct", "incorrect"], fill_value=0)
        if table["incorrect"].sum() > 0 and len(table) > 1:
            chi2, p, dof, exp = stats.chi2_contingency(table)
            note = "  (some expected counts < 5: read with care)" if (exp < 5).any() else ""
            say(f"  Chi-square, accuracy across A-D: chi2={chi2:.2f}, dof={dof}, p={p:.4f}{note}")
        else:
            say("  Chi-square across A-D: not possible (no incorrect claims)")
        if {"A", "D"} <= set(table.index):
            odds, p = stats.fisher_exact(table.loc[["A", "D"]].values)
            say(f"  Fisher exact, A vs D: p={p:.4f}")
        fame = pd.crosstab(g.fame, g.verdict).reindex(columns=["correct", "incorrect"], fill_value=0)
        if len(fame) == 2:
            odds, p = stats.fisher_exact(fame.values)
            say(f"  Fisher exact, famous vs ordinary (all conditions): p={p:.4f}")
        ga = g[g.condition == "A"]
        fame_a = pd.crosstab(ga.fame, ga.verdict).reindex(columns=["correct", "incorrect"], fill_value=0)
        if len(fame_a) == 2:
            odds, p = stats.fisher_exact(fame_a.values)
            say(f"  Fisher exact, famous vs ordinary (condition A only): p={p:.4f}")

        # paired tests on matches that have all four reports for this model
        r = rep[rep.model == model]
        complete = r.groupby("match_id").condition.nunique()
        complete = complete[complete == 4].index
        rc = r[r.match_id.isin(complete)].pivot(index="match_id", columns="condition")
        say(f"  Paired tests on {len(complete)} complete matches (Wilcoxon signed-rank):")
        for a, b in [("A", "B"), ("A", "C"), ("A", "D"), ("C", "D")]:
            s1, p1 = safe_wilcoxon(rc["checkable"][a], rc["checkable"][b])
            s2, p2 = safe_wilcoxon(rc["incorrect"][a], rc["incorrect"][b])
            say(f"    {a} vs {b}: checkable claims p={p1:.4f} | incorrect claims p={p2:.4f}")

    if auto.model.nunique() == 2:
        t = pd.crosstab(auto.model, auto.verdict).reindex(columns=["correct", "incorrect"], fill_value=0)
        odds, p = stats.fisher_exact(t.values)
        say(f"\n[Model comparison] Fisher exact, {t.index[0]} vs {t.index[1]} (all conditions): p={p:.4f}")


# ---------------- charts ----------------
def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_accuracy(t2, models):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        d = t2[t2.model == m].set_index("condition").reindex(CONDS)
        x = np.arange(len(CONDS)) + (i - (len(models) - 1) / 2) * width
        ax.errorbar(x, d.accuracy_pct, yerr=[d.accuracy_pct - d.ci_low, d.ci_high - d.accuracy_pct],
                    fmt="o", capsize=5, markersize=8, color=COLORS[i], label=m)
        for xi, v in zip(x, d.accuracy_pct):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(10, -4), fontsize=8)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(["A: no data", "B: raw events", "C: summary", "D: summary + rule"])
    ax.set_ylabel("Accuracy of checkable claims (%)")
    ax.set_ylim(max(0, t2.ci_low.min() - 3), 100.8)
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_1_accuracy.png", dpi=300)
    plt.close(fig)


def fig_density(rep, models):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    width = 0.8 / len(models)
    for ax, fame in zip(axes, ["famous", "ordinary"]):
        for i, m in enumerate(models):
            d = rep[(rep.model == m) & (rep.fame == fame)].groupby("condition").checkable.mean().reindex(CONDS)
            x = np.arange(len(CONDS)) + (i - (len(models) - 1) / 2) * width
            ax.bar(x, d.values, width=width * 0.95, color=COLORS[i], label=m)
            for xi, v in zip(x, d.values):
                if not np.isnan(v):
                    ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_title(f"{fame.capitalize()} matches", fontsize=11)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels(CONDS)
        ax.set_xlabel("Condition")
        style_axes(ax)
    axes[0].set_ylabel("Checkable claims per report (mean)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_2_density.png", dpi=300)
    plt.close(fig)


def fig_verdicts(t6, models):
    order = ["correct", "needs_human", "unverifiable", "incorrect"]
    colors = {"correct": "#2F7D4F", "needs_human": "#B9C4BD", "unverifiable": "#7A8A80", "incorrect": "#C0392B"}
    labels = [f"{m}\n{c}" for m in models for c in CONDS]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.5))
    bottom = np.zeros(len(labels))
    for v in order:
        vals = [t6[(t6.model == m) & (t6.condition == c)][f"{v}_pct"].sum() for m in models for c in CONDS]
        ax.bar(range(len(labels)), vals, bottom=bottom, color=colors[v], label=v.replace("_", " "), width=0.7)
        bottom += np.array(vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Share of all claims (%)")
    ax.set_ylim(0, 100)
    style_axes(ax)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / "fig4_3_verdicts.png", dpi=300)
    plt.close(fig)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Analyse checked claims for Chapter 4")
    ap.add_argument("--folders", nargs="*", help="report/claim folder names (default: all found)")
    args = ap.parse_args()
    folders = args.folders or find_folders()
    if not folders:
        raise SystemExit("No data/results/<folder>_claims.csv found. Run check_claims.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    claims, reports = load(folders)
    models = [label(f) for f in folders]
    rep = per_report(claims, reports)
    say(f"Models: {', '.join(models)}")

    # T1 generation
    t1 = reports.groupby(["model", "condition"]).agg(expected=("generated", "size"),
                                                   generated=("generated", "sum")).reset_index()
    t1["failed"] = t1.expected - t1.generated
    show("T1 Reports generated", t1)
    failed = reports[~reports.generated].merge(pd.read_csv(GT_DIR / "matches_index.csv")[["match_id", "home", "away"]],
                                                on="match_id")
    if len(failed):
        say("\nMissing reports:")
        for r in failed.itertuples():
            say(f"  {r.model}: {r.home} vs {r.away} ({r.condition}, {r.fame})")

    # T2/T3 accuracy
    t2 = accuracy_table(claims, ["model", "condition"])
    t3 = accuracy_table(claims, ["model", "condition", "fame"])
    show("T2 Accuracy of checkable claims (95% Wilson CI)", t2)
    show("T3 Accuracy by match fame", t3)
    show("Overall accuracy per model", accuracy_table(claims, ["model"]))

    # T4 density
    t4 = rep.groupby(["model", "condition", "fame"]).agg(
        reports=("match_id", "size"), words=("words", "mean"), claims=("claims", "mean"),
        checkable=("checkable", "mean"), needs_human=("needs_human", "mean")).round(1).reset_index()
    show("T4 Claim density (mean per report)", t4)

    # T5 report level
    t5 = rep.groupby(["model", "condition"]).agg(
        reports=("match_id", "size"), with_error=("hallucinated", "sum"),
        with_error_pct=("hallucinated", lambda s: round(100 * s.mean(), 1)),
        correct_score_pct=("score_ok", lambda s: round(100 * s.mean(), 1))).reset_index()
    show("T5 Report level: reports with >=1 wrong claim, reports stating the correct score", t5)

    # T6 verdict shares
    t6 = claims.groupby(["model", "condition"]).verdict.value_counts(normalize=True).mul(100).round(1) \
        .unstack(fill_value=0).reindex(columns=["correct", "incorrect", "needs_human", "unverifiable"], fill_value=0)
    t6.columns = [f"{c}_pct" for c in t6.columns]
    t6 = t6.reset_index()
    show("T6 Verdict shares (% of all claims)", t6)

    # T7 error types + list
    wrong = claims[claims.verdict == "incorrect"]
    t7 = wrong.groupby(["model", "condition", "error_type"]).size().rename("count").reset_index()
    show("T7 Error types", t7 if len(t7) else pd.DataFrame({"info": ["no incorrect claims"]}))
    wrong_list = wrong.merge(pd.read_csv(GT_DIR / "matches_index.csv")[["match_id", "home", "away"]], on="match_id")
    wrong_list = wrong_list[["model", "condition", "fame", "home", "away", "type", "text", "error_type", "note"]]

    run_tests(claims, rep)

    # save
    for name, df in [("t1_generation", t1), ("t2_accuracy", t2), ("t3_accuracy_fame", t3), ("t4_density", t4),
                     ("t5_report_level", t5), ("t6_verdicts", t6), ("t7_error_types", t7),
                     ("incorrect_claims", wrong_list), ("per_report", rep)]:
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    fig_accuracy(t2, models)
    fig_density(rep, models)
    fig_verdicts(t6, models)
    (OUT / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nSaved tables, charts and summary.txt -> {OUT}/")


if __name__ == "__main__":
    main()