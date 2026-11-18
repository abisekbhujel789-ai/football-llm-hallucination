"""
app.py — demo app (for supervisor meetings and the thesis defence)
Pick a model, a match and an input condition, and see the generated report with every claim checked.

Run:
  pip install streamlit
  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

GT_DIR = Path("data/ground_truth")
REPORTS_DIR = Path("data/reports")
RESULTS_DIR = Path("data/results")
ANALYSIS_DIR = RESULTS_DIR / "analysis"
NICE = {"gemini_gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemini_gemma-4-26b-a4b-it": "Gemma 4 (26B A4B)"}
CONDITIONS = {"A": "A · No data (memory only)", "B": "B · Raw match events",
              "C": "C · Statistics summary", "D": "D · Summary + grounding rule"}
BADGE = {"correct": ("✅", "Correct"), "incorrect": ("❌", "Incorrect"),
         "needs_human": ("🟡", "Needs human"), "unverifiable": ("⚪", "Unverifiable")}

st.set_page_config(page_title="Football Report Fact Checker", page_icon="⚽", layout="wide")


# ---------------- data ----------------
@st.cache_data
def load_index():
    idx = pd.read_csv(GT_DIR / "matches_index.csv")
    idx["label"] = (idx.home + " vs " + idx.away + " — " + idx.competition + ", " + idx.stage)
    return idx.sort_values(["competition", "date"])


@st.cache_data
def load_claims(folder):
    p = RESULTS_DIR / f"{folder}_claims.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def find_folders():
    return sorted(p.name[:-len("_claims.csv")] for p in RESULTS_DIR.glob("*_claims.csv")
                  if (REPORTS_DIR / p.name[:-len("_claims.csv")]).is_dir())


def load_report(folder, match_id, cond):
    p = REPORTS_DIR / folder / f"{match_id}_{cond}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def counts(cl):
    ok, bad = (cl.verdict == "correct").sum(), (cl.verdict == "incorrect").sum()
    acc = 100 * ok / (ok + bad) if ok + bad else float("nan")
    return ok, bad, acc


# ---------------- sidebar ----------------
folders = find_folders()
if not folders:
    st.error("No results found. Run check_claims.py first (data/results/<model>_claims.csv).")
    st.stop()
index = load_index()

st.sidebar.title("⚽ Report Fact Checker")
folder = st.sidebar.selectbox("Model", folders, format_func=lambda f: NICE.get(f, f))
match_label = st.sidebar.selectbox("Match", index.label.tolist())
cond = st.sidebar.radio("Input condition", list(CONDITIONS), format_func=CONDITIONS.get)
match = index[index.label == match_label].iloc[0]
mid = int(match.match_id)
st.sidebar.caption(f"Match type: **{match.fame}** · Real score: **{match.home} {match.score} {match.away}**")
st.sidebar.divider()
st.sidebar.caption("Ground truth: StatsBomb open data. Claims extracted by an LLM and checked by rule-based code; "
                   "tactical claims are left for human annotation.")

claims_all = load_claims(folder)
tab1, tab2, tab3, tab4 = st.tabs(["Report check", "Compare conditions", "Match data", "Overall results"])

# ---------------- tab 1: one report ----------------
with tab1:
    st.subheader(f"{match.home} vs {match.away}")
    st.caption(f"{match.competition} · {match.stage} · {match.date} · {NICE.get(folder, folder)} · "
               f"{CONDITIONS[cond]}")
    rep = load_report(folder, mid, cond)
    if rep is None:
        st.warning("No report: the model failed to produce one for this match and condition "
                   "(e.g. it was cut off at the output token limit). This is recorded as a generation failure.")
    else:
        cl = claims_all[(claims_all.match_id == mid) & (claims_all.condition == cond)]
        ok, bad, acc = counts(cl)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Claims extracted", len(cl))
        c2.metric("Correct", int(ok))
        c3.metric("Incorrect", int(bad))
        c4.metric("Accuracy (checkable)", "—" if pd.isna(acc) else f"{acc:.1f}%")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Generated report")
            st.write(rep["report"])
            with st.expander("Show the prompt sent to the model"):
                st.code(rep.get("prompt", "(not saved)"), language=None)
        with right:
            st.markdown("#### Claims and verdicts")
            if cl.empty:
                st.info("No claims extracted for this report yet.")
            order = {"incorrect": 0, "unverifiable": 1, "needs_human": 2, "correct": 3}
            for r in cl.sort_values("verdict", key=lambda s: s.map(order)).itertuples():
                icon, word = BADGE.get(r.verdict, ("•", r.verdict))
                note = f" — *{r.note}*" if isinstance(r.note, str) and r.note and r.verdict != "correct" else ""
                etype = f" `{r.error_type}`" if isinstance(r.error_type, str) and r.error_type else ""
                st.markdown(f"{icon} {r.text}{etype}{note}")

# ---------------- tab 2: all four conditions ----------------
with tab2:
    st.subheader("Same match, four kinds of input")
    cols = st.columns(4)
    for col, c in zip(cols, CONDITIONS):
        with col:
            st.markdown(f"**{CONDITIONS[c]}**")
            rep = load_report(folder, mid, c)
            if rep is None:
                st.warning("No report (generation failed)")
                continue
            cl = claims_all[(claims_all.match_id == mid) & (claims_all.condition == c)]
            ok, bad, acc = counts(cl)
            st.metric("Incorrect claims", int(bad), help=f"{int(ok)} correct, {len(cl)} claims in total")
            st.caption(f"Checkable claims: {int(ok + bad)} · accuracy: {'—' if pd.isna(acc) else f'{acc:.1f}%'}")
            with st.expander("Read report"):
                st.write(rep["report"])
            for r in cl[cl.verdict == "incorrect"].itertuples():
                st.markdown(f"❌ {r.text}  \n*{r.note}*")

# ---------------- tab 3: ground truth ----------------
with tab3:
    st.subheader("Ground truth (what really happened)")
    summary = GT_DIR / f"summary_{mid}.txt"
    st.code(summary.read_text(encoding="utf-8") if summary.exists() else "summary not found", language=None)
    gt = json.loads((GT_DIR / f"gt_{mid}.json").read_text(encoding="utf-8"))
    st.markdown("**Team statistics**")
    st.dataframe(pd.DataFrame(gt["team_stats"]).T, width="stretch")
    st.markdown("**Player statistics**")
    st.dataframe(pd.DataFrame(gt["players"]).sort_values(["team", "goals", "xg"], ascending=[True, False, False]),
                 width="stretch", hide_index=True)

# ---------------- tab 4: overall ----------------
with tab4:
    st.subheader("Overall results (all 36 matches)")
    t2 = ANALYSIS_DIR / "t2_accuracy.csv"
    if t2.exists():
        st.markdown("**Accuracy of automatically checkable claims (95% Wilson CI)**")
        st.dataframe(pd.read_csv(t2), width="stretch", hide_index=True)
    else:
        st.info("Run analyze.py to create the summary tables and charts.")
    for name, cap in [("fig4_1_accuracy.png", "Accuracy by condition and model"),
                      ("fig4_2_density.png", "Checkable claims per report"),
                      ("fig4_3_verdicts.png", "Verdict shares")]:
        if (ANALYSIS_DIR / name).exists():
            st.image(str(ANALYSIS_DIR / name), caption=cap, width="stretch")