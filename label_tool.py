"""
label_tool.py — faster human labelling (Phase 5, helper)
Shows one claim at a time with the matching evidence from the real match data, and saves your label
straight into the annotation Excel file. It never shows the model, condition or checker verdict.

Run (close the Excel file first!):
  streamlit run label_tool.py
"""
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from openpyxl import load_workbook

from lookup import fact_sheet

ANN = Path("data/annotation")
GT_DIR = Path("data/ground_truth")
LABELS = {"correct": "✅ Correct", "incorrect": "❌ Incorrect", "not_checkable": "⚪ Not checkable"}
KEYWORDS = ["card", "yellow", "red", "substitut", "shot", "corner", "foul", "penalt", "save", "post", "woodwork",
            "assist", "pass", "xg", "own goal", "header"]

st.set_page_config(page_title="Claim labelling", page_icon="🏷️", layout="wide")


# ---------------- data ----------------
def sheet_path():
    files = sorted(ANN.glob("annotation_*.xlsx"))
    if not files:
        st.error("No annotation files in data/annotation. Run annotate_export.py first.")
        st.stop()
    return st.sidebar.selectbox("Annotation file", files, format_func=lambda p: p.name)


def read_rows(path):
    df = pd.read_excel(path, sheet_name="claims")
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df["comment"] = df["comment"].fillna("").astype(str)
    return df


@st.cache_data
def reports(path):
    r = pd.read_excel(path, sheet_name="reports")
    return dict(zip(r.report, r.text))


@st.cache_data
def match_ids():
    """row_id -> match_id from key.csv (only the match id is used, never the verdict)."""
    p = ANN / "key.csv"
    return dict(zip(*pd.read_csv(p, usecols=["row_id", "match_id"]).values.T)) if p.exists() else {}


@st.cache_data
def index():
    idx = pd.read_csv(GT_DIR / "matches_index.csv")
    idx["long"] = idx.home + " vs " + idx.away + " (" + idx.competition + ", " + idx.stage + ")"
    idx["short"] = idx.home + " vs " + idx.away + " (" + idx.competition + ")"
    return idx


def find_match(row):
    idx = index()
    mid = match_ids().get(row.row_id)
    hit = idx[idx.match_id == mid] if mid is not None else idx[(idx.long == row.match) | (idx.short == row.match)]
    return hit.iloc[0] if len(hit) == 1 else None


def evidence(match, claim):
    """Summary + only the event / player lines that mention a name or keyword from the claim."""
    lines = fact_sheet(match)
    cut = next((i for i, l in enumerate(lines) if l.startswith("Cards:")), len(lines))
    summary, details = lines[:cut], lines[cut:]
    names = {w.lower() for w in re.findall(r"[A-ZÀ-Ý][\wÀ-ÿ'’-]{2,}", claim)}
    names -= {match.home.lower(), match.away.lower(), "the", "they", "their", "his", "after", "before"}
    words = names | {k for k in KEYWORDS if k in claim.lower()}
    player_section = False
    hits = []
    for l in details:
        if l and not l.startswith("  "):
            player_section = l.startswith("All players")
            continue
        # player lines only by name (keywords like 'pass' or 'assist' would match every player)
        keys = names if player_section else words
        if l.startswith("  ") and any(w in l.lower() for w in keys):
            hits.append(l)
    # whole sections when the claim is about that kind of event (e.g. "a substitution at halftime")
    sections, current = {}, None
    for l in details:
        if l and not l.startswith("  "):
            current = l
        elif current:
            sections.setdefault(current, []).append(l)
    c = claim.lower()
    for head, keys in [("Substitutions:", ["substitut", "replaced", "came on", "brought on", "introduced", "change"]),
                       ("Cards:", ["card", "booked", "booking", "caution", "sent off", "dismiss"]),
                       ("Penalty shootout (in kick order):", ["shootout", "shoot-out", "spot-kick", "spot kick",
                                                               "decisive penalty", "on penalties"])]:
        if any(k in c for k in keys):
            hits = [f"--- all {head[:-1].lower()} ---"] + sections.get(head, ["  (none)"]) + hits
    return "\n".join(summary), "\n".join(dict.fromkeys(hits))


def save(path, row_id, label, comment):
    wb = load_workbook(path)
    ws = wb["claims"]
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, 1).value == row_id:
            ws.cell(r, 5).value = label or None
            ws.cell(r, 6).value = comment or None
            break
    wb.save(path)


# ---------------- page ----------------
path = sheet_path()
try:
    df = read_rows(path)
except PermissionError:
    st.error("The Excel file is open. Close it in Excel, then refresh this page.")
    st.stop()

done = (df.label != "").sum()
target = st.sidebar.number_input("Your target (rows)", 50, len(df), min(300, len(df)), step=50)
st.sidebar.progress(min(done / target, 1.0), text=f"{done} / {target} labelled (file has {len(df)})")
only_open = st.sidebar.toggle("Show only unlabelled rows", value=True)
b_file = ANN / "annotation_B.xlsx"
shared_first = False
if path.name == "annotation_A.xlsx" and b_file.exists():
    shared_first = st.sidebar.toggle("Rows shared with annotator B first", value=True,
                                     help="These rows are needed for Cohen's kappa, so do them first.")
if shared_first:
    shared = set(pd.read_excel(b_file, sheet_name="claims", usecols=["row_id"]).row_id)
    df["shared"] = df.row_id.isin(shared)
    df = df.sort_values(["shared", "row_id"], ascending=[False, True])
    st.sidebar.caption(f"Shared rows done: {((df.label != '') & df.shared).sum()} / {df.shared.sum()}")
todo = df[df.label == ""] if only_open else df
if todo.empty:
    st.success("All rows are labelled. Run `python agreement.py` next.")
    st.stop()

if "pos" not in st.session_state:
    st.session_state.pos = 0
st.session_state.pos = min(st.session_state.pos, len(todo) - 1)
row = todo.iloc[st.session_state.pos]

st.caption(f"Row {row.row_id} · report {row.report} · {st.session_state.pos + 1} of {len(todo)} in this view")
st.markdown(f"### {row.claim}")
st.markdown(f"**Match:** {row.match}")

match = find_match(row)
left, right = st.columns([1, 1])
with left:
    if match is None:
        st.warning("Could not identify the match; use lookup.py or the 'matches' sheet.")
    else:
        summary, hits = evidence(match, str(row.claim))
        st.markdown("**Relevant events and player lines**")
        st.code(hits or "(nothing specific found — use the summary)", language=None)
        st.markdown("**Match summary**")
        st.code(summary, language=None)
with right:
    with st.expander("Full report this claim came from"):
        st.write(reports(path).get(row.report, "(report not found)"))
    st.markdown("**Your label**")
    comment = st.text_input("Comment (optional)", value=row.comment, key=f"c{row.row_id}")
    cols = st.columns(3)
    for col, (value, text) in zip(cols, LABELS.items()):
        if col.button(text, key=f"{value}{row.row_id}", width="stretch",
                      type="primary" if row.label == value else "secondary"):
            try:
                save(path, int(row.row_id), value, comment)
            except PermissionError:
                st.error("Could not save: close the Excel file first.")
                st.stop()
            if not only_open:
                st.session_state.pos += 1
            st.rerun()
    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button("◀ Previous", width="stretch"):
        st.session_state.pos = max(0, st.session_state.pos - 1)
        st.rerun()
    if b2.button("Skip ▶", width="stretch"):
        st.session_state.pos += 1
        st.rerun()
    if b3.button("Clear label", width="stretch"):
        save(path, int(row.row_id), "", comment)
        st.rerun()
    st.caption("Rules: correct = data or well-known facts agree · incorrect = data contradicts it · "
               "not checkable = opinion, tactics without clear numbers, or tournament context you cannot verify.")