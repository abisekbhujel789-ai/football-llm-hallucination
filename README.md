# Measuring and Reducing Hallucinations in LLM-Generated Football Match Analysis

Bachelor's thesis project, Smart Computing, Kyungdong University Global (2026).
Author: Bhujel Abisek

Large language models (LLMs) can write football match reports, but they sometimes state facts
that are not true. This project measures how often that happens and whether giving the model
match data reduces it.

Two models — **Gemini 3.1 Flash-Lite** and **Gemma 4 26B-A4B** — wrote reports on **36 matches**
(FIFA World Cup 2022, UEFA Euro 2024, Copa América 2024) under four input conditions:

| Condition | What the model receives |
|---|---|
| A | No data (teams, competition and date only) |
| B | Raw key match events |
| C | A short statistics summary |
| D | The summary plus a rule to use only the given facts |

Every report is split into atomic claims, and each claim is checked automatically against a
ground truth built from [StatsBomb open data](https://github.com/statsbomb/open-data).

## Main results

| Accuracy of checkable claims | A: no data | B: events | C: summary | D: summary + rule |
|---|---|---|---|---|
| Gemini 3.1 Flash-Lite | 99.3% | 99.4% | 99.7% | 99.9% |
| Gemma 4 | **50.7%** | 98.8% | 99.7% | 100% |

- With match data, both models were almost perfectly accurate (6 errors in 2,630 claims).
- Without data, Gemma invented scorers, minutes and even players, and was much less accurate on
  ordinary group games (41.5%) than on famous matches (70.5%), which points to memorisation.
- The rule-based checker agreed with human judgement on 97.4% of sampled claims.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `ground_truth.py` | Ground truth, summaries and events for each match |
| 2 | `prompts.py` | Prompts for conditions A–D (used by step 3) |
| 3 | `generate.py` | One report per match, condition and model |
| 4 | `extract_claims.py` | Atomic claims for each report |
| 5 | `check_claims.py` | A verdict for every claim |
| 6 | `analyze.py` | Tables, statistical tests and charts |
| 7 | `annotate_export.py`, `label_tool.py`, `lookup.py` | Blind human labelling |
| 8 | `agreement.py` | Checker–human agreement and final accuracy |
| – | `app.py` | Demo app to browse reports and their checked claims |

## Setup (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The models are called through the Gemini API (the free tier is enough). Get a key from
[Google AI Studio](https://aistudio.google.com/) and set it as an environment variable,
**never in the code**:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

## How to reproduce

```powershell
# 1. Ground truth for the 36 matches
python ground_truth.py sample

# 2. Reports (run once per model; scripts resume if interrupted)
python generate.py --provider gemini --model gemini-3.1-flash-lite --sleep 10
python generate.py --provider gemini --model gemma-4-26b-a4b-it --sleep 10

# 3. Claims (Flash-Lite is the extractor for both models)
python extract_claims.py --reports data/reports/gemini_gemini-3.1-flash-lite --provider gemini --model gemini-3.1-flash-lite --sleep 10
python extract_claims.py --reports data/reports/gemini_gemma-4-26b-a4b-it --provider gemini --model gemini-3.1-flash-lite --sleep 10

# 4. Check every claim
python check_claims.py --claims data/claims/gemini_gemini-3.1-flash-lite
python check_claims.py --claims data/claims/gemini_gemma-4-26b-a4b-it

# 5. Tables, statistics and charts (saved in data/results/analysis/)
python analyze.py

# 6. Human validation (optional)
python annotate_export.py --auto-frac 0.1
python -m streamlit run label_tool.py
python agreement.py

# Demo app
python -m streamlit run app.py
```

Because the models run at temperature 0, the same prompts give the same reports, but model
versions on the API may change over time.

## Data and credits

Match data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under its free-data
terms. Generated data are not included in this repository; every file in `data/` is
recreated by the scripts above.