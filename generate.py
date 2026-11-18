"""
generate.py — Phase 2b (M4 Report generator)
Sends every (match, condition) prompt to an LLM and saves the report with full metadata.
Resumable: reports that already exist are skipped, so you can stop and restart safely.

Providers (pick one with --provider):
  gemini     needs GEMINI_API_KEY      (free tier at aistudio.google.com)
  openai     needs OPENAI_API_KEY
  anthropic  needs ANTHROPIC_API_KEY
  ollama     free, runs locally         (install from ollama.com, then: ollama pull qwen2.5:7b)
  dry        no API call; saves prompts only (use this to test the pipeline)

Examples:
  python generate.py --provider dry --model none
  python generate.py --provider ollama --model qwen2.5:7b
  python generate.py --provider gemini --model <gemini model name> --conditions C D --limit 5
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from prompts import CONDITIONS, GT_DIR, build_prompt

OUT_DIR = Path("data/reports")
TEMPERATURE = 0.0      # deterministic as possible, so results are reproducible
MAX_TOKENS = 700       # ~250-word report fits easily
TIMEOUT = 180          # seconds per request


# ---------------- provider calls ----------------
def call_gemini(model, system, user):
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        # thinking models spend tokens on hidden reasoning first, so the budget must be large
        "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": 8192},
    }
    r = requests.post(url, params={"key": key}, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    cand = data["candidates"][0]
    # keep only the visible answer, skip any "thought" parts
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    if cand.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("report was cut off (MAX_TOKENS) - not saved")
    return text, data.get("modelVersion", model)

def call_openai(model, system, user):
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"], data.get("model", model)


def call_anthropic(model, system, user):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        json={"model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return "".join(b.get("text", "") for b in data["content"]), data.get("model", model)


def call_ollama(model, system, user):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    r = requests.post(f"{host}/api/chat", json={
        "model": model, "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "seed": 42},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=TIMEOUT * 3)
    r.raise_for_status()
    return r.json()["message"]["content"], model


def call_dry(model, system, user):
    return "[DRY RUN — no model called]", "dry"


PROVIDERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic,
             "ollama": call_ollama, "dry": call_dry}


def call_with_retry(fn, model, system, user, tries=5):
    """Retry on rate limits / temporary server errors with exponential backoff."""
    for attempt in range(tries):
        try:
            return fn(model, system, user)
        except requests.HTTPError as e:
            code = e.response.status_code
            if code in (429, 500, 502, 503, 504, 529) and attempt < tries - 1:
                wait = 2 ** attempt * 5
                print(f"    HTTP {code}, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"HTTP {code}: {e.response.text[:300]}") from e
        except requests.ConnectionError as e:
            if attempt < tries - 1:
                time.sleep(10)
            else:
                raise


# ---------------- main loop ----------------
def safe(name):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def main():
    ap = argparse.ArgumentParser(description="Generate LLM match reports")
    ap.add_argument("--provider", required=True, choices=PROVIDERS)
    ap.add_argument("--model", required=True, help="model name as the provider spells it")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N matches (for a pilot)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls (free tiers need more)")
    args = ap.parse_args()

    index = pd.read_csv(GT_DIR / "matches_index.csv")
    if args.limit:
        index = index.head(args.limit)

    fn = PROVIDERS[args.provider]
    out_dir = OUT_DIR / safe(f"{args.provider}_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0

    for row in index.itertuples():
        for cond in args.conditions:
            out = out_dir / f"{row.match_id}_{cond}.json"
            if out.exists():
                skipped += 1
                continue
            prompt = build_prompt(row.match_id, cond)
            t0 = time.time()
            try:
                text, model_version = call_with_retry(fn, args.model, prompt["system"], prompt["user"])
            except Exception as e:
                failed += 1
                print(f"  [fail] {row.home} vs {row.away} ({cond}): {e}")
                continue
            record = {
                "match_id": int(row.match_id), "competition": row.competition,
                "home": row.home, "away": row.away, "condition": cond,
                "provider": args.provider, "model": args.model, "model_version": model_version,
                "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "latency_s": round(time.time() - t0, 2),
                "prompt": prompt, "report": text.strip(),
            }
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            done += 1
            print(f"  [ok] {row.home} vs {row.away} ({cond}) - {len(text.split())} words")
            if args.provider != "dry":
                time.sleep(args.sleep)

    print(f"\nDone: {done} new, {skipped} already existed, {failed} failed -> {out_dir}/")


if __name__ == "__main__":
    main()