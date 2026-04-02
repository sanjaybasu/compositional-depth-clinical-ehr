"""
Experiment 2: Architectural Comparison Across Hop-Count Classes
===============================================================
Tests three inference architectures on a stratified sample of MedAlign
questions, measuring omission and hallucination rates per hop count.

Architectures tested:
  A) Zero-shot (direct): baseline transformer, no explicit reasoning
  B) Chain-of-thought (CoT): test-time compute — Peng solution class 1
  C) RAG-grounded: retrieval from EHR — Peng solution class 2

Scoring: judge LLM (Claude) assesses correctness against clinician ground truth.
Output: results/arch_comparison_results.csv, results/arch_comparison_scores.csv

Scientific prediction (Peng 2402.08164):
  - Error rate should increase with hop count for zero-shot
  - CoT should partially flatten the hop_count → error_rate curve
  - RAG should outperform both at hop=3+ for EHR-requiring questions
"""

import os
import json
import time
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import anthropic
from pathlib import Path

BASE     = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
EHR_DIR  = BASE / "medalign_instructions_v1_3/ehrs"
RESP_TSV = BASE / "medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv"
ANNOT    = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/hop_annotations.csv")
OUT_RAW  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/arch_comparison_raw.csv")
OUT_SCORE= Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/arch_comparison_scores.csv")
OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

N_PER_HOP = 50  # stratified sample per hop count class
SEED      = 42
MODEL     = "claude-sonnet-4-6"

client = anthropic.Anthropic()


# ── EHR parsing ─────────────────────────────────────────────────────────────

def extract_ehr_text(xml_path: Path, max_chars: int = 8000) -> str:
    """Extract relevant text from EHR XML, truncated to fit context window."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        segments = []
        for encounter in root.findall("encounter"):
            for entry in encounter.findall(".//entry"):
                ts = entry.get("timestamp", "")
                for event in entry.findall("event"):
                    etype = event.get("type", "")
                    name  = event.get("name", "")
                    text  = (event.text or "").strip()
                    if text:
                        segments.append(f"[{ts}] {etype}/{name}: {text}")
        full = "\n".join(segments)
        return full[:max_chars]
    except Exception as e:
        return f"[EHR parse error: {e}]"


# ── Generation functions ─────────────────────────────────────────────────────

def generate_zero_shot(question: str, ehr_text: str) -> str:
    """Zero-shot: just answer the question."""
    msgs = [{"role": "user", "content":
             f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\nAnswer concisely."}]
    r = client.messages.create(model=MODEL, max_tokens=512, messages=msgs)
    return r.content[0].text.strip()


def generate_cot(question: str, ehr_text: str) -> str:
    """CoT: explicit step-by-step reasoning (test-time compute)."""
    msgs = [{"role": "user", "content":
             f"""Patient EHR:
{ehr_text}

Question: {question}

Instructions: Think step by step. First identify what information is needed, then locate each piece in the EHR, compose the pieces, and provide your final answer. Format:
Step 1 - Identify needed information: ...
Step 2 - Retrieve from EHR: ...
Step 3 - Compose/reason: ...
Final Answer: ..."""}]
    r = client.messages.create(model=MODEL, max_tokens=1024, messages=msgs)
    return r.content[0].text.strip()


def generate_rag(question: str, ehr_text: str) -> str:
    """RAG-grounded: extract relevant EHR sections explicitly before answering."""
    # Step 1: extract relevant passages
    extract_msgs = [{"role": "user", "content":
                     f"""EHR:
{ehr_text}

Question: {question}

Extract only the EHR passages (verbatim) that are relevant to answering this question. If nothing is relevant, say "NO RELEVANT EHR DATA"."""}]
    r1 = client.messages.create(model=MODEL, max_tokens=512, messages=extract_msgs)
    retrieved = r1.content[0].text.strip()

    # Step 2: answer using only retrieved passages
    answer_msgs = [{"role": "user", "content":
                    f"""Relevant EHR passages:
{retrieved}

Question: {question}

Answer based only on the above passages. If the answer cannot be determined, state that explicitly."""}]
    r2 = client.messages.create(model=MODEL, max_tokens=512, messages=answer_msgs)
    return r2.content[0].text.strip()


# ── Scoring function ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert medical annotator. Given a clinical question, a clinician's reference answer,
and a model's response, assess whether the model's response is correct.

Classify as:
  "correct": Model answer substantially agrees with the clinician reference answer
  "partial": Model answer contains relevant information but is incomplete or partially wrong
  "incorrect": Model answer is wrong, irrelevant, or contradicts the clinician reference
  "omission": Model refused to answer, said it couldn't determine, or gave an empty response

Also classify error type:
  "none": answer is correct
  "omission": model failed to retrieve/provide required information
  "hallucination": model stated something not in the EHR or medically incorrect
  "reasoning_error": model had the right facts but composed them incorrectly

Return ONLY valid JSON: {"correctness": "...", "error_type": "...", "reasoning": "one sentence"}"""


def score_response(question: str, clinician_ref: str, model_response: str) -> dict:
    msgs = [{"role": "user", "content":
             f"Question: {question}\n\nClinician reference: {clinician_ref}\n\nModel response: {model_response}"}]
    r = client.messages.create(model=MODEL, max_tokens=256,
                               system=JUDGE_SYSTEM, messages=msgs)
    raw = r.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not ANNOT.exists():
        print(f"ERROR: Run 01_hop_annotation.py first. Missing {ANNOT}")
        return

    annot = pd.read_csv(ANNOT)
    annot = annot.dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int)
    annot["hop_class"] = annot["hop_count"].clip(upper=4)  # 4 = "4+"

    df_resp = pd.read_csv(RESP_TSV, sep="\t")
    # Get unique question → filename + clinician_response mapping
    ref_map = df_resp.drop_duplicates("instruction_id")[
        ["instruction_id","filename","clinician_response"]
    ].set_index("instruction_id")

    # Stratified sample: N_PER_HOP per hop class
    rng = np.random.default_rng(SEED)
    sample_ids = []
    for hop in [1, 2, 3, 4]:
        pool = annot[annot["hop_class"] == hop]["instruction_id"].tolist()
        n    = min(N_PER_HOP, len(pool))
        chosen = rng.choice(pool, size=n, replace=False).tolist()
        sample_ids.extend(chosen)
        print(f"  hop={hop}: {n} questions sampled from {len(pool)} available")

    sample = annot[annot["instruction_id"].isin(sample_ids)].copy()
    print(f"\nTotal sample: {len(sample)} questions")

    # Resume support
    if OUT_RAW.exists():
        done_df = pd.read_csv(OUT_RAW)
        done_key = set(zip(done_df["instruction_id"], done_df["condition"]))
        print(f"Resuming: {len(done_df)} rows already done")
    else:
        done_df = pd.DataFrame()
        done_key = set()

    records = done_df.to_dict("records") if len(done_df) else []

    conditions = ["zero_shot", "cot", "rag"]
    generators = {"zero_shot": generate_zero_shot,
                  "cot":       generate_cot,
                  "rag":       generate_rag}

    total = len(sample) * len(conditions)
    n_done = 0

    for _, row in sample.iterrows():
        iid      = row["instruction_id"]
        question = row["question"]
        hop      = row["hop_class"]
        req_ehr  = row.get("requires_ehr", True)

        if iid not in ref_map.index:
            continue
        filename  = ref_map.loc[iid, "filename"]
        clinician = str(ref_map.loc[iid, "clinician_response"])

        ehr_path = EHR_DIR / filename
        ehr_text = extract_ehr_text(ehr_path) if ehr_path.exists() else "[EHR not found]"

        for cond in conditions:
            if (iid, cond) in done_key:
                n_done += 1
                continue

            try:
                response = generators[cond](question, ehr_text)
                score    = score_response(question, clinician, response)
                binary   = 1 if score["correctness"] == "correct" else 0

                record = {
                    "instruction_id":  iid,
                    "condition":       cond,
                    "hop_count":       hop,
                    "requires_ehr":    req_ehr,
                    "question":        question,
                    "clinician_ref":   clinician,
                    "model_response":  response,
                    "correctness":     score["correctness"],
                    "error_type":      score["error_type"],
                    "binary_correct":  binary,
                    "score_reasoning": score.get("reasoning",""),
                }
                records.append(record)
                n_done += 1
                print(f"[{n_done:4d}/{total}] hop={hop} {cond:10s} "
                      f"{score['correctness']:12s} {score['error_type']:18s}: {question[:50]}")

            except Exception as e:
                print(f"  ERROR {iid}/{cond}: {e}")
                n_done += 1

            time.sleep(0.2)

        # Checkpoint every 30 instruction_ids
        if n_done % 90 == 0 and records:
            pd.DataFrame(records).to_csv(OUT_RAW, index=False)
            print(f"  [checkpoint: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_RAW, index=False)
    print(f"\nSaved raw results: {len(df_out)} rows → {OUT_RAW}")

    # Aggregate scores
    agg = df_out.groupby(["condition","hop_count"]).agg(
        n=("binary_correct","count"),
        accuracy=("binary_correct","mean"),
        omission_rate=("error_type", lambda x: (x == "omission").mean()),
        hallucination_rate=("error_type", lambda x: (x == "hallucination").mean()),
        reasoning_error_rate=("error_type", lambda x: (x == "reasoning_error").mean()),
    ).reset_index()
    agg.to_csv(OUT_SCORE, index=False)
    print(f"Saved aggregated scores → {OUT_SCORE}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
