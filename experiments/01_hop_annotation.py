"""
Experiment 1: Hop-Count Annotation of MedAlign Questions
=========================================================
Annotates each of the 402 unique MedAlign questions by compositional
depth (hop count), operationalizing the Peng et al. (2402.08164)
framework for clinical reasoning.

Hop-count taxonomy (tied to communication complexity theory):
  1: Direct fact retrieval — single lookup, no composition
  2: Two-fact composition — combine two EHR/knowledge pieces
  3: Multi-step reasoning — 3 sequential retrieval + composition steps
  4: Complex synthesis — 4+ steps, differential Dx, guideline application

Also annotates:
  - requires_ehr (bool): answer requires patient-specific EHR data
  - reasoning_type: factual / computational / comparative / generative

Output: results/hop_annotations.csv
"""

import os
import json
import time
import pandas as pd
import anthropic
from pathlib import Path

BASE    = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
TSV     = BASE / "medalign_instructions_v1_3/clinician-instruction-responses.tsv"
OUT     = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/hop_annotations.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic()

SYSTEM = """You are a medical AI evaluation expert. Your task is to classify clinical EHR questions by their
compositional reasoning depth ("hop count"), as defined by the Peng et al. (2024) communication complexity
framework for transformer limitations.

HOP COUNT DEFINITIONS:
  1 (Direct Retrieval): Answer requires finding exactly ONE piece of information from the EHR
    or applying a single basic medical fact. No composition needed.
    Examples: "What medications is the patient on?", "Does the patient smoke?", "Who is the treating physician?"

  2 (Simple Composition): Answer requires combining EXACTLY TWO pieces of information,
    OR applying one formula/calculation, OR matching one patient fact against one guideline.
    Examples: "Calculate fluid requirement by weight", "Is this A1c within target?",
    "What screening tests are appropriate for this patient's age?"

  3 (Multi-Step Reasoning): Answer requires 3 distinct information pieces or sequential reasoning steps.
    Examples: "Given kidney function and current medications, what dose adjustments are needed?",
    "Is this imaging finding significant given the patient's history and symptoms?"

  4 (Complex Synthesis): Answer requires 4+ reasoning steps, differential diagnosis,
    cross-domain guideline application, complex clinical judgment, or full synthesis.
    Examples: "What is the 10-year ASCVD risk?", "Draft a management plan",
    "What are the Fleischner recommendations given this patient's history?",
    "What are typical artifacts in PSMA PET?" (pure knowledge composition without EHR)

REASONING TYPE:
  factual: retrieve and report a fact
  computational: perform a calculation or formula application
  comparative: compare patient data against a standard/threshold/guideline
  generative: synthesize/draft/summarize free text

Return ONLY valid JSON, no other text:
{
  "hop_count": <1|2|3|4>,
  "requires_ehr": <true|false>,
  "reasoning_type": "<factual|computational|comparative|generative>",
  "rationale": "<one sentence explaining hop count assignment>"
}"""


def annotate_question(question: str, clinician_response: str, evidence: str) -> dict:
    """Call Claude to annotate a single question."""
    user_content = f"""Question: {question}

Clinician reference answer: {clinician_response}

Evidence cited: {evidence}

Classify this question by hop count."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_content}]
    )
    raw = msg.content[0].text.strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    # Check if partial results exist (resume support)
    df_instruct = pd.read_csv(TSV, sep="\t")
    print(f"Loaded {len(df_instruct)} unique questions")

    if OUT.exists():
        done = pd.read_csv(OUT)
        done_ids = set(done["instruction_id"].tolist())
        print(f"Resuming: {len(done_ids)} already annotated")
    else:
        done = pd.DataFrame()
        done_ids = set()

    results = done.to_dict("records") if len(done) else []

    for i, row in df_instruct.iterrows():
        iid = row["instruction_id"]
        if iid in done_ids:
            continue

        q = str(row["question"])
        cr = str(row.get("clinician_response", ""))
        ev = str(row.get("evidence", ""))

        try:
            annotation = annotate_question(q, cr, ev)
            record = {
                "instruction_id": iid,
                "question": q,
                "submitter_specialty": row.get("submitter_specialty", ""),
                "hop_count": annotation["hop_count"],
                "requires_ehr": annotation["requires_ehr"],
                "reasoning_type": annotation["reasoning_type"],
                "rationale": annotation["rationale"],
            }
            results.append(record)
            print(f"[{i+1:3d}/{len(df_instruct)}] hop={annotation['hop_count']} "
                  f"ehr={annotation['requires_ehr']} type={annotation['reasoning_type']}: {q[:60]}")
        except Exception as e:
            print(f"  ERROR on {iid}: {e}")
            record = {
                "instruction_id": iid,
                "question": q,
                "submitter_specialty": row.get("submitter_specialty", ""),
                "hop_count": None,
                "requires_ehr": None,
                "reasoning_type": None,
                "rationale": f"ERROR: {e}",
            }
            results.append(record)

        # Save checkpoint every 20 questions
        if (i + 1) % 20 == 0:
            pd.DataFrame(results).to_csv(OUT, index=False)
            print(f"  [checkpoint saved: {len(results)} rows]")

        time.sleep(0.1)  # rate limit

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT, index=False)
    print(f"\nSaved {len(df_out)} annotations to {OUT}")

    # Summary
    print("\nHop count distribution:")
    print(df_out["hop_count"].value_counts().sort_index())
    print("\nReasoning type distribution:")
    print(df_out["reasoning_type"].value_counts())


if __name__ == "__main__":
    main()
