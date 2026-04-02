"""
Experiment 8: Rerun with ET-16K and Explicit-CoT conditions
=============================================================
Addresses reviewer concern that:
  1. budget_tokens=3000 may have constrained thinking (ET-16K condition)
  2. Naive CoT prompt does not force decomposition (ExplicitCoT condition)

Two new conditions run against the SAME 301 questions as Experiment 6:
  C: claude-sonnet-4-6  ET-16K        — thinking enabled, budget_tokens=16000
  D: claude-sonnet-4-6  ExplicitCoT   — no thinking, structured step-by-step prompt

ZS results from Experiment 6 are unchanged and will be used in the combined analysis.

Outputs:
  results/claude_et16k_raw.csv        — condition C
  results/claude_explicit_cot_raw.csv — condition D

Run:
  python3 experiments/08_et16k_explicit_cot_rerun.py
"""

import json, time, xml.etree.ElementTree as ET_xml, os
from pathlib import Path
import pandas as pd, numpy as np
import anthropic, openai
from anthropic import RateLimitError as AnthropicRateLimitError

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
              "/medalign_instructions_v1_3")
ANNOT  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
              "/results/hop_annotations.csv")
ORIG   = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
              "/results/claude_experiment_raw.csv")
OUT    = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results")

def load_key(env_path, key):
    with open(env_path) as f:
        for line in f:
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""

ENV           = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or load_key(ENV, "ANTHROPIC_API_KEY")
OPENAI_KEY    = load_key(ENV, "OPENAI_API_KEY")

anth_client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

# ── Judge (same as Experiment 6) ───────────────────────────────────────────────
JUDGE_SYS = """You are a clinical QA evaluator. Score model responses against clinician reference answers.
Return ONLY valid JSON — no other text:
{"correct": true or false, "error_type": "none" | "omission" | "hallucination" | "reasoning_error", "confidence": 0.0-1.0}
Definitions:
- correct: model response substantially agrees with the clinician reference
- omission: model failed to answer, refused, gave empty/irrelevant response
- hallucination: model stated facts not in the EHR or contradicts medical knowledge
- reasoning_error: model had the right facts but reached the wrong conclusion
- confidence: your confidence in this scoring (0=unsure, 1=certain)"""


# ── EHR parsing (same as Experiment 6) ────────────────────────────────────────
def extract_ehr(xml_path: Path, max_chars: int = 8000) -> tuple:
    """Returns (truncated_ehr_text, full_char_count)."""
    try:
        tree = ET_xml.parse(xml_path)
        segs = []
        for enc in tree.getroot().findall("encounter"):
            for entry in enc.findall(".//entry"):
                ts = entry.get("timestamp", "")
                for ev in entry.findall("event"):
                    txt = (ev.text or "").strip()
                    if txt:
                        segs.append(f"[{ts}] {ev.get('type','')}: {txt}")
        full = "\n".join(segs)
        return full[:max_chars], len(full)
    except Exception as e:
        return f"[EHR error: {e}]", 0


# ── Retry wrapper ──────────────────────────────────────────────────────────────
def _call_claude_with_retry(kwargs: dict, max_retries: int = 6) -> object:
    delay = 30
    for attempt in range(max_retries):
        try:
            return anth_client.messages.create(**kwargs)
        except AnthropicRateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [429 rate-limit, sleeping {delay}s]")
            time.sleep(delay)
            delay = min(delay * 2, 300)
        except Exception:
            raise


# ── Generators ─────────────────────────────────────────────────────────────────
def claude_et16k(question: str, ehr: str) -> tuple:
    """Extended thinking with budget_tokens=16000. Tests whether 3K budget was limiting.
    max_tokens must exceed budget_tokens to leave room for the visible text response.
    Using max_tokens=17000 (budget_tokens=16000 + ~1000 for output).
    """
    r = _call_claude_with_retry(dict(
        model="claude-sonnet-4-6",
        max_tokens=17000,
        thinking={"type": "enabled", "budget_tokens": 16000},
        messages=[{"role": "user", "content":
            f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."}]
    ))
    # Estimate thinking tokens from word count × 1.3 (same metric as Exp 6)
    think_tokens = sum(
        len(b.thinking.split()) * 1.3
        for b in r.content if hasattr(b, "thinking") and b.type == "thinking"
    )
    text_blocks = [b for b in r.content if b.type == "text"]
    text = text_blocks[0].text.strip() if text_blocks else ""
    return text, int(think_tokens)


# Explicit step-by-step CoT prompt — forces structured decomposition without thinking
EXPLICIT_COT_SYS = (
    "You are a clinical AI assistant. When answering questions about a patient's "
    "EHR, always reason step by step before giving your final answer."
)
EXPLICIT_COT_TMPL = """\
Patient EHR:
{ehr}

Clinical Question: {question}

Please answer this question by working through these steps explicitly:
Step 1 — What specific pieces of information from the EHR are needed to answer this question?
Step 2 — Find and quote the relevant EHR entries or state if the information is not present.
Step 3 — Reason through how these entries connect to answer the question.
Step 4 — Final Answer: State your answer clearly and concisely."""


def claude_explicit_cot(question: str, ehr: str) -> tuple:
    """Structured step-by-step CoT prompt, no extended thinking. Tests decomposition benefit."""
    r = _call_claude_with_retry(dict(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=EXPLICIT_COT_SYS,
        messages=[{"role": "user", "content":
            EXPLICIT_COT_TMPL.format(ehr=ehr, question=question)}]
    ))
    return r.content[0].text.strip(), 0


# ── Judge ──────────────────────────────────────────────────────────────────────
def judge(question: str, ref: str, response: str) -> dict:
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=120,
        messages=[
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user",   "content":
                f"Question: {question}\nClinician reference: {ref}\n"
                f"Model response: {response}"}
        ]
    )
    raw = r.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Main ───────────────────────────────────────────────────────────────────────
def run_condition(condition_name: str, gen_fn, out_csv: Path, sample: pd.DataFrame,
                  ref_map: pd.DataFrame):
    """Run one new condition, with resume support."""
    if out_csv.exists():
        done_df  = pd.read_csv(out_csv)
        done_ids = set(done_df["instruction_id"].tolist())
        records  = done_df.to_dict("records")
        print(f"[{condition_name}] Resuming from {len(done_df)} existing rows")
    else:
        done_ids, records = set(), []

    total = len(sample)
    for i, (_, row) in enumerate(sample.iterrows()):
        iid      = row["instruction_id"]
        if iid in done_ids:
            continue
        question = row["question"]
        hop      = int(row["hop_count"])
        fn_      = str(ref_map.loc[iid, "filename"])
        cr       = str(ref_map.loc[iid, "clinician_response"])
        ehr, ehr_chars = extract_ehr(BASE / "ehrs" / fn_)
        q_tokens = len(question.split())
        domain   = str(row.get("submitter_specialty", ""))

        try:
            resp, think_tokens = gen_fn(question, ehr)
            sc = judge(question, cr, resp)
            records.append({
                "instruction_id":       iid,
                "condition":            condition_name,
                "hop_count":            hop,
                "ehr_char_count":       ehr_chars,
                "question_token_count": q_tokens,
                "domain":               domain,
                "question":             question,
                "clinician_ref":        cr,
                "model_response":       resp,
                "correct":              int(sc.get("correct", False)),
                "error_type":           sc.get("error_type", "error"),
                "judge_confidence":     sc.get("confidence", 0.0),
                "thinking_tokens":      think_tokens,
            })
            done_ids.add(iid)
            mark = "✓" if sc.get("correct") else "✗"
            print(f"[{i+1:3d}/{total}] {condition_name:28s} {mark} "
                  f"hop={hop} {sc.get('error_type','?'):18s}: {question[:45]}")
        except Exception as e:
            print(f"  ERROR {condition_name} on {iid}: {e}")

        # Checkpoint every 30
        if (i + 1) % 30 == 0 and records:
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print(f"  [checkpoint: {len(records)}/{total}]")

        time.sleep(8)

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[{condition_name}] Saved {len(df_out)} rows → {out_csv}")
    return df_out


def main():
    annot = pd.read_csv(ANNOT).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)

    df_resp = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map = (df_resp.drop_duplicates("instruction_id")
               [["instruction_id", "filename", "clinician_response"]]
               .set_index("instruction_id"))

    # Use same 301 questions as Experiment 6
    orig = pd.read_csv(ORIG)
    exp6_ids = set(orig["instruction_id"].unique())
    sample = annot[annot["instruction_id"].isin(exp6_ids)].copy()
    print(f"Questions from Exp 6: {len(sample)}")
    print("Hop distribution:", sample["hop_count"].value_counts().sort_index().to_dict())

    # Run ET-16K
    print("\n" + "="*60)
    print("CONDITION C: Extended Thinking, budget_tokens=16000")
    print("="*60)
    run_condition(
        "claude_et16k",
        claude_et16k,
        OUT / "claude_et16k_raw.csv",
        sample, ref_map
    )

    # Run Explicit CoT
    print("\n" + "="*60)
    print("CONDITION D: Explicit Step-by-Step CoT (no thinking)")
    print("="*60)
    run_condition(
        "claude_explicit_cot",
        claude_explicit_cot,
        OUT / "claude_explicit_cot_raw.csv",
        sample, ref_map
    )

    print("\nAll conditions complete.")
    print("Run 07_analysis_and_figures.py to generate combined analysis.")


if __name__ == "__main__":
    main()
