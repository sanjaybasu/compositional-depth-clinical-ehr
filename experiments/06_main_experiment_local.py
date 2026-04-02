"""
Experiment 6: Main Compositional Depth Study — Local Sequential Execution
==========================================================================
Same scientific design as Experiment 5 but runs locally to avoid Anthropic
concurrent connection rate limits. GPT-5.4 is the only API without these
limits and runs with Modal (see 05_main_experiment_modal.py for GPT-5.4 only).

This script handles:
  A: claude-sonnet-4-6  zero-shot          — dense baseline
  B: claude-sonnet-4-6  extended thinking  — reasoning enabled (same model)
     A↔B is the primary within-model ablation for the Peng theorem test.

Primary outcome: cross-model agreement-when-wrong by hop count
  (computed post-hoc in 07_analysis.py by joining with GPT-5.4 results)

Secondary: binary accuracy, error type, thinking tokens, EHR completeness

Run:
  ANTHROPIC_API_KEY=... python3 experiments/06_main_experiment_local.py
"""

import json, time, xml.etree.ElementTree as ET, os
from pathlib import Path
import pandas as pd, numpy as np
import anthropic, openai
from anthropic import RateLimitError as AnthropicRateLimitError

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
             "/medalign_instructions_v1_3")
ANNOT = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
             "/results/hop_annotations.csv")
OUT   = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results")

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

# ── Pre-registered judge prompt ───────────────────────────────────────────────
JUDGE_SYS = """You are a clinical QA evaluator. Score model responses against clinician reference answers.
Return ONLY valid JSON — no other text:
{"correct": true or false, "error_type": "none" | "omission" | "hallucination" | "reasoning_error", "confidence": 0.0-1.0}
Definitions:
- correct: model response substantially agrees with the clinician reference
- omission: model failed to answer, refused, gave empty/irrelevant response
- hallucination: model stated facts not in the EHR or contradicts medical knowledge
- reasoning_error: model had the right facts but reached the wrong conclusion
- confidence: your confidence in this scoring (0=unsure, 1=certain)"""


# ── EHR parsing ───────────────────────────────────────────────────────────────
def extract_ehr(xml_path: Path, max_chars: int = 8000) -> tuple[str, int]:
    """Returns (truncated_ehr_text, full_char_count). Full count is completeness signal."""
    try:
        tree = ET.parse(xml_path)
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


# ── Generators ────────────────────────────────────────────────────────────────
def _call_claude_with_retry(kwargs: dict, max_retries: int = 6) -> object:
    """Call anth_client.messages.create with exponential backoff on 429."""
    delay = 30  # start at 30s — well above TCP keep-alive window
    for attempt in range(max_retries):
        try:
            return anth_client.messages.create(**kwargs)
        except AnthropicRateLimitError as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [429 rate-limit, sleeping {delay}s before retry {attempt+1}/{max_retries}]")
            time.sleep(delay)
            delay = min(delay * 2, 300)  # cap at 5 min
        except Exception:
            raise


def claude_zeroshot(question: str, ehr: str) -> tuple[str, int]:
    r = _call_claude_with_retry(dict(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content":
            f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."}]
    ))
    return r.content[0].text.strip(), 0


def claude_extended_thinking(question: str, ehr: str) -> tuple[str, int]:
    """Same model, reasoning enabled. Primary within-model ablation."""
    r = _call_claude_with_retry(dict(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 3000},
        messages=[{"role": "user", "content":
            f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."}]
    ))
    think_tokens = sum(
        len(b.thinking.split()) * 1.3
        for b in r.content if hasattr(b, "thinking") and b.type == "thinking"
    )
    text_blocks = [b for b in r.content if b.type == "text"]
    text = text_blocks[0].text.strip() if text_blocks else ""
    return text, int(think_tokens)


# ── Judge ─────────────────────────────────────────────────────────────────────
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


def infer_domain(specialty: str) -> str:
    s = str(specialty).lower()
    for domain, keywords in {
        "cardiology":  ["cardiol", "heart", "cardiac"],
        "nephrology":  ["nephrol", "renal", "kidney"],
        "oncology":    ["oncol", "cancer", "tumor"],
        "pulmonary":   ["pulmon", "lung", "respir"],
        "neurology":   ["neurol", "neuro"],
        "endocrine":   ["endocrin", "diabet", "thyroid"],
        "primary_care":["primary", "general", "internal", "family"],
        "surgery":     ["surg"],
        "psychiatry":  ["psych"],
    }.items():
        if any(kw in s for kw in keywords):
            return domain
    return "other"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    annot = pd.read_csv(ANNOT).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)

    df_resp = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map = (df_resp.drop_duplicates("instruction_id")
               [["instruction_id", "filename", "clinician_response"]]
               .set_index("instruction_id"))

    sample = annot[annot["instruction_id"].isin(ref_map.index)].copy()
    print(f"Questions: {len(sample)}")
    print("Hop distribution:", sample["hop_count"].value_counts().sort_index().to_dict())

    raw_path = OUT / "claude_experiment_raw.csv"
    if raw_path.exists():
        done_df  = pd.read_csv(raw_path)
        done_key = set(zip(done_df["instruction_id"], done_df["condition"]))
        records  = done_df.to_dict("records")
        print(f"Resuming from {len(done_df)} existing rows")
    else:
        done_key, records = set(), []

    conditions = {
        "claude_zeroshot":          claude_zeroshot,
        "claude_extended_thinking": claude_extended_thinking,
    }

    total = len(sample) * len(conditions)
    n_done = len(records)

    for _, row in sample.iterrows():
        iid      = row["instruction_id"]
        question = row["question"]
        hop      = int(row["hop_count"])
        fn_      = str(ref_map.loc[iid, "filename"])
        cr       = str(ref_map.loc[iid, "clinician_response"])
        ehr, ehr_chars = extract_ehr(BASE / "ehrs" / fn_)
        domain   = infer_domain(row.get("submitter_specialty", ""))
        q_tokens = len(question.split())

        for cond, gen_fn in conditions.items():
            if (iid, cond) in done_key:
                n_done += 1
                continue
            try:
                resp, think_tokens = gen_fn(question, ehr)
                sc = judge(question, cr, resp)
                records.append({
                    "instruction_id":       iid,
                    "condition":            cond,
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
                mark = "✓" if sc.get("correct") else "✗"
                n_done += 1
                print(f"[{n_done:3d}/{total}] {cond:28s} {mark} "
                      f"hop={hop} {sc.get('error_type','?'):18s}: {question[:40]}")
            except Exception as e:
                print(f"  ERROR {cond}: {e}")
                n_done += 1
            time.sleep(8)   # space calls to avoid concurrent connection 429s

        # Checkpoint every 20 questions
        if n_done % 40 == 0 and records:
            pd.DataFrame(records).to_csv(raw_path, index=False)
            print(f"  [checkpoint: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df_out)} rows → {raw_path}")

    agg = df_out.groupby(["condition", "hop_count"]).agg(
        n                    = ("correct", "count"),
        accuracy             = ("correct", "mean"),
        omission_rate        = ("error_type", lambda x: (x == "omission").mean()),
        hallucination_rate   = ("error_type", lambda x: (x == "hallucination").mean()),
        reasoning_error_rate = ("error_type", lambda x: (x == "reasoning_error").mean()),
        mean_thinking_tokens = ("thinking_tokens", "mean"),
    ).reset_index()

    agg.to_csv(OUT / "claude_experiment_scores.csv", index=False)
    print("\n=== Claude results ===")
    print(agg.to_string(index=False))

    print("\n=== CoT benefit by hop (primary interaction test) ===")
    zs = agg[agg["condition"] == "claude_zeroshot"].set_index("hop_count")["accuracy"]
    et = agg[agg["condition"] == "claude_extended_thinking"].set_index("hop_count")["accuracy"]
    benefit = (et - zs).dropna()
    print("Extended thinking - zero-shot (should increase with hop):")
    print(benefit.round(3))

    print("\n=== Thinking token scaling (Peng mechanistic test) ===")
    tt = df_out[df_out["condition"] == "claude_extended_thinking"].groupby("hop_count")[
        "thinking_tokens"].mean()
    print(tt.round(0))


if __name__ == "__main__":
    main()
