"""
Experiment 4: SOTA Model Comparison — Testing the Peng Theorem (2402.08164)
===========================================================================
Tests whether compositional depth (hop count) predicts accuracy degradation in SOTA 2026 LLMs,
and whether test-time compute (reasoning/CoT) mitigates this scaling as Peng predicts.

Core hypothesis (Peng 2402.08164):
  - Single-layer attention cannot compose functions when n log n > H·d·p
  - Error rate should scale monotonically with hop count for standard models
  - CoT/reasoning models should show FLATTER scaling (lower slope on hop→error curve)

Models and conditions:
  A) gpt-5.4 (zero-shot)       — latest OpenAI dense model, no CoT
  B) gpt-5.4 (CoT prompt)      — same model, explicit step-by-step reasoning (within-model ablation)
  C) Claude claude-sonnet-4-6 (zero-shot)  — Anthropic SOTA dense transformer
  D) Claude claude-sonnet-4-6 (CoT)        — same model, explicit CoT (within-model ablation)
  E) DeepSeek R1 (no thinking) — open-source reasoning model, thinking stripped
  F) DeepSeek R1 (with thinking)— same model, reasoning tokens enabled (within-model ablation)

Design rationale:
  - A vs B: Does CoT prompt help GPT-5.4? By how much at each hop level?
  - C vs D: Does CoT prompt help Claude? By how much?
  - E vs F: Does DeepSeek R1's internal chain-of-thought help? Cleanest ablation (same model, same weights)
  - The E/F comparison is methodologically strongest: isolates reasoning from knowledge
  - All three comparisons should show widening CoT benefit with hop count (Peng prediction)

DeepSeek R1 via Ollama (local):
  - Requires: ollama pull deepseek-r1:8b  (or 70b if available)
  - R1 naturally emits <think>...</think> blocks before the answer
  - "no thinking" condition: strip <think> block, return only final answer
  - "with thinking" condition: let model reason fully
  - Set OLLAMA_AVAILABLE=True once confirmed running

Dataset:
  MedAlign n=238 questions (annotated by hop count 1–4)
  Stratified sample: 20 per hop class × 4 = 80 questions × 6 conditions = 480 evaluations

Output:
  results/sota_comparison_raw.csv
  results/sota_comparison_scores.csv
"""

import json, time, xml.etree.ElementTree as ET, os, re
from pathlib import Path
import pandas as pd, numpy as np
import anthropic, openai

# ── Config ─────────────────────────────────────────────────────────────────
BASE  = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files/medalign_instructions_v1_3")
ANNOT = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/hop_annotations.csv")
OUT   = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results")

# Load OpenAI key from .env
def load_key(env_path: str, key: str) -> str:
    with open(env_path) as f:
        for line in f:
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""

ENV = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"
OPENAI_KEY    = load_key(ENV, "OPENAI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
openai_client    = openai.OpenAI(api_key=OPENAI_KEY)

# DeepSeek R1 via Ollama (local inference)
# Set to True if Ollama is running with deepseek-r1 pulled
OLLAMA_AVAILABLE = False
OLLAMA_MODEL     = "deepseek-r1:8b"   # or deepseek-r1:70b if available
OLLAMA_URL       = "http://localhost:11434/api/generate"

N_PER_HOP = 20   # 20 × 4 hops = 80 questions × 6 conditions = 480 evaluations
SEED      = 42

JUDGE_SYS = """You are a clinical QA evaluator. Return ONLY valid JSON:
{"correct": true|false, "error_type": "none|omission|hallucination|reasoning_error"}
Definitions: correct=substantially matches reference; omission=model failed/refused/empty;
hallucination=stated something not in EHR or medically incorrect; reasoning_error=right facts wrong conclusion"""


# ── Generators ────────────────────────────────────────────────────────────────

def gpt54_zeroshot(question: str, ehr: str) -> str:
    r = openai_client.chat.completions.create(
        model="gpt-5.4",
        max_completion_tokens=512,
        messages=[{"role":"user","content": f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."}]
    )
    return r.choices[0].message.content.strip()


def gpt54_cot(question: str, ehr: str) -> str:
    prompt = f"""Patient EHR:
{ehr}

Question: {question}

Think step by step before answering:
Step 1 - What information do I need to answer this?
Step 2 - Where in the EHR is this information?
Step 3 - How do I compose these pieces to answer the question?
Final Answer:"""
    r = openai_client.chat.completions.create(
        model="gpt-5.4",
        max_completion_tokens=1024,
        messages=[{"role":"user","content": prompt}]
    )
    return r.choices[0].message.content.strip()


def claude_zeroshot(question: str, ehr: str) -> str:
    r = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role":"user","content": f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."}]
    )
    return r.content[0].text.strip()


def claude_cot(question: str, ehr: str) -> str:
    prompt = f"""Patient EHR:
{ehr}

Question: {question}

Think step by step before answering:
Step 1 - What information do I need to answer this?
Step 2 - Where in the EHR is this information?
Step 3 - How do I compose these pieces to answer the question?
Final Answer:"""
    r = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role":"user","content": prompt}]
    )
    return r.content[0].text.strip()


def deepseek_ollama(question: str, ehr: str, use_thinking: bool) -> str:
    """DeepSeek R1 via Ollama. use_thinking=False strips <think> blocks."""
    import urllib.request
    prompt = f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1024}
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())["response"]
    if not use_thinking:
        # Strip <think>...</think> block emitted by R1
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


def deepseek_nothink(question: str, ehr: str) -> str:
    return deepseek_ollama(question, ehr, use_thinking=False)


def deepseek_think(question: str, ehr: str) -> str:
    return deepseek_ollama(question, ehr, use_thinking=True)


def judge_openai(question: str, ref: str, response: str) -> dict:
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",   # cheap judge; not being evaluated
        max_tokens=100,
        messages=[
            {"role":"system","content": JUDGE_SYS},
            {"role":"user","content": f"Q: {question}\nRef: {ref}\nResponse: {response}"}
        ]
    )
    raw = r.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except:
        return {"correct": False, "error_type": "error"}


def extract_ehr(xml_path: Path, max_chars: int = 7000) -> str:
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
        return "\n".join(segs)[:max_chars]
    except Exception as e:
        return f"[EHR error: {e}]"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    annot  = pd.read_csv(ANNOT).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)

    df_resp = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map = (df_resp.drop_duplicates("instruction_id")
               [["instruction_id","filename","clinician_response"]]
               .set_index("instruction_id"))

    rng = np.random.default_rng(SEED)
    sample_ids = []
    for hop in [1, 2, 3, 4]:
        pool  = annot[annot["hop_count"] == hop]["instruction_id"].tolist()
        pool  = [p for p in pool if p in ref_map.index]
        n     = min(N_PER_HOP, len(pool))
        chosen = rng.choice(pool, size=n, replace=False).tolist()
        sample_ids.extend(chosen)
        print(f"  hop={hop}: {n} sampled from {len(pool)} available")

    sample = annot[annot["instruction_id"].isin(sample_ids)].copy()
    print(f"\nTotal: {len(sample)} questions")

    # Resume support
    raw_path = OUT / "sota_comparison_raw.csv"
    if raw_path.exists():
        done_df  = pd.read_csv(raw_path)
        done_key = set(zip(done_df["instruction_id"], done_df["condition"]))
        records  = done_df.to_dict("records")
        print(f"Resuming from {len(done_df)} existing rows")
    else:
        done_key, records = set(), []

    # Build condition dict — skip DeepSeek if Ollama not available
    conditions = {
        "gpt54_zeroshot":  lambda q,e: (gpt54_zeroshot(q,e), 0),
        "gpt54_cot":       lambda q,e: (gpt54_cot(q,e), 0),
        "claude_zeroshot": lambda q,e: (claude_zeroshot(q,e), 0),
        "claude_cot":      lambda q,e: (claude_cot(q,e), 0),
    }
    if OLLAMA_AVAILABLE:
        conditions["deepseek_nothink"] = lambda q,e: (deepseek_nothink(q,e), 0)
        conditions["deepseek_think"]   = lambda q,e: (deepseek_think(q,e), 0)
        print("DeepSeek R1 via Ollama: ENABLED")
    else:
        print("DeepSeek R1 via Ollama: DISABLED (set OLLAMA_AVAILABLE=True to enable)")

    total  = len(sample) * len(conditions)
    n_done = len(records)

    for _, row in sample.iterrows():
        iid      = row["instruction_id"]
        question = row["question"]
        hop      = int(row["hop_count"])
        fn       = str(ref_map.loc[iid,"filename"])
        cr       = str(ref_map.loc[iid,"clinician_response"])
        ehr_txt  = extract_ehr(BASE / "ehrs" / fn)

        for cond, gen_fn in conditions.items():
            if (iid, cond) in done_key:
                n_done += 1
                continue
            try:
                resp, think_tokens = gen_fn(question, ehr_txt)
                sc = judge_openai(question, cr, resp)
                records.append({
                    "instruction_id":  iid,
                    "condition":       cond,
                    "hop_count":       hop,
                    "question":        question,
                    "clinician_ref":   cr,
                    "model_response":  resp,
                    "correct":         int(sc["correct"]),
                    "error_type":      sc["error_type"],
                    "thinking_tokens": think_tokens,
                })
                mark = "✓" if sc["correct"] else "✗"
                n_done += 1
                print(f"[{n_done:3d}/{total}] {cond:22s} {mark} hop={hop} {sc['error_type']:18s}: {question[:45]}")
            except Exception as e:
                print(f"  ERROR {cond}: {e}")
                n_done += 1
            time.sleep(1.0)

        if n_done % 40 == 0 and records:
            pd.DataFrame(records).to_csv(raw_path, index=False)
            print(f"  [checkpoint: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df_out)} rows → {raw_path}")

    # Aggregate
    agg = df_out.groupby(["condition","hop_count"]).agg(
        n=("correct","count"),
        accuracy=("correct","mean"),
        omission_rate=("error_type", lambda x: (x=="omission").mean()),
        hallucination_rate=("error_type", lambda x: (x=="hallucination").mean()),
        reasoning_error_rate=("error_type", lambda x: (x=="reasoning_error").mean()),
        mean_thinking_tokens=("thinking_tokens","mean"),
    ).reset_index()

    agg.to_csv(OUT / "sota_comparison_scores.csv", index=False)
    print(f"\n=== SOTA Results ===")
    print(agg.to_string(index=False))

    # CoT benefit by hop count (the key Peng theorem test)
    print("\n--- CoT benefit by hop count (Peng prediction: should grow with hop) ---")
    for model_base, zs_cond, cot_cond in [
        ("GPT-5.4",  "gpt54_zeroshot",  "gpt54_cot"),
        ("Claude",   "claude_zeroshot", "claude_cot"),
    ]:
        zs  = agg[agg["condition"]==zs_cond].set_index("hop_count")["accuracy"]
        cot = agg[agg["condition"]==cot_cond].set_index("hop_count")["accuracy"]
        print(f"\n{model_base} CoT benefit (CoT - ZeroShot):")
        print((cot - zs).round(3))

    if OLLAMA_AVAILABLE:
        nothink = agg[agg["condition"]=="deepseek_nothink"].set_index("hop_count")["accuracy"]
        think   = agg[agg["condition"]=="deepseek_think"].set_index("hop_count")["accuracy"]
        print("\nDeepSeek R1 thinking benefit (think - nothink):")
        print((think - nothink).round(3))


if __name__ == "__main__":
    main()
