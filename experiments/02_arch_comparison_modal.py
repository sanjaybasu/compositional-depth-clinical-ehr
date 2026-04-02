"""
Experiment 2 (Modal): Architectural Comparison Across Hop-Count Classes
========================================================================
Parallelised on Modal — data passed as JSON payloads (no file mounts).
Runs 200 questions × 3 conditions (zero-shot / CoT / RAG) concurrently.

Run with:  modal run experiments/02_arch_comparison_modal.py
"""

import json, time, xml.etree.ElementTree as ET
from pathlib import Path
import modal, numpy as np, pandas as pd

app   = modal.App("clinical-arch-comparison-v2")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("anthropic>=0.49.0", "numpy>=1.26", "pandas>=2.0"))

MODEL     = "claude-sonnet-4-6"
N_PER_HOP = 50
SEED      = 42
OUT_DIR   = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results")

MEDAL_BASE = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
                  "/medalign_instructions_v1_3")
ANNOT_PATH = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                  "/results/hop_annotations.csv")

# ── EHR parsing (runs locally before dispatch) ────────────────────────────────

def extract_ehr_text(xml_path: Path, max_chars: int = 8000) -> str:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        segs = []
        for enc in root.findall("encounter"):
            for entry in enc.findall(".//entry"):
                ts = entry.get("timestamp", "")
                for ev in entry.findall("event"):
                    text = (ev.text or "").strip()
                    if text:
                        segs.append(f"[{ts}] {ev.get('type','')}: {text}")
        return "\n".join(segs)[:max_chars]
    except Exception as e:
        return f"[EHR error: {e}]"


# ── Remote function: evaluate one question under all 3 conditions ─────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    timeout=300,
    retries=2,
)
def evaluate_question(payload: str) -> str:
    """payload: JSON with {instruction_id, question, hop_count, requires_ehr,
                           clinician_response, ehr_text}"""
    import anthropic, json as _json
    record   = _json.loads(payload)
    client   = anthropic.Anthropic()
    question = record["question"]
    ehr_text = record["ehr_text"]
    clinician= record["clinician_response"]
    iid      = record["instruction_id"]
    hop      = record["hop_count"]
    req_ehr  = record["requires_ehr"]

    def call(msgs, system=None, max_tokens=512):
        kw = dict(model=MODEL, max_tokens=max_tokens, messages=msgs)
        if system: kw["system"] = system
        return client.messages.create(**kw).content[0].text.strip()

    def zero_shot():
        return call([{"role":"user","content":
            f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\nAnswer concisely."}])

    def cot():
        return call([{"role":"user","content":
            f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\n"
            "Think step by step.\n"
            "Step 1 - Identify needed information:\n"
            "Step 2 - Retrieve from EHR:\n"
            "Step 3 - Compose and reason:\n"
            "Final Answer:"}], max_tokens=1024)

    def rag():
        retrieved = call([{"role":"user","content":
            f"EHR:\n{ehr_text}\n\nQuestion: {question}\n\n"
            "Extract only verbatim EHR passages relevant to this question. "
            "If nothing relevant, say 'NO RELEVANT EHR DATA'."}])
        return call([{"role":"user","content":
            f"Relevant EHR passages:\n{retrieved}\n\nQuestion: {question}\n\n"
            "Answer using only the above passages. "
            "State explicitly if answer cannot be determined."}])

    JUDGE = """Assess correctness. Return ONLY valid JSON:
{"correctness":"correct|partial|incorrect|omission",
 "error_type":"none|omission|hallucination|reasoning_error",
 "reasoning":"one sentence"}"""

    def score(response):
        raw = call([{"role":"user","content":
            f"Question: {question}\nClinician: {clinician}\nModel: {response}"}],
            system=JUDGE, max_tokens=200)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        try:
            return _json.loads(raw.strip())
        except Exception:
            return {"correctness":"error","error_type":"error","reasoning":raw[:100]}

    generators = {"zero_shot": zero_shot, "cot": cot, "rag": rag}
    results = []
    for cond, fn in generators.items():
        try:
            resp  = fn()
            sc    = score(resp)
            binary = 1 if sc["correctness"] == "correct" else 0
        except Exception as e:
            resp   = f"ERROR: {e}"
            sc     = {"correctness":"error","error_type":"error","reasoning":str(e)}
            binary = 0
        results.append({
            "instruction_id":  iid,
            "condition":       cond,
            "hop_count":       hop,
            "requires_ehr":    req_ehr,
            "question":        question,
            "clinician_ref":   clinician,
            "model_response":  resp,
            "correctness":     sc["correctness"],
            "error_type":      sc["error_type"],
            "binary_correct":  binary,
            "score_reasoning": sc.get("reasoning",""),
        })
        time.sleep(0.2)

    return _json.dumps(results)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    annot = pd.read_csv(ANNOT_PATH).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)

    df_resp = pd.read_csv(MEDAL_BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map = (df_resp.drop_duplicates("instruction_id")
               [["instruction_id","filename","clinician_response"]]
               .set_index("instruction_id"))

    rng = np.random.default_rng(SEED)
    sample_ids = []
    for hop in [1, 2, 3, 4]:
        pool  = annot[annot["hop_count"] == hop]["instruction_id"].tolist()
        n     = min(N_PER_HOP, len(pool))
        chosen = rng.choice(pool, size=n, replace=False).tolist()
        sample_ids.extend(chosen)
        print(f"  hop={hop}: {n} sampled ({len(pool)} available)")

    sample = annot[annot["instruction_id"].isin(sample_ids)].copy()
    sample = sample[sample["instruction_id"].isin(ref_map.index)]

    # Build payloads (serialize EHR text locally — no mount needed)
    payloads = []
    for _, row in sample.iterrows():
        iid      = row["instruction_id"]
        filename = str(ref_map.loc[iid, "filename"])
        ehr_text = extract_ehr_text(MEDAL_BASE / "ehrs" / filename)
        payloads.append(json.dumps({
            "instruction_id":    iid,
            "question":          row["question"],
            "hop_count":         int(row["hop_count"]),
            "requires_ehr":      bool(row.get("requires_ehr", True)),
            "clinician_response": str(ref_map.loc[iid, "clinician_response"]),
            "ehr_text":          ehr_text,
        }))

    print(f"\nDispatching {len(payloads)} questions to Modal...")
    all_rows, n_done = [], 0
    for raw_out in evaluate_question.map(payloads, order_outputs=False):
        rows = json.loads(raw_out)
        all_rows.extend(rows)
        n_done += 1
        iid = rows[0]["instruction_id"] if rows else "?"
        print(f"  [{n_done:3d}/{len(payloads)}] {iid} hop={rows[0]['hop_count'] if rows else '?'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    raw_path   = OUT_DIR / "arch_comparison_raw.csv"
    score_path = OUT_DIR / "arch_comparison_scores.csv"
    df_out.to_csv(raw_path, index=False)

    agg = df_out.groupby(["condition","hop_count"]).agg(
        n=("binary_correct","count"),
        accuracy=("binary_correct","mean"),
        omission_rate=("error_type", lambda x: (x=="omission").mean()),
        hallucination_rate=("error_type", lambda x: (x=="hallucination").mean()),
        reasoning_error_rate=("error_type", lambda x: (x=="reasoning_error").mean()),
    ).reset_index()
    agg.to_csv(score_path, index=False)
    print(f"\n=== Results saved ===")
    print(f"Raw:   {raw_path}")
    print(f"Agg:   {score_path}")
    print(agg.to_string(index=False))
