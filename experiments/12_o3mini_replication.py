"""
Experiment 12: o3-mini Replication Study — P2 Test (Accuracy-Depth Curve)
==========================================================================
Evaluates OpenAI o3-mini (RL-based reasoning model) on the same 301 clinical
EHR questions used in the main experiment to test whether reasoning-specialized
models (o1/DeepSeek-R1 class) flatten the accuracy-depth curve.

Directly addresses reviewer Q7: do RL-trained reasoning models escape the
accuracy ceiling at higher hop counts?

Model fallback order:
  1. o3-mini  (reasoning_effort="high")
  2. o1-mini  (reasoning_effort="high" if supported)
  3. gpt-4o-mini  (standard, no reasoning_effort)

Run:
  python3 experiments/12_o3mini_replication.py
"""

import json, time, xml.etree.ElementTree as ET, os
from pathlib import Path

import pandas as pd
import numpy as np
import openai
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
                "/medalign_instructions_v1_3")
CLAUDE   = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                "/results/claude_experiment_raw.csv")
OUT_CSV  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                "/results/o3mini_zeroshot_raw.csv")

# ── API key ──────────────────────────────────────────────────────────────────
ENV_PATH = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"

def load_key(env_path: str, key: str) -> str:
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    except FileNotFoundError:
        pass
    return ""

OPENAI_KEY = (os.environ.get("OPENAI_API_KEY", "")
              or load_key(ENV_PATH, "OPENAI_API_KEY")
              or load_key("/Users/sanjaybasu/waymark-local/.env", "OPENAI_API_KEY"))

client = openai.OpenAI(api_key=OPENAI_KEY)

# ── Judge prompt (pre-registered, same as main experiment) ────────────────────
JUDGE_PROMPT = (
    "You are a clinical QA evaluator. Score model responses against clinician "
    "reference answers. Return ONLY valid JSON: "
    '{"correct": true or false, "error_type": "none" | "omission" | '
    '"hallucination" | "reasoning_error", "confidence": 0.0-1.0}. '
    "Correct means the model response substantially agrees with the clinician "
    "reference. Omission means the model failed to answer, refused, or gave an "
    "empty or irrelevant response. Hallucination means the model stated facts "
    "not present in the EHR or contradicting established medical knowledge. "
    "Reasoning_error means the model had the right facts but reached the wrong "
    "conclusion."
)

# ── EHR parsing (spec-defined, same as main experiment) ──────────────────────
def extract_ehr(xml_path: Path, max_chars: int = 8000) -> tuple[str, int]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                lines.append(f"{tag}: {elem.text.strip()}")
        text = "\n".join(lines)
        return text[:max_chars], len(text)
    except Exception as e:
        return f"[EHR error: {e}]", 0


# ── Model probe: detect which model is available ──────────────────────────────
def probe_model() -> tuple[str, bool]:
    """
    Try models in priority order.
    Returns (model_name, supports_reasoning_effort).
    """
    candidates = [
        ("o3-mini",      True),
        ("o1-mini",      True),
        ("gpt-4o-mini",  False),
    ]
    for model, has_reasoning in candidates:
        try:
            kwargs = dict(
                model=model,
                max_completion_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
            if has_reasoning:
                kwargs["reasoning_effort"] = "high"
            client.chat.completions.create(**kwargs)
            print(f"[probe] Model available: {model} (reasoning_effort={has_reasoning})")
            return model, has_reasoning
        except openai.NotFoundError:
            print(f"[probe] {model} not found, trying next...")
        except openai.BadRequestError as e:
            # Model exists but rejects reasoning_effort; retry without it
            if has_reasoning:
                try:
                    client.chat.completions.create(
                        model=model,
                        max_completion_tokens=8,
                        messages=[{"role": "user", "content": "hi"}],
                    )
                    print(f"[probe] {model} available WITHOUT reasoning_effort")
                    return model, False
                except Exception:
                    print(f"[probe] {model} failed: {e}")
            else:
                print(f"[probe] {model} failed: {e}")
        except Exception as e:
            print(f"[probe] {model} error: {e}")
    raise RuntimeError("No usable OpenAI model found.")


# ── Inference call ────────────────────────────────────────────────────────────
def call_model(question: str, ehr: str, model: str,
               supports_reasoning: bool, max_retries: int = 5) -> str:
    kwargs = dict(
        model=model,
        max_completion_tokens=4096,  # o3-mini uses most tokens for internal reasoning; 4096 ensures answer fits even for complex hop=4 questions
        messages=[{
            "role": "user",
            "content": (f"Patient EHR:\n{ehr}\n\n"
                        f"Question: {question}\n\n"
                        "Answer concisely.")
        }],
    )
    if supports_reasoning:
        kwargs["reasoning_effort"] = "high"

    delay = 5
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [rate-limit, sleeping {delay}s]")
            time.sleep(delay)
            delay = min(delay * 2, 120)
        except Exception:
            raise
    return ""


# ── Judge ─────────────────────────────────────────────────────────────────────
def judge(question: str, ref: str, response: str,
          max_retries: int = 3) -> dict:
    delay = 3
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=120,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user",   "content":
                        f"Question: {question}\n"
                        f"Clinician reference: {ref}\n"
                        f"Model response: {response}"},
                ]
            )
            raw = r.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                return {"correct": False, "error_type": "error", "confidence": 0.0}
            time.sleep(delay)
            delay *= 2
        except Exception:
            return {"correct": False, "error_type": "error", "confidence": 0.0}
    return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Confidence interval helper ────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load source questions (unique ZS rows from main experiment)
    claude_df = pd.read_csv(CLAUDE)
    questions_df = (claude_df[claude_df["condition"] == "claude_zeroshot"]
                    .drop_duplicates("instruction_id")
                    .copy())
    print(f"Source questions: {len(questions_df)} unique instruction_ids")
    print("Hop distribution:", questions_df["hop_count"].value_counts().sort_index().to_dict())

    # 2. Build filename + clinician_ref lookup from TSV
    tsv = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map = (tsv.drop_duplicates("instruction_id")
               [["instruction_id", "filename", "clinician_response"]]
               .set_index("instruction_id"))

    # 3. Probe available model
    model_name, supports_reasoning = probe_model()
    print(f"\nUsing model: {model_name}  reasoning_effort={'high' if supports_reasoning else 'N/A'}\n")

    # 4. Resume logic
    if OUT_CSV.exists():
        done_df  = pd.read_csv(OUT_CSV)
        done_ids = set(done_df["instruction_id"].tolist())
        records  = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} already complete")
    else:
        done_ids, records = set(), []

    total = len(questions_df)

    # 5. Main loop
    for idx, (_, row) in enumerate(questions_df.iterrows(), start=1):
        iid      = int(row["instruction_id"])
        hop      = int(row["hop_count"])
        question = str(row["question"])
        cr_from_claude = str(row["clinician_ref"])

        if iid in done_ids:
            continue

        # Clinician reference: prefer TSV (authoritative), fall back to claude csv
        if iid in ref_map.index:
            fn_      = str(ref_map.loc[iid, "filename"])
            cr       = str(ref_map.loc[iid, "clinician_response"])
        else:
            fn_      = None
            cr       = cr_from_claude

        # Load EHR
        if fn_:
            ehr, _ = extract_ehr(BASE / "ehrs" / fn_)
        else:
            ehr = "[EHR not found]"

        # Model call
        try:
            response = call_model(question, ehr, model_name, supports_reasoning)
        except Exception as e:
            print(f"  ERROR calling model for {iid}: {e}")
            response = ""

        # Judge
        sc = judge(question, cr, response)

        records.append({
            "instruction_id":   iid,
            "hop_count":        hop,
            "model_used":       model_name,
            "question":         question,
            "clinician_ref":    cr,
            "model_response":   response,
            "correct":          int(bool(sc.get("correct", False))),
            "error_type":       sc.get("error_type", "error"),
            "judge_confidence": float(sc.get("confidence", 0.0)),
        })

        mark = "OK" if sc.get("correct") else "--"
        print(f"[{idx:3d}/{total}] {model_name}  {mark}  hop={hop}  "
              f"{sc.get('error_type','?'):18s}  {question[:50]}")

        done_ids.add(iid)

        # Checkpoint every 25 questions
        if len(records) % 25 == 0:
            pd.DataFrame(records).to_csv(OUT_CSV, index=False)
            print(f"  [checkpoint: {len(records)}/{total}]")

        time.sleep(1)

    # 6. Save final
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT_CSV}")

    # 7. Results summary
    HOP_LABELS = {1: 111, 2: 46, 3: 42, 4: 104}

    print(f"\nRESULTS (model: {model_name}):")
    for hop, expected_n in sorted(HOP_LABELS.items()):
        sub = df_out[df_out["hop_count"] == hop]
        n   = len(sub)
        k   = sub["correct"].sum()
        acc = k / n if n > 0 else 0.0
        lo, hi = wilson_ci(k, n)
        print(f"  hop={hop} (n={n}): {acc*100:.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")

    n_all = len(df_out)
    k_all = df_out["correct"].sum()
    acc_all = k_all / n_all if n_all > 0 else 0.0
    lo_all, hi_all = wilson_ci(k_all, n_all)
    print(f"  Overall: {acc_all*100:.1f}%  [{lo_all*100:.1f}, {hi_all*100:.1f}]")

    # Cochran-Armitage trend test across hop counts (P2 test)
    hop_order = sorted(df_out["hop_count"].unique())
    counts = [(df_out[df_out["hop_count"] == h]["correct"].sum(),
               (df_out["hop_count"] == h).sum()) for h in hop_order]

    n_vec  = np.array([c[1] for c in counts], dtype=float)
    k_vec  = np.array([c[0] for c in counts], dtype=float)
    t_vec  = np.array(hop_order, dtype=float)  # trend scores = hop counts

    # CA statistic: T = sum(t_i * k_i) - N_bar * sum(t_i * n_i)
    N  = n_vec.sum()
    K  = k_vec.sum()
    p0 = K / N
    num = (t_vec * k_vec).sum() - (K / N) * (t_vec * n_vec).sum()
    denom = np.sqrt(p0 * (1 - p0) * (
        (t_vec**2 * n_vec).sum() - ((t_vec * n_vec).sum())**2 / N
    ))
    z_ca = num / denom if denom > 0 else 0.0
    p_ca = 2 * (1 - stats.norm.cdf(abs(z_ca)))

    print(f"  CA z={z_ca:.2f}, p={p_ca:.3f}")

    # Compare against main experiment (claude_zeroshot) for context
    zs_agg = (claude_df[claude_df["condition"] == "claude_zeroshot"]
              .groupby("hop_count")["correct"].mean() * 100)
    o3_agg = df_out.groupby("hop_count")["correct"].mean() * 100
    print(f"\nComparison vs. claude_zeroshot (same 301 questions):")
    print(f"  {'hop':>4}  {'claude_zs':>10}  {'o3-mini':>10}  {'delta':>8}")
    for h in sorted(hop_order):
        zs_acc = zs_agg.get(h, float("nan"))
        o3_acc = o3_agg.get(h, float("nan"))
        delta  = o3_acc - zs_acc
        print(f"  {h:>4}  {zs_acc:>9.1f}%  {o3_acc:>9.1f}%  {delta:>+7.1f}pp")


if __name__ == "__main__":
    main()
