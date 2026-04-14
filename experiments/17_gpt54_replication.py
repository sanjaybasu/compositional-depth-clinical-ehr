"""
Experiment 17: GPT-5.4 Zero-Shot Cross-Architecture Replication
===============================================================
Replicates the main P1 finding using gpt-5.4-2026-03-05, the current
OpenAI flagship model (March 2026), updating the cross-architecture
replication column in Table 1 beyond GPT-4o (2024).

Identical protocol to the GPT-4o replication (Exp 02/main experiment):
  - Zero-shot, 8000-char EHR truncation
  - Same 301 questions, same judge (gpt-4o-mini), same EHR context

Adds a third architecture to Table 1, strengthening the claim that
hop-count-predicted failure is a current property of the best available
models, not an artifact of 2024-era LLMs.

Run:
  python3 experiments/17_gpt54_replication.py
"""

import json, time, xml.etree.ElementTree as ET, os, math
from pathlib import Path

import pandas as pd
import numpy as np
import openai
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
               "/medalign_instructions_v1_3")
CLAUDE  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
               "/results/claude_experiment_raw.csv")
OUT     = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
               "/results/gpt54_zeroshot_raw.csv")

MODEL   = "gpt-5.4-2026-03-05"   # pinned for reproducibility

# ── API key ────────────────────────────────────────────────────────────────────
_ENV = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"

def _load_key(env_path, key):
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return ""

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "") or _load_key(_ENV, "OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_KEY)

# ── Judge prompt (pre-registered, identical to all other experiments) ──────────
JUDGE_SYS = (
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

# ── EHR parsing (identical to main experiment) ────────────────────────────────
def _extract_ehr(xml_path: Path, max_chars: int = 8000) -> tuple[str, int]:
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


# ── Inference ──────────────────────────────────────────────────────────────────
def _call_gpt54(question: str, ehr: str, max_retries: int = 6) -> str:
    prompt = f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."
    delay = 10
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [rate-limit, sleeping {delay}s]")
            time.sleep(delay)
            delay = min(delay * 2, 180)
        except Exception:
            raise
    return ""


# ── Judge ──────────────────────────────────────────────────────────────────────
def _judge(question: str, ref: str, response: str, max_retries: int = 3) -> dict:
    delay = 3
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=120,
                messages=[
                    {"role": "system", "content": JUDGE_SYS},
                    {"role": "user",   "content":
                     f"Question: {question}\nClinician reference: {ref}\n"
                     f"Model response: {response}"},
                ],
            )
            raw = r.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay); delay = min(delay * 2, 60)
        except Exception as e:
            return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Statistics ─────────────────────────────────────────────────────────────────
def _wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; denom = 1 + z**2 / n
    ctr  = (p + z**2 / (2 * n)) / denom
    marg = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, ctr - marg), min(1.0, ctr + marg)

def _cochran_armitage(ks, ns):
    """One-tailed CA trend test (decline direction)."""
    ks, ns = np.array(ks, float), np.array(ns, float)
    scores = np.arange(1, len(ks) + 1, dtype=float)
    N = ns.sum(); pbar = ks.sum() / N
    S = (scores * (ks - ns * pbar)).sum()
    V = pbar * (1 - pbar) * (N * (scores**2 * ns).sum()
                              - ((scores * ns).sum())**2) / N
    if V <= 0: return float("nan"), float("nan")
    z = S / math.sqrt(V)
    return z, stats.norm.cdf(z)   # left tail for decline


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load question sample from Claude baseline (same 301 questions)
    df_base = (pd.read_csv(CLAUDE)
               [lambda d: d["condition"] == "claude_zeroshot"]
               .drop_duplicates("instruction_id"))
    sample  = df_base[["instruction_id", "hop_count", "question", "clinician_ref"]].copy()
    sample["instruction_id"] = sample["instruction_id"].astype(int)

    df_tsv   = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    file_map = df_tsv.drop_duplicates("instruction_id").set_index("instruction_id")["filename"]
    sample   = sample[sample["instruction_id"].isin(file_map.index)].copy()

    print(f"Model: {MODEL}")
    print(f"Questions: {len(sample)}")
    print("Hop distribution:", sample["hop_count"].value_counts().sort_index().to_dict())

    # Resume
    if OUT.exists():
        done_df  = pd.read_csv(OUT)
        done_ids = set(done_df["instruction_id"].astype(int))
        records  = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} done")
    else:
        done_ids, records = set(), []

    total = len(sample); n_done = len(records)

    for _, row in sample.iterrows():
        iid = int(row["instruction_id"])
        if iid in done_ids:
            continue

        question = str(row["question"])
        hop      = int(row["hop_count"])
        cr       = str(row["clinician_ref"])
        ehr, ehr_chars = _extract_ehr(BASE / "ehrs" / str(file_map.loc[iid]))

        try:
            resp = _call_gpt54(question, ehr)
            sc   = _judge(question, cr, resp)
            records.append({
                "instruction_id":   iid,
                "condition":        "gpt54_zeroshot",
                "hop_count":        hop,
                "ehr_char_count":   ehr_chars,
                "question":         question,
                "clinician_ref":    cr,
                "model_response":   resp,
                "correct":          int(sc.get("correct", False)),
                "error_type":       sc.get("error_type", "error"),
                "judge_confidence": sc.get("confidence", 0.0),
            })
            mark   = "+" if sc.get("correct") else "-"
            n_done += 1
            print(f"[{n_done:3d}/{total}] GPT-5.4 {mark} hop={hop} "
                  f"{sc.get('error_type','?'):18s}: {question[:50]}")
        except Exception as e:
            print(f"  ERROR iid={iid}: {e}")
            n_done += 1

        time.sleep(0.4)
        if n_done % 25 == 0 and records:
            pd.DataFrame(records).to_csv(OUT, index=False)
            print(f"  [checkpoint: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"GPT-5.4 ZERO-SHOT RESULTS (vs GPT-4o baseline):")
    gpt4o_acc = {1: 0.378, 2: 0.261, 3: 0.238, 4: 0.147}
    ks, ns = [], []
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        n = len(sub); k = int(sub["correct"].sum())
        ks.append(k); ns.append(n)
        lo, hi = _wilson_ci(k, n)
        print(f"  hop={h} (n={n}): {k/n:.1%} [{lo:.1%},{hi:.1%}]  "
              f"GPT-4o baseline: {gpt4o_acc.get(h, 0):.1%}")
    overall = sum(ks) / sum(ns)
    lo_a, hi_a = _wilson_ci(sum(ks), sum(ns))
    print(f"  Overall: {overall:.1%} [{lo_a:.1%},{hi_a:.1%}]")
    ca_z, ca_p = _cochran_armitage(ks, ns)
    print(f"  CA trend (one-tailed): z={ca_z:.3f}, p={ca_p:.4f}")
    if ca_p < 0.05:
        print("  ✓ P1 confirmed — GPT-5.4 shows same monotone decline.")
    else:
        print("  ✗ P1 not significant — investigate.")

    # Error type profile
    print("\n  Error types:")
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        if not len(sub): continue
        et = sub["error_type"].value_counts(normalize=True)
        print(f"  hop={h}: correct={et.get('none',0):.1%}  "
              f"omission={et.get('omission',0):.1%}  "
              f"hallucination={et.get('hallucination',0):.1%}  "
              f"reasoning_error={et.get('reasoning_error',0):.1%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
