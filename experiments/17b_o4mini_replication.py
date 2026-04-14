"""
Experiment 17b: o4-mini Reasoning Model Replication
====================================================
Replaces o3-mini (Jan 2025) with o4-mini-2025-04-16, the current generation
of OpenAI's RL-trained reasoning models, in the reasoning-model appendix.

Keeps the scientific role intact: testing whether RL-based chain-of-thought
training (the theoretically motivated P2 escape route) eliminates the
hop-accuracy slope. o4-mini is the same paradigm as o3-mini but current.

Protocol identical to Exp 12 (o3-mini): reasoning_effort="high", same 301
questions, same 8000-char EHR truncation, same judge.

Run:
  python3 experiments/17b_o4mini_replication.py
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
               "/results/o4mini_zeroshot_raw.csv")

MODEL             = "o4-mini-2025-04-16"   # pinned for reproducibility
REASONING_EFFORT  = "high"                 # matches o3-mini protocol

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

# ── Judge prompt (pre-registered) ─────────────────────────────────────────────
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

# ── EHR parsing ────────────────────────────────────────────────────────────────
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
def _call_o4mini(question: str, ehr: str, max_retries: int = 6) -> tuple[str, int]:
    """
    Returns (response_text, reasoning_tokens).
    o4-mini exposes reasoning token count in usage.completion_tokens_details.
    """
    prompt  = f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."
    delay   = 10
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=4096,
                reasoning_effort=REASONING_EFFORT,
                messages=[{"role": "user", "content": prompt}],
            )
            text    = r.choices[0].message.content.strip()
            # Extract reasoning token count if available
            details = getattr(r.usage, "completion_tokens_details", None)
            r_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
            return text, r_tokens
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [rate-limit, sleeping {delay}s]")
            time.sleep(delay)
            delay = min(delay * 2, 180)
        except Exception:
            raise
    return "", 0


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
        except Exception:
            return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Statistics ─────────────────────────────────────────────────────────────────
def _wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; denom = 1 + z**2 / n
    ctr  = (p + z**2 / (2 * n)) / denom
    marg = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, ctr - marg), min(1.0, ctr + marg)

def _cochran_armitage(ks, ns):
    ks, ns = np.array(ks, float), np.array(ns, float)
    scores = np.arange(1, len(ks) + 1, dtype=float)
    N = ns.sum(); pbar = ks.sum() / N
    S = (scores * (ks - ns * pbar)).sum()
    V = pbar * (1 - pbar) * (N * (scores**2 * ns).sum()
                              - ((scores * ns).sum())**2) / N
    if V <= 0: return float("nan"), float("nan")
    z = S / math.sqrt(V)
    return z, stats.norm.cdf(z)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df_base = (pd.read_csv(CLAUDE)
               [lambda d: d["condition"] == "claude_zeroshot"]
               .drop_duplicates("instruction_id"))
    sample  = df_base[["instruction_id", "hop_count", "question", "clinician_ref"]].copy()
    sample["instruction_id"] = sample["instruction_id"].astype(int)

    df_tsv   = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    file_map = df_tsv.drop_duplicates("instruction_id").set_index("instruction_id")["filename"]
    sample   = sample[sample["instruction_id"].isin(file_map.index)].copy()

    print(f"Model: {MODEL}  reasoning_effort={REASONING_EFFORT}")
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
            resp, r_tokens = _call_o4mini(question, ehr)
            sc = _judge(question, cr, resp)
            records.append({
                "instruction_id":    iid,
                "condition":         "o4mini_reasoning_high",
                "hop_count":         hop,
                "ehr_char_count":    ehr_chars,
                "question":          question,
                "clinician_ref":     cr,
                "model_response":    resp,
                "correct":           int(sc.get("correct", False)),
                "error_type":        sc.get("error_type", "error"),
                "judge_confidence":  sc.get("confidence", 0.0),
                "reasoning_tokens":  r_tokens,
            })
            mark   = "+" if sc.get("correct") else "-"
            n_done += 1
            print(f"[{n_done:3d}/{total}] o4-mini {mark} hop={hop} "
                  f"r_tok={r_tokens:5d} {sc.get('error_type','?'):18s}: {question[:45]}")
        except Exception as e:
            print(f"  ERROR iid={iid}: {e}")
            n_done += 1

        time.sleep(0.5)
        if n_done % 25 == 0 and records:
            pd.DataFrame(records).to_csv(OUT, index=False)
            print(f"  [checkpoint: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT}")

    # ── Summary ────────────────────────────────────────────────────────────────
    o3mini_acc = {1: 0.405, 2: 0.304, 3: 0.238, 4: 0.147}  # from appendix
    print("\n" + "=" * 65)
    print(f"o4-mini REASONING (reasoning_effort=high) RESULTS:")
    ks, ns = [], []
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        n = len(sub); k = int(sub["correct"].sum())
        ks.append(k); ns.append(n)
        lo, hi = _wilson_ci(k, n)
        print(f"  hop={h} (n={n}): {k/n:.1%} [{lo:.1%},{hi:.1%}]  "
              f"o3-mini baseline: {o3mini_acc.get(h,0):.1%}")
    ca_z, ca_p = _cochran_armitage(ks, ns)
    lo_a, hi_a = _wilson_ci(sum(ks), sum(ns))
    print(f"  Overall: {sum(ks)/sum(ns):.1%} [{lo_a:.1%},{hi_a:.1%}]")
    print(f"  CA trend (one-tailed): z={ca_z:.3f}, p={ca_p:.4f}")

    # Reasoning token scaling (P3 analog)
    df_tok = df_out[df_out["reasoning_tokens"] > 0]
    if len(df_tok) > 10:
        from scipy.stats import pearsonr
        r_val, r_p = pearsonr(df_tok["hop_count"], df_tok["reasoning_tokens"])
        print(f"\n  P3 (reasoning token scaling): r={r_val:.3f}, p={r_p:.4f}")
        for h in [1, 2, 3, 4]:
            sub = df_tok[df_tok["hop_count"] == h]
            if len(sub):
                print(f"  hop={h}: mean={sub['reasoning_tokens'].mean():.0f} tokens  "
                      f"(n={len(sub)})")

    if ca_p < 0.05:
        print("\n  ✓ P1 confirmed — o4-mini shows same monotone decline.")
        print("    RL-trained reasoning does NOT eliminate compositional depth ceiling.")
    else:
        print("\n  ✗ P1 not significant — investigate.")
    print("=" * 65)


if __name__ == "__main__":
    main()
