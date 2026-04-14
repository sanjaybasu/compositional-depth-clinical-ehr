"""
Experiment 16: Cross-Provider Judge Validation for GPT-4o Condition
=====================================================================
The automated judge (gpt-4o-mini, OpenAI) and one evaluated model (gpt-4o,
OpenAI) come from the same provider family. This experiment empirically
eliminates the concern by re-scoring all 301 GPT-4o responses using a
Claude-based judge (claude-haiku-4-5) and computing:
  1. κ(Claude judge, GPT-4o-mini judge) — should exceed 0.80 to rule out bias
  2. Per-hop accuracy under each judge — should show the same P1 slope
  3. CA trend test under Claude judge — confirms provider independence

If κ > 0.80 and slopes are concordant, the same-provider concern is
empirically refuted rather than merely acknowledged.

Run:
  ANTHROPIC_API_KEY=... python3 experiments/16_cross_provider_judge.py
"""

import json, time, os, math
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from scipy import stats
from sklearn.metrics import cohen_kappa_score

# ── Paths ──────────────────────────────────────────────────────────────────────
GPT4O_CSV = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                 "/results/gpt4o_zeroshot_raw.csv")
OUT       = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                 "/results/cross_provider_judge_raw.csv")

# ── API key ────────────────────────────────────────────────────────────────────
def _load_key(env_path, key):
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return ""

_ENV          = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or _load_key(_ENV, "ANTHROPIC_API_KEY")
anth_client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── Claude judge prompt (same scoring schema as pre-registered judge) ──────────
CLAUDE_JUDGE_SYS = """You are a clinical QA evaluator. Score model responses against clinician reference answers.
Return ONLY valid JSON — no other text:
{"correct": true or false, "error_type": "none" | "omission" | "hallucination" | "reasoning_error", "confidence": 0.0-1.0}
Definitions:
- correct: model response substantially agrees with the clinician reference
- omission: model failed to answer, refused, gave empty/irrelevant response
- hallucination: model stated facts not in the EHR or contradicts medical knowledge
- reasoning_error: model had the right facts but reached the wrong conclusion
- confidence: your confidence in this scoring (0=unsure, 1=certain)"""


def _claude_judge(question: str, ref: str, response: str,
                  max_retries: int = 6) -> dict:
    """Score one response using claude-haiku-4-5 as the judge."""
    delay = 20
    for attempt in range(max_retries):
        try:
            r = anth_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                system=CLAUDE_JUDGE_SYS,
                messages=[{"role": "user", "content":
                           f"Question: {question}\n"
                           f"Clinician reference: {ref}\n"
                           f"Model response: {response}"}],
            )
            raw = r.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except AnthropicRateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [429 rate-limit, sleeping {delay}s …]")
            time.sleep(delay)
            delay = min(delay * 2, 300)
        except Exception as e:
            return {"correct": False, "error_type": "error",
                    "confidence": 0.0, "parse_error": str(e)}


# ── Statistics helpers ─────────────────────────────────────────────────────────
def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p     = k / n
    denom = 1 + z**2 / n
    ctr   = (p + z**2 / (2 * n)) / denom
    marg  = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, ctr - marg), min(1.0, ctr + marg)


def _cochran_armitage(ks, ns):
    """One-tailed CA trend (decline direction)."""
    ks, ns = np.array(ks, float), np.array(ns, float)
    scores = np.arange(1, len(ks) + 1, dtype=float)
    N    = ns.sum()
    pbar = ks.sum() / N
    S    = (scores * (ks - ns * pbar)).sum()
    V    = pbar * (1 - pbar) * (N * (scores**2 * ns).sum()
                                 - ((scores * ns).sum())**2) / N
    if V <= 0:
        return float("nan"), float("nan")
    z = S / math.sqrt(V)
    p = 1 - stats.norm.cdf(z)
    return z, p


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(GPT4O_CSV)
    print(f"Loaded {len(df)} GPT-4o responses.")
    print("Hop distribution:", df["hop_count"].value_counts().sort_index().to_dict())

    # Resume logic
    if OUT.exists():
        done_df  = pd.read_csv(OUT)
        done_ids = set(done_df["instruction_id"].astype(int))
        records  = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} already scored")
    else:
        done_ids, records = set(), []

    total  = len(df)
    n_done = len(records)

    for _, row in df.iterrows():
        iid = int(row["instruction_id"])
        if iid in done_ids:
            continue

        question = str(row["question"])
        ref      = str(row["clinician_ref"])
        response = str(row["model_response"])
        hop      = int(row["hop_count"])
        orig_correct = int(row["correct"])

        try:
            sc = _claude_judge(question, ref, response)

            records.append({
                "instruction_id":          iid,
                "hop_count":               hop,
                "gpt4o_mini_correct":      orig_correct,       # original judge
                "claude_judge_correct":    int(sc.get("correct", False)),
                "claude_judge_error_type": sc.get("error_type", "error"),
                "claude_judge_confidence": sc.get("confidence", 0.0),
            })

            agree = "✓" if orig_correct == int(sc.get("correct", False)) else "✗"
            n_done += 1
            print(f"[{n_done:3d}/{total}] hop={hop} "
                  f"gpt4o-mini={'Y' if orig_correct else 'N'} "
                  f"claude={'Y' if sc.get('correct') else 'N'} "
                  f"{agree}  {sc.get('error_type','?')}")

        except Exception as e:
            print(f"  ERROR iid={iid}: {e}")
            n_done += 1

        time.sleep(0.3)

        if n_done % 50 == 0 and records:
            pd.DataFrame(records).to_csv(OUT, index=False)
            print(f"  [checkpoint: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT}")

    # ── Agreement analysis ─────────────────────────────────────────────────────
    orig  = df_out["gpt4o_mini_correct"].values
    claude = df_out["claude_judge_correct"].values

    kappa   = cohen_kappa_score(orig, claude)
    agree_n = int((orig == claude).sum())
    pct_agree = agree_n / len(orig)

    print("\n" + "=" * 65)
    print("CROSS-PROVIDER JUDGE AGREEMENT:")
    print(f"  n = {len(df_out)}")
    print(f"  % agreement = {pct_agree:.1%}  ({agree_n}/{len(df_out)})")
    print(f"  Cohen's κ   = {kappa:.3f}")

    if kappa >= 0.80:
        print("  ✓ κ ≥ 0.80 — provider-family bias empirically ruled out.")
    elif kappa >= 0.60:
        print("  ~ κ ≥ 0.60 — substantial agreement, minor residual concern.")
    else:
        print("  ✗ κ < 0.60 — meaningful judge disagreement; report both.")

    # Per-hop agreement
    print("\n  Per-hop κ:")
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        if len(sub) < 2:
            continue
        k_h = cohen_kappa_score(sub["gpt4o_mini_correct"], sub["claude_judge_correct"])
        print(f"    hop={h} (n={len(sub)}): κ={k_h:.3f}")

    # Accuracy by hop under each judge
    print("\n  Accuracy by hop — original GPT-4o-mini judge vs Claude judge:")
    ks_orig, ks_claude, ns = [], [], []
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        n   = len(sub)
        k_o = int(sub["gpt4o_mini_correct"].sum())
        k_c = int(sub["claude_judge_correct"].sum())
        ns.append(n); ks_orig.append(k_o); ks_claude.append(k_c)
        lo_o, hi_o = _wilson_ci(k_o, n)
        lo_c, hi_c = _wilson_ci(k_c, n)
        print(f"    hop={h}: gpt4o-mini={k_o/n:.1%} [{lo_o:.1%},{hi_o:.1%}]  "
              f"claude={k_c/n:.1%} [{lo_c:.1%},{hi_c:.1%}]")

    # CA trends
    ca_z_o, ca_p_o = _cochran_armitage(ks_orig,   ns)
    ca_z_c, ca_p_c = _cochran_armitage(ks_claude, ns)
    print(f"\n  Cochran–Armitage (one-tailed):")
    print(f"    GPT-4o-mini judge: z={ca_z_o:.3f}, p={ca_p_o:.4f}")
    print(f"    Claude judge:      z={ca_z_c:.3f}, p={ca_p_c:.4f}")

    if ca_p_c < 0.05:
        print("  ✓ P1 confirmed under Claude judge — result is judge-independent.")
    else:
        print("  ✗ P1 not significant under Claude judge — investigate discrepancy.")

    print("=" * 65)


if __name__ == "__main__":
    main()
