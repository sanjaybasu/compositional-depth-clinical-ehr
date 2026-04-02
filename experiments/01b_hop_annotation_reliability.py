"""
Experiment 1b: Hop Annotation Inter-Rater Reliability
======================================================
Computes automated inter-rater reliability for the LLM-based hop annotation
by running a SECOND independent annotation pass on a stratified random sample
of 50 questions (12-13 per hop level) and computing Cohen's κ against the
original annotations.

Also runs the annotation prompt on GPT-4o (different model family) as a
cross-model reliability check.

Output:
  results/hop_annotation_reliability.json   — κ statistics
  results/hop_annotations_second_pass.csv   — second-pass labels
"""

import os, json, time
from pathlib import Path
import pandas as pd
import numpy as np
import anthropic, openai
from scipy.stats import norm

BASE   = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
              "/medalign_instructions_v1_3")
ANNOT  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
              "/results/hop_annotations.csv")
OUT    = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results")
OUT.mkdir(parents=True, exist_ok=True)

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

# ── Same annotation system prompt as Experiment 1 ────────────────────────────
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


def annotate_claude(question: str, clinician_response: str, evidence: str) -> dict:
    msg = anth_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content":
            f"Question: {question}\n\nClinician reference answer: {clinician_response}\n\n"
            f"Evidence cited: {evidence}\n\nClassify this question by hop count."}]
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def annotate_gpt4o(question: str, clinician_response: str, evidence: str) -> dict:
    r = openai_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content":
                f"Question: {question}\n\nClinician reference answer: {clinician_response}\n\n"
                f"Evidence cited: {evidence}\n\nClassify this question by hop count."}
        ]
    )
    raw = r.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Cohen's κ with Fleiss standard error and 95% CI ─────────────────────────
def cohens_kappa(y1, y2, categories=None):
    """Compute Cohen's κ with Fleiss SE and 95% CI."""
    if categories is None:
        categories = sorted(set(list(y1) + list(y2)))
    k = len(categories)
    n = len(y1)
    cat_idx = {c: i for i, c in enumerate(categories)}

    conf_mat = np.zeros((k, k), dtype=int)
    for a, b in zip(y1, y2):
        if a in cat_idx and b in cat_idx:
            conf_mat[cat_idx[a], cat_idx[b]] += 1

    p_o = np.trace(conf_mat) / n  # observed agreement
    row_sums = conf_mat.sum(axis=1) / n
    col_sums = conf_mat.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)  # expected agreement

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0

    # Fleiss SE
    # σ² = [p_e + p_e² - Σ p_i(col_i + row_i)²] / [n(1-p_e)²]
    se_num = 0.0
    for i in range(k):
        se_num += row_sums[i] * col_sums[i] * (row_sums[i] + col_sums[i])**2
    if n > 0 and (1 - p_e) > 0:
        se2 = (p_e + p_e**2 - se_num) / (n * (1 - p_e)**2)
        se = max(np.sqrt(abs(se2)), 1e-9)
    else:
        se = 0.0

    z = norm.ppf(0.975)
    ci_lo = max(-1.0, kappa - z * se)
    ci_hi = min( 1.0, kappa + z * se)

    return {
        "kappa": round(kappa, 3),
        "se": round(se, 4),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "p_observed": round(p_o, 3),
        "p_expected": round(p_e, 3),
        "n": n,
    }


def main():
    df_orig = pd.read_csv(ANNOT).dropna(subset=["hop_count"])
    df_orig["hop_count"] = df_orig["hop_count"].astype(int).clip(upper=4)

    df_resp = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    resp_map = (df_resp.drop_duplicates("instruction_id")
                [["instruction_id", "clinician_response"]]
                .set_index("instruction_id"))

    # Stratified sample: 12-13 per hop (total ~50)
    np.random.seed(42)
    sample_ids = []
    for h in [1, 2, 3, 4]:
        n_per_hop = 13 if h <= 2 else 12
        hop_ids = df_orig[df_orig["hop_count"] == h]["instruction_id"].tolist()
        chosen = np.random.choice(hop_ids, size=min(n_per_hop, len(hop_ids)), replace=False)
        sample_ids.extend(chosen)

    sample = df_orig[df_orig["instruction_id"].isin(sample_ids)].copy()
    print(f"Sample size: {len(sample)} (target 50)")
    print("Hop distribution:", sample["hop_count"].value_counts().sort_index().to_dict())

    # Load or initialize second-pass results
    out2_path = OUT / "hop_annotations_second_pass.csv"
    if out2_path.exists():
        done2 = pd.read_csv(out2_path)
        done2_ids = set(done2["instruction_id"].tolist())
        records2 = done2.to_dict("records")
        print(f"Resuming second-pass: {len(done2_ids)} done")
    else:
        done2_ids, records2 = set(), []

    for _, row in sample.iterrows():
        iid = row["instruction_id"]
        if iid in done2_ids:
            continue
        q  = str(row["question"])
        cr = str(resp_map.loc[iid, "clinician_response"]) if iid in resp_map.index else ""
        ev = ""

        try:
            ann_claude = annotate_claude(q, cr, ev)
            time.sleep(1)
            ann_gpt    = annotate_gpt4o(q, cr, ev)
            records2.append({
                "instruction_id":      iid,
                "hop_count_orig":      int(row["hop_count"]),
                "hop_count_claude2":   ann_claude.get("hop_count"),
                "hop_count_gpt4o":     ann_gpt.get("hop_count"),
                "rationale_claude2":   ann_claude.get("rationale", ""),
                "rationale_gpt4o":     ann_gpt.get("rationale", ""),
            })
            done2_ids.add(iid)
            print(f"  {iid}: orig={row['hop_count']} claude2={ann_claude['hop_count']} "
                  f"gpt4o={ann_gpt.get('hop_count')} | {q[:50]}")
        except Exception as e:
            print(f"  ERROR on {iid}: {e}")

        time.sleep(1)

    df2 = pd.DataFrame(records2)
    df2.to_csv(out2_path, index=False)
    print(f"\nSaved second-pass annotations → {out2_path}")

    # ── Compute κ ─────────────────────────────────────────────────────────────
    df2 = df2.dropna(subset=["hop_count_orig", "hop_count_claude2", "hop_count_gpt4o"])
    df2["hop_count_claude2"] = df2["hop_count_claude2"].astype(int)
    df2["hop_count_gpt4o"]   = df2["hop_count_gpt4o"].astype(int)
    df2["hop_count_orig"]    = df2["hop_count_orig"].astype(int)

    kappa_claude_vs_orig = cohens_kappa(df2["hop_count_orig"], df2["hop_count_claude2"])
    kappa_gpt_vs_orig    = cohens_kappa(df2["hop_count_orig"], df2["hop_count_gpt4o"])
    kappa_claude_vs_gpt  = cohens_kappa(df2["hop_count_claude2"], df2["hop_count_gpt4o"])

    print("\n=== Hop Annotation Inter-Rater Reliability ===")
    print(f"κ(Claude orig, Claude 2nd-pass) = {kappa_claude_vs_orig['kappa']:.3f} "
          f"[{kappa_claude_vs_orig['ci_lo']:.3f}, {kappa_claude_vs_orig['ci_hi']:.3f}]")
    print(f"κ(Claude orig, GPT-4o)          = {kappa_gpt_vs_orig['kappa']:.3f} "
          f"[{kappa_gpt_vs_orig['ci_lo']:.3f}, {kappa_gpt_vs_orig['ci_hi']:.3f}]")
    print(f"κ(Claude 2nd-pass, GPT-4o)      = {kappa_claude_vs_gpt['kappa']:.3f} "
          f"[{kappa_claude_vs_gpt['ci_lo']:.3f}, {kappa_claude_vs_gpt['ci_hi']:.3f}]")

    # ── Percent agreement ──────────────────────────────────────────────────────
    pa_claude = (df2["hop_count_orig"] == df2["hop_count_claude2"]).mean()
    pa_gpt    = (df2["hop_count_orig"] == df2["hop_count_gpt4o"]).mean()
    pa_cross  = (df2["hop_count_claude2"] == df2["hop_count_gpt4o"]).mean()
    print(f"\nPercent exact agreement:")
    print(f"  Claude orig vs 2nd-pass: {pa_claude:.1%}")
    print(f"  Claude orig vs GPT-4o:   {pa_gpt:.1%}")
    print(f"  Claude 2nd-pass vs GPT-4o: {pa_cross:.1%}")

    # ── Adjacent hop agreement (±1) ────────────────────────────────────────────
    adj_claude = (abs(df2["hop_count_orig"] - df2["hop_count_claude2"]) <= 1).mean()
    adj_gpt    = (abs(df2["hop_count_orig"] - df2["hop_count_gpt4o"]) <= 1).mean()
    print(f"\nAdjacent (±1 hop) agreement:")
    print(f"  Claude orig vs 2nd-pass: {adj_claude:.1%}")
    print(f"  Claude orig vs GPT-4o:   {adj_gpt:.1%}")

    results = {
        "n_sample": len(df2),
        "kappa_claude_orig_vs_2ndpass": kappa_claude_vs_orig,
        "kappa_claude_orig_vs_gpt4o":   kappa_gpt_vs_orig,
        "kappa_claude_2ndpass_vs_gpt4o": kappa_claude_vs_gpt,
        "percent_exact_claude_orig_vs_2ndpass": round(pa_claude, 4),
        "percent_exact_claude_orig_vs_gpt4o":   round(pa_gpt, 4),
        "percent_adjacent_claude_orig_vs_2ndpass": round(adj_claude, 4),
        "percent_adjacent_claude_orig_vs_gpt4o":   round(adj_gpt, 4),
    }

    out_json = OUT / "hop_annotation_reliability.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")


if __name__ == "__main__":
    main()
