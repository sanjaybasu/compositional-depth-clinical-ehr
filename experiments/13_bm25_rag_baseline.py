"""
Experiment 13: BM25 RAG Baseline
=================================
Tests the Peng theoretical 'escape route': RAG decomposes the composition
problem into explicit retrieval + single-hop reasoning, which should flatten
the accuracy-depth curve.

Directly addresses reviewer Q6.

Design:
  - BM25 retrieval (rank_bm25.BM25Okapi) over full EHR chunks
  - Top-5 chunks (up to 2000 chars) passed to claude-sonnet-4-6 zero-shot
  - Same gpt-4o-mini judge as main experiment
  - GEE hop × condition interaction (RAG vs ZS) at end

Run:
  ANTHROPIC_API_KEY=... python3 experiments/13_bm25_rag_baseline.py
"""

import json, re, time, os, math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic, openai
from rank_bm25 import BM25Okapi
from anthropic import RateLimitError as AnthropicRateLimitError
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
            "/medalign_instructions_v1_3")
RAW_ZS = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
              "/results/claude_experiment_raw.csv")
OUT = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
           "/results/bm25_rag_raw.csv")

# ── API keys ──────────────────────────────────────────────────────────────────
def load_key(env_path, key):
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return ""

ENV_FILE = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or load_key(ENV_FILE, "ANTHROPIC_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "") or load_key(ENV_FILE, "OPENAI_API_KEY")

anth_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

# ── Judge prompt (pre-registered, same as main experiment) ────────────────────
JUDGE_SYS = """You are a clinical QA evaluator. Score model responses against clinician reference answers.
Return ONLY valid JSON — no other text:
{"correct": true or false, "error_type": "none" | "omission" | "hallucination" | "reasoning_error", "confidence": 0.0-1.0}
Definitions:
- correct: model response substantially agrees with the clinician reference
- omission: model failed to answer, refused, gave empty/irrelevant response
- hallucination: model stated facts not in the EHR or contradicts medical knowledge
- reasoning_error: model had the right facts but reached the wrong conclusion
- confidence: your confidence in this scoring (0=unsure, 1=certain)"""


# ── EHR loading (full, untruncated as specified) ──────────────────────────────
def extract_ehr_full(xml_path):
    """Return list of 'tag: value' chunk strings from all XML text nodes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        chunks = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                chunks.append(f"{tag}: {elem.text.strip()}")
        return chunks
    except Exception as e:
        return [f"EHR error: {e}"]


# ── BM25 retrieval ─────────────────────────────────────────────────────────────
def tokenize(text):
    """Lowercase, split on whitespace and punctuation."""
    return re.split(r"[\s\W]+", text.lower())


def bm25_retrieve(question, chunks, top_k=5, max_chars=2000):
    """
    Build BM25 index over EHR chunks, query with question tokens,
    return (retrieved_text, n_retrieved, n_chars).
    """
    if not chunks:
        return "", 0, 0

    tokenized_chunks = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    # top_k indices by descending score
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_idx if scores[i] > 0]

    # fall back to top_k even if score=0 (very short EHRs)
    if not top_chunks:
        top_chunks = [chunks[i] for i in top_idx[:top_k]]

    # concatenate up to max_chars
    retrieved = ""
    for ch in top_chunks:
        candidate = (retrieved + "\n" + ch).strip() if retrieved else ch
        if len(candidate) <= max_chars:
            retrieved = candidate
        else:
            # add truncated remainder if we have room
            remaining = max_chars - len(retrieved) - 1
            if remaining > 20:
                retrieved = (retrieved + "\n" + ch[:remaining]).strip()
            break

    return retrieved, len(top_chunks), len(retrieved)


# ── Generation ─────────────────────────────────────────────────────────────────
def _call_claude_with_retry(kwargs, max_retries=6):
    delay = 30
    for attempt in range(max_retries):
        try:
            return anth_client.messages.create(**kwargs)
        except AnthropicRateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [429 rate-limit, sleeping {delay}s before retry {attempt+1}/{max_retries}]")
            time.sleep(delay)
            delay = min(delay * 2, 300)
        except Exception:
            raise


def claude_bm25_rag(question, retrieved_context):
    """claude-sonnet-4-6 zero-shot with BM25-retrieved EHR context."""
    prompt = (
        f"Patient EHR (retrieved relevant sections):\n{retrieved_context}\n\n"
        f"Question: {question}\n\nAnswer concisely."
    )
    r = _call_claude_with_retry(dict(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ))
    return r.content[0].text.strip()


# ── Judge ─────────────────────────────────────────────────────────────────────
def judge(question, ref, response):
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=120,
        messages=[
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content":
                f"Question: {question}\nClinician reference: {ref}\n"
                f"Model response: {response}"},
        ],
    )
    raw = r.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(n_correct, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = n_correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


def prop_z_test(n1, k1, n2, k2):
    """Two-proportion z-test: RAG (k1/n1) vs ZS (k2/n2). Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load ZS baseline to get instruction_id / hop_count / question / clinician_ref
    df_zs = pd.read_csv(RAW_ZS)
    df_zs = df_zs[df_zs["condition"] == "claude_zeroshot"].copy()
    df_zs = df_zs.drop_duplicates("instruction_id")
    sample = df_zs[["instruction_id", "hop_count", "question", "clinician_ref"]].copy()
    sample["instruction_id"] = sample["instruction_id"].astype(int)

    # Load EHR filename map from TSV
    df_tsv = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    file_map = (df_tsv.drop_duplicates("instruction_id")
                .set_index("instruction_id")["filename"])

    sample = sample[sample["instruction_id"].isin(file_map.index)].copy()
    print(f"Questions: {len(sample)}")
    print("Hop distribution:", sample["hop_count"].value_counts().sort_index().to_dict())

    # Resume logic
    if OUT.exists():
        done_df = pd.read_csv(OUT)
        done_ids = set(done_df["instruction_id"].astype(int))
        records = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} already complete")
    else:
        done_ids, records = set(), []

    total = len(sample)
    n_done = len(records)

    for _, row in sample.iterrows():
        iid = int(row["instruction_id"])
        if iid in done_ids:
            continue

        question = str(row["question"])
        hop = int(row["hop_count"])
        cr = str(row["clinician_ref"])
        fn = str(file_map.loc[iid])

        xml_path = BASE / "ehrs" / fn
        chunks = extract_ehr_full(xml_path)

        retrieved_ctx, n_chunks, n_chars = bm25_retrieve(question, chunks)

        try:
            resp = claude_bm25_rag(question, retrieved_ctx)
            sc = judge(question, cr, resp)

            records.append({
                "instruction_id":    iid,
                "hop_count":         hop,
                "question":          question,
                "clinician_ref":     cr,
                "retrieved_context": retrieved_ctx,
                "model_response":    resp,
                "correct":           int(sc.get("correct", False)),
                "error_type":        sc.get("error_type", "error"),
                "judge_confidence":  sc.get("confidence", 0.0),
                "n_retrieved_chunks": n_chunks,
                "retrieved_chars":    n_chars,
            })
            mark = "+" if sc.get("correct") else "-"
            n_done += 1
            print(f"[{n_done:3d}/{total}] BM25_RAG {mark} hop={hop} "
                  f"chunks={n_chunks} chars={n_chars:4d} "
                  f"{sc.get('error_type','?'):18s}: {question[:45]}")

        except Exception as e:
            print(f"  ERROR iid={iid}: {e}")
            n_done += 1

        time.sleep(0.5)

        # Checkpoint every 25 questions
        if n_done % 25 == 0 and records:
            pd.DataFrame(records).to_csv(OUT, index=False)
            print(f"  [checkpoint saved: {n_done}/{total}]")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT}")

    # ── Results summary ───────────────────────────────────────────────────────
    # ZS baseline accuracies (pre-computed, verified against claude_experiment_raw.csv)
    zs_acc = {1: (0.306, 111), 2: (0.283, 46), 3: (0.214, 42), 4: (0.173, 104)}

    print("\n" + "=" * 65)
    print("BM25 RAG RESULTS:")
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        n = len(sub)
        if n == 0:
            continue
        k = sub["correct"].sum()
        acc = k / n
        lo, hi = wilson_ci(k, n)
        zs_a, zs_n = zs_acc[h]
        print(f"hop={h} (n={n}): {acc:.1%} [{lo:.1%}, {hi:.1%}]  vs ZS baseline {zs_a:.1%}")

    n_all = len(df_out)
    k_all = df_out["correct"].sum()
    acc_all = k_all / n_all
    lo_all, hi_all = wilson_ci(k_all, n_all)
    print(f"Overall: {acc_all:.1%} [{lo_all:.1%}, {hi_all:.1%}]")

    # CA z-test: overall RAG vs ZS
    zs_n_total = sum(v[1] for v in zs_acc.values())
    zs_k_total = sum(round(v[0] * v[1]) for v in zs_acc.values())
    z_stat, p_val = prop_z_test(n_all, k_all, zs_n_total, zs_k_total)
    print(f"CA z={z_stat:.2f}, p={p_val:.3f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    rag_accs = []
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        rag_accs.append(sub["correct"].mean() if len(sub) > 0 else np.nan)

    rag_accs_arr = np.array(rag_accs)
    hops = np.array([1, 2, 3, 4])
    # simple linear slope via least-squares
    rag_slope = np.polyfit(hops, rag_accs_arr, 1)[0]
    zs_slope_vals = [zs_acc[h][0] for h in [1, 2, 3, 4]]
    zs_slope = np.polyfit(hops, zs_slope_vals, 1)[0]

    print("\nINTERPRETATION:")
    print(f"  ZS accuracy slope:  {zs_slope:+.4f} per hop")
    print(f"  RAG accuracy slope: {rag_slope:+.4f} per hop")
    if abs(rag_slope) < abs(zs_slope) * 0.5:
        print("  RAG substantially FLATTENS the accuracy-depth curve (slope halved).")
        print("  Consistent with Peng theorem: explicit retrieval reduces composition burden.")
    elif rag_slope > -0.01:
        print("  RAG nearly ELIMINATES the accuracy-depth slope (slope ~0).")
        print("  Strong support for Peng theorem escape route.")
    else:
        print("  RAG still shows MONOTONE DECLINE — retrieval does not escape the composition trap.")
        print("  Suggests failure is reasoning, not retrieval, even with explicit context.")

    # ── GEE hop × condition interaction ──────────────────────────────────────
    print("\nGEE interaction (RAG vs ZS): hop × condition")
    try:
        # Build combined long-form DataFrame
        rag_long = df_out[["instruction_id", "hop_count", "correct"]].copy()
        rag_long["condition"] = "bm25_rag"

        df_zs_full = pd.read_csv(RAW_ZS)
        df_zs_long = (df_zs_full[df_zs_full["condition"] == "claude_zeroshot"]
                      .drop_duplicates("instruction_id")
                      [["instruction_id", "hop_count", "correct"]].copy())
        df_zs_long["condition"] = "zeroshot"

        gee_df = pd.concat([rag_long, df_zs_long], ignore_index=True)
        gee_df["correct"] = gee_df["correct"].astype(float)
        gee_df["hop_count"] = gee_df["hop_count"].astype(float)
        gee_df["condition_bin"] = (gee_df["condition"] == "bm25_rag").astype(float)
        gee_df["hop_x_cond"] = gee_df["hop_count"] * gee_df["condition_bin"]

        # GEE with independence working correlation (exchangeable optional but
        # instruction_id appears once per condition so independence is appropriate)
        fam = sm.families.Binomial()
        model = smf.gee(
            "correct ~ hop_count + condition_bin + hop_x_cond",
            groups="instruction_id",
            data=gee_df,
            family=fam,
            cov_struct=sm.cov_struct.Independence(),
        )
        res = model.fit()
        print(res.summary())

        # Pull the interaction coefficient
        int_coef = res.params.get("hop_x_cond", float("nan"))
        int_pval = res.pvalues.get("hop_x_cond", float("nan"))
        int_se   = res.bse.get("hop_x_cond", float("nan"))
        print(f"\nInteraction (hop × RAG condition): β={int_coef:.4f}, "
              f"SE={int_se:.4f}, p={int_pval:.3f}")
        if int_pval < 0.05 and int_coef > 0:
            print("  Significant positive interaction: RAG attenuates accuracy decline per hop.")
        elif int_pval < 0.05 and int_coef < 0:
            print("  Significant negative interaction: RAG worsens accuracy decline per hop.")
        else:
            print("  Non-significant interaction: no differential slope change by condition.")

    except Exception as e:
        print(f"  GEE failed: {e}")

    print("=" * 65)


if __name__ == "__main__":
    main()
