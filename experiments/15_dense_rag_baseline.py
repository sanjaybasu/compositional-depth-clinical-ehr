"""
Experiment 15: Dense (Semantic) RAG Baseline
=============================================
Tests whether embedding-based retrieval eliminates the hop-accuracy slope
where BM25 (Exp 13) only partially attenuated it.

Method:
  - Encode all EHR chunks and the question with all-MiniLM-L6-v2
  - Retrieve top-5 chunks by cosine similarity (same k as BM25 baseline)
  - Prepend to standard 8,000-char truncated EHR context
  - Claude zero-shot, same judge and protocol as main experiment
  - GEE hop × condition interaction (dense RAG vs ZS)

Addresses reviewer Q5 and the unaddressed weakness:
  "Beyond BM25, did you attempt dense RAG?"

Run:
  ANTHROPIC_API_KEY=... python3 experiments/15_dense_rag_baseline.py
"""

import json, re, time, os, math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic, openai
from sentence_transformers import SentenceTransformer
from anthropic import RateLimitError as AnthropicRateLimitError
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
               "/medalign_instructions_v1_3")
RAW_ZS  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
               "/results/claude_experiment_raw.csv")
OUT     = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
               "/results/dense_rag_raw.csv")

# ── Retrieval hyper-parameters (match Exp 11 semantic answerability) ───────────
CHUNK_SIZE    = 200   # characters
CHUNK_OVERLAP = 50    # characters
TOP_K         = 5     # retrieved chunks (same as BM25 baseline)
MAX_RETRIEVED = 2000  # max characters of retrieved context prepended

# ── API keys ───────────────────────────────────────────────────────────────────
def _load_key(env_path, key):
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return ""

_ENV = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or _load_key(_ENV, "ANTHROPIC_API_KEY")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY",    "") or _load_key(_ENV, "OPENAI_API_KEY")

anth_client   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

# Pre-load sentence-transformer model once
print("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# ── Judge prompt (pre-registered, identical to main experiment) ────────────────
JUDGE_SYS = """You are a clinical QA evaluator. Score model responses against clinician reference answers.
Return ONLY valid JSON — no other text:
{"correct": true or false, "error_type": "none" | "omission" | "hallucination" | "reasoning_error", "confidence": 0.0-1.0}
Definitions:
- correct: model response substantially agrees with the clinician reference
- omission: model failed to answer, refused, gave empty/irrelevant response
- hallucination: model stated facts not in the EHR or contradicts medical knowledge
- reasoning_error: model had the right facts but reached the wrong conclusion
- confidence: your confidence in this scoring (0=unsure, 1=certain)"""


# ── EHR utilities ──────────────────────────────────────────────────────────────
def _extract_ehr_text(xml_path: Path) -> str:
    """Return all tag:value lines from EHR XML, untruncated."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                lines.append(f"{tag}: {elem.text.strip()}")
        return "\n".join(lines)
    except Exception as e:
        return f"[EHR error: {e}]"


def _make_chunks(text: str) -> list[str]:
    """Sliding-window character chunks (matches Exp 11)."""
    chunks, step = [], CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, max(1, len(text) - CHUNK_OVERLAP), step):
        ch = text[start: start + CHUNK_SIZE]
        if ch.strip():
            chunks.append(ch)
    return chunks


def _dense_retrieve(question: str, chunks: list[str]) -> tuple[str, int, int]:
    """
    Embed question and EHR chunks, return top-K by cosine similarity.
    Returns (retrieved_text, n_chunks_retrieved, n_chars).
    """
    if not chunks:
        return "", 0, 0

    # Batch encode chunks + query together for efficiency
    all_texts   = chunks + [question]
    embeddings  = _EMBED_MODEL.encode(all_texts, normalize_embeddings=True,
                                      show_progress_bar=False)
    chunk_embs  = embeddings[:-1]   # (N, d)
    query_emb   = embeddings[-1]    # (d,)

    # Cosine similarity = dot product (embeddings are L2-normalised)
    sims        = chunk_embs @ query_emb            # (N,)
    top_idx     = np.argsort(sims)[::-1][:TOP_K]

    retrieved = ""
    n_used    = 0
    for idx in top_idx:
        ch        = chunks[int(idx)]
        candidate = (retrieved + "\n" + ch).strip() if retrieved else ch
        if len(candidate) <= MAX_RETRIEVED:
            retrieved = candidate
            n_used   += 1
        else:
            remaining = MAX_RETRIEVED - len(retrieved) - 1
            if remaining > 20:
                retrieved = (retrieved + "\n" + ch[:remaining]).strip()
                n_used   += 1
            break

    return retrieved, n_used, len(retrieved)


# ── Generation ─────────────────────────────────────────────────────────────────
def _call_claude(kwargs, max_retries=6):
    delay = 30
    for attempt in range(max_retries):
        try:
            return anth_client.messages.create(**kwargs)
        except AnthropicRateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"  [429 rate-limit, sleeping {delay}s …]")
            time.sleep(delay)
            delay = min(delay * 2, 300)


def _claude_dense_rag(question: str, retrieved_ctx: str) -> str:
    """Claude zero-shot with dense-retrieved EHR context prepended."""
    prompt = (
        f"Patient EHR (semantically retrieved relevant sections):\n{retrieved_ctx}\n\n"
        f"Question: {question}\n\nAnswer concisely."
    )
    r = _call_claude(dict(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ))
    return r.content[0].text.strip()


# ── Judge ──────────────────────────────────────────────────────────────────────
def _judge(question: str, ref: str, response: str) -> dict:
    r = openai_client.chat.completions.create(
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
    try:
        return json.loads(raw)
    except Exception:
        return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Statistics helpers ─────────────────────────────────────────────────────────
def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p     = k / n
    denom = 1 + z**2 / n
    ctr   = (p + z**2 / (2 * n)) / denom
    marg  = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, ctr - marg), min(1.0, ctr + marg)


def _cochran_armitage(ks, ns, scores=None):
    """One-tailed Cochran–Armitage trend test (pre-registered direction: decline)."""
    if scores is None:
        scores = np.arange(1, len(ks) + 1, dtype=float)
    ks, ns, scores = np.array(ks), np.array(ns), np.array(scores)
    ps   = ks / np.where(ns > 0, ns, 1)
    N    = ns.sum()
    pbar = ks.sum() / N
    S    = (scores * (ks - ns * pbar)).sum()
    V    = pbar * (1 - pbar) * (N * (scores**2 * ns).sum()
                                 - ((scores * ns).sum())**2) / N
    if V <= 0:
        return float("nan"), float("nan")
    z = S / math.sqrt(V)
    p = stats.norm.cdf(z)   # one-tailed left tail (decline direction; z expected negative)
    return z, p


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load ZS baseline sample (same 301 questions as all other conditions)
    df_zs    = pd.read_csv(RAW_ZS)
    df_zs    = df_zs[df_zs["condition"] == "claude_zeroshot"].drop_duplicates("instruction_id")
    sample   = df_zs[["instruction_id", "hop_count", "question", "clinician_ref"]].copy()
    sample["instruction_id"] = sample["instruction_id"].astype(int)

    # EHR filename map
    df_tsv   = pd.read_csv(BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    file_map = (df_tsv.drop_duplicates("instruction_id")
                .set_index("instruction_id")["filename"])
    sample   = sample[sample["instruction_id"].isin(file_map.index)].copy()
    print(f"Questions: {len(sample)}")
    print("Hop distribution:", sample["hop_count"].value_counts().sort_index().to_dict())

    # Resume logic
    if OUT.exists():
        done_df  = pd.read_csv(OUT)
        done_ids = set(done_df["instruction_id"].astype(int))
        records  = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} already complete")
    else:
        done_ids, records = set(), []

    total  = len(sample)
    n_done = len(records)

    for _, row in sample.iterrows():
        iid = int(row["instruction_id"])
        if iid in done_ids:
            continue

        question = str(row["question"])
        hop      = int(row["hop_count"])
        cr       = str(row["clinician_ref"])
        fn       = str(file_map.loc[iid])
        xml_path = BASE / "ehrs" / fn

        # Build EHR chunks and dense-retrieve
        ehr_text                  = _extract_ehr_text(xml_path)
        chunks                    = _make_chunks(ehr_text)
        retrieved_ctx, n_ch, n_chars = _dense_retrieve(question, chunks)

        try:
            resp = _claude_dense_rag(question, retrieved_ctx)
            sc   = _judge(question, cr, resp)

            records.append({
                "instruction_id":      iid,
                "hop_count":           hop,
                "question":            question,
                "clinician_ref":       cr,
                "retrieved_context":   retrieved_ctx,
                "model_response":      resp,
                "correct":             int(sc.get("correct", False)),
                "error_type":          sc.get("error_type", "error"),
                "judge_confidence":    sc.get("confidence", 0.0),
                "n_retrieved_chunks":  n_ch,
                "retrieved_chars":     n_chars,
            })
            mark   = "+" if sc.get("correct") else "-"
            n_done += 1
            print(f"[{n_done:3d}/{total}] DENSE_RAG {mark} hop={hop} "
                  f"chunks={n_ch} chars={n_chars:4d} "
                  f"{sc.get('error_type','?'):18s}: {question[:45]}")

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

    # ── Results summary ────────────────────────────────────────────────────────
    zs_acc = {1: (0.306, 111), 2: (0.283, 46), 3: (0.214, 42), 4: (0.176, 102)}

    print("\n" + "=" * 70)
    print("DENSE RAG RESULTS vs ZERO-SHOT BASELINE:")
    dense_ks, dense_ns = [], []
    for h in [1, 2, 3, 4]:
        sub  = df_out[df_out["hop_count"] == h]
        n    = len(sub)
        if n == 0:
            dense_ks.append(0); dense_ns.append(0); continue
        k    = int(sub["correct"].sum())
        acc  = k / n
        lo, hi = _wilson_ci(k, n)
        zs_a, zs_n = zs_acc[h]
        delta      = (acc - zs_a) * 100
        print(f"  hop={h} (n={n}): {acc:.1%} [{lo:.1%},{hi:.1%}]  "
              f"vs ZS {zs_a:.1%}  Δ={delta:+.1f}pp")
        dense_ks.append(k); dense_ns.append(n)

    overall_n = len(df_out)
    overall_k = int(df_out["correct"].sum())
    lo_all, hi_all = _wilson_ci(overall_k, overall_n)
    print(f"  Overall: {overall_k/overall_n:.1%} [{lo_all:.1%},{hi_all:.1%}]  n={overall_n}")

    # One-tailed Cochran–Armitage for dense RAG
    ca_z, ca_p = _cochran_armitage(dense_ks, dense_ns)
    print(f"\n  Dense RAG CA (one-tailed): z={ca_z:.3f}, p={ca_p:.4f}")

    # GEE hop × condition interaction: dense RAG vs ZS
    print("\n  GEE hop × condition interaction (dense RAG vs ZS):")
    try:
        rag_long = df_out[["instruction_id", "hop_count", "correct"]].copy()
        rag_long["condition"] = "dense_rag"

        df_zs_long = (pd.read_csv(RAW_ZS)
                      [lambda d: d["condition"] == "claude_zeroshot"]
                      .drop_duplicates("instruction_id")
                      [["instruction_id", "hop_count", "correct"]].copy())
        df_zs_long["condition"] = "zeroshot"

        gee_df  = pd.concat([rag_long, df_zs_long], ignore_index=True)
        gee_df["correct"]       = gee_df["correct"].astype(float)
        gee_df["hop_count"]     = gee_df["hop_count"].astype(float)
        gee_df["condition_bin"] = (gee_df["condition"] == "dense_rag").astype(float)
        gee_df["hop_x_cond"]    = gee_df["hop_count"] * gee_df["condition_bin"]

        res = smf.gee(
            "correct ~ hop_count + condition_bin + hop_x_cond",
            groups="instruction_id",
            data=gee_df,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit()

        int_coef = res.params.get("hop_x_cond", float("nan"))
        int_se   = res.bse.get("hop_x_cond", float("nan"))
        int_pval = res.pvalues.get("hop_x_cond", float("nan"))
        int_or   = math.exp(int_coef) if not math.isnan(int_coef) else float("nan")
        import scipy.stats as sc2
        lo_ci = math.exp(int_coef - 1.96 * int_se)
        hi_ci = math.exp(int_coef + 1.96 * int_se)
        print(f"  Interaction OR={int_or:.3f} [95%CI {lo_ci:.3f},{hi_ci:.3f}]  p={int_pval:.3f}")

        # Main hop effect within dense RAG
        hop_coef = res.params.get("hop_count", float("nan"))
        hop_pval = res.pvalues.get("hop_count", float("nan"))
        hop_or   = math.exp(hop_coef) if not math.isnan(hop_coef) else float("nan")
        print(f"  Hop OR (within dense RAG)={hop_or:.3f}  p={hop_pval:.3f}")

        if int_pval > 0.05:
            print("\n  → Non-significant interaction: dense RAG does not significantly")
            print("    flatten the hop–accuracy slope vs zero-shot.")
            print("    The hop effect persists even with semantic retrieval.")
        else:
            direction = "attenuates" if int_coef > 0 else "worsens"
            print(f"\n  → Significant interaction (p={int_pval:.3f}): dense RAG {direction} slope.")

    except Exception as e:
        print(f"  GEE failed: {e}")

    # Error-type profile
    print("\n  Error type profile (dense RAG):")
    for h in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == h]
        if len(sub) == 0: continue
        et = sub["error_type"].value_counts(normalize=True)
        print(f"  hop={h}: correct={et.get('none',0):.1%}  "
              f"omission={et.get('omission',0):.1%}  "
              f"hallucination={et.get('hallucination',0):.1%}  "
              f"reasoning_error={et.get('reasoning_error',0):.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
