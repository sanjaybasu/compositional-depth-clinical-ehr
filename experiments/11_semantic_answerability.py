"""
Experiment 11: Sentence-Level Semantic Answerability
=====================================================
Sensitivity analysis for context-sufficiency audit using semantic matching
(sentence-level cosine similarity via sentence-transformers) rather than
lexical token overlap.

Reviewer request: "a sensitivity analysis of the context sufficiency audit
using semantic matching (e.g., sentence-level entailment or synonym-aware
alignment) rather than lexical overlap."

Method:
  1. Load full EHR XML (no truncation, max_chars=999999)
  2. Chunk EHR into 200-char sliding windows (50-char overlap)
  3. Split clinician_ref into sentences (split on '. ')
  4. Embed chunks + ref sentences via all-MiniLM-L6-v2
  5. Mark sentence "supported" if max cosine similarity to any chunk > 0.5
  6. semantic_coverage = fraction of ref sentences supported
  7. any_sentence_supported = 1 if ≥1 sentence supported

Output: results/semantic_answerability.csv
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
                "/medalign_instructions_v1_3")
EHR_DIR  = BASE / "ehrs"
TSV_PATH = BASE / "clinician-reviewed-model-responses.tsv"
RAW_CSV  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                "/results/claude_experiment_raw.csv")
OUT_CSV  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                "/results/semantic_answerability.csv")

# ── Hyper-parameters ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 200   # characters
CHUNK_OVERLAP = 50    # characters
SIM_THRESHOLD = 0.5   # cosine similarity threshold for "supported"
MODEL_NAME    = "all-MiniLM-L6-v2"


# ── EHR extraction (task-specified generic iter approach) ──────────────────────
def extract_ehr(xml_path: Path, max_chars: int = 999999) -> tuple[str, int]:
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


def make_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunks."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, max(1, len(text) - overlap), step):
        chunk = text[start: start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def split_sentences(text: str) -> list[str]:
    """Split clinician reference on '. ' and strip empties."""
    parts = text.split(". ")
    return [s.strip() for s in parts if s.strip()]


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a and rows of b → (len_a, len_b)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def process_one(instruction_id: int, hop_count: int, clinician_ref: str,
                ehr_filename: str, model: SentenceTransformer) -> dict:
    xml_path = EHR_DIR / ehr_filename
    ehr_text, _ = extract_ehr(xml_path)

    # chunk EHR
    chunks = make_chunks(ehr_text)
    if not chunks:
        return dict(
            instruction_id=instruction_id, hop_count=hop_count,
            semantic_coverage=0.0, any_sentence_supported=0,
            n_ref_sentences=0, n_supported_sentences=0
        )

    # split reference
    ref_sentences = split_sentences(str(clinician_ref))
    if not ref_sentences:
        return dict(
            instruction_id=instruction_id, hop_count=hop_count,
            semantic_coverage=0.0, any_sentence_supported=0,
            n_ref_sentences=0, n_supported_sentences=0
        )

    # encode
    chunk_embs = model.encode(chunks, batch_size=256, show_progress_bar=False,
                              convert_to_numpy=True)
    ref_embs   = model.encode(ref_sentences, batch_size=256,
                              show_progress_bar=False, convert_to_numpy=True)

    # similarity: (n_ref, n_chunks)
    sim = cosine_sim_matrix(ref_embs, chunk_embs)
    max_sim_per_sentence = sim.max(axis=1)  # shape (n_ref,)

    supported = (max_sim_per_sentence > SIM_THRESHOLD)
    n_supported = int(supported.sum())
    n_ref = len(ref_sentences)
    coverage = n_supported / n_ref if n_ref > 0 else 0.0
    any_supported = int(n_supported >= 1)

    return dict(
        instruction_id=instruction_id,
        hop_count=hop_count,
        semantic_coverage=coverage,
        any_sentence_supported=any_supported,
        n_ref_sentences=n_ref,
        n_supported_sentences=n_supported
    )


def main():
    print("Loading data and model...")
    raw_df  = pd.read_csv(RAW_CSV)
    tsv_df  = pd.read_table(TSV_PATH)

    # ZS condition only; deduplicate to one row per instruction_id
    zs = (raw_df[raw_df['condition'] == 'claude_zeroshot']
          [['instruction_id', 'hop_count', 'clinician_ref']]
          .drop_duplicates('instruction_id')
          .reset_index(drop=True))
    print(f"  {len(zs)} unique ZS questions")

    # instruction_id → filename mapping
    mapping = (tsv_df[['instruction_id', 'filename']]
               .drop_duplicates('instruction_id')
               .set_index('instruction_id')['filename'])

    zs['ehr_filename'] = zs['instruction_id'].map(mapping)
    missing = zs['ehr_filename'].isna().sum()
    if missing:
        print(f"  WARNING: {missing} questions have no EHR filename mapping; skipping")
    zs = zs.dropna(subset=['ehr_filename']).reset_index(drop=True)

    # Load sentence-transformer model once
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Process each question
    records = []
    n = len(zs)
    for i, row in zs.iterrows():
        if i % 50 == 0:
            print(f"  Processing {i+1}/{n} ...")
        rec = process_one(
            instruction_id=row['instruction_id'],
            hop_count=int(row['hop_count']),
            clinician_ref=row['clinician_ref'],
            ehr_filename=row['ehr_filename'],
            model=model
        )
        records.append(rec)

    results = pd.DataFrame(records)
    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Rows: {len(results)}")

    # ── Analysis ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SEMANTIC ANSWERABILITY RESULTS:")
    print("="*60)

    summary_rows = []
    for hop in sorted(results['hop_count'].unique()):
        sub = results[results['hop_count'] == hop]
        any_pct  = sub['any_sentence_supported'].mean() * 100
        mean_cov = sub['semantic_coverage'].mean()
        n_q      = len(sub)
        print(f"hop={hop}: any_supported={any_pct:.1f}%,  "
              f"mean_coverage={mean_cov:.3f}  (n={n_q})")
        summary_rows.append(dict(hop=hop, any_pct=any_pct,
                                 mean_cov=mean_cov, n=n_q))

    # Kruskal-Wallis across hop groups
    groups = [results[results['hop_count'] == h]['semantic_coverage'].values
              for h in sorted(results['hop_count'].unique())]
    h_stat, p_val = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis H={h_stat:.2f}, p={p_val:.3f}")

    # Conclusion
    hop_means = {row['hop']: row['mean_cov'] for row in summary_rows}
    slope_sign = hop_means[max(hop_means)] - hop_means[min(hop_means)]
    direction = "lower" if slope_sign < 0 else "higher"
    sig_str   = "statistically significant" if p_val < 0.05 else "not statistically significant"
    conclusion = (
        f"Higher-hop questions show {direction} semantic coverage "
        f"(hop-1 mean={hop_means[1]:.3f} vs hop-4 mean={hop_means[4]:.3f}); "
        f"difference is {sig_str} (p={p_val:.3f})."
    )
    print(f"\nCONCLUSION: {conclusion}")
    print("="*60)

    # Additional descriptive stats
    print("\nOverall statistics:")
    print(f"  N questions: {len(results)}")
    print(f"  Mean semantic coverage: {results['semantic_coverage'].mean():.3f} "
          f"(SD={results['semantic_coverage'].std():.3f})")
    print(f"  Fraction any-sentence supported: "
          f"{results['any_sentence_supported'].mean()*100:.1f}%")
    print(f"  Mean ref sentences per question: {results['n_ref_sentences'].mean():.1f}")


if __name__ == "__main__":
    main()
