"""
Experiment 5: Compositional Depth and Clinical AI Failure — Main Study
=======================================================================
Tests the Peng theorem (arXiv:2402.08164) in clinical AI:

  CORE HYPOTHESIS: Transformer accuracy degrades with compositional depth (hop
  count). The predicted failure SIGNATURE is not just accuracy loss but
  OUTPUT HOMOGENIZATION — cross-model agreement-when-wrong increases with
  hop count because all models compress toward the same high-prior response.
  Reasoning models (CoT enabled) restore output diversity and flatten the
  hop→error curve.

Within-model ablations (same weights, reasoning toggled — cleanest test):
  A: claude-sonnet-4-6  zero-shot            — dense baseline
  B: claude-sonnet-4-6  extended thinking    — reasoning enabled
     → A↔B isolates CoT mechanism from model capability

Cross-model reference conditions:
  C: gemini-2.5-flash                        — best NOHARM performance
  D: gpt-5.4                                 — latest OpenAI dense model

Pre-registered primary test:
  β₃ in logit(accuracy) ~ hop_count × is_CoT + covariates + (1|question)
  emtrends: CoT benefit slope vs. zero-shot benefit slope across hop levels

Pre-registered variance test:
  cross_model_agreement_when_wrong ~ hop_count
  (Peng prediction: increases with hop count)

Covariates:
  - ehr_char_count: context completeness (separates context truncation from
    reasoning failure — critical confound identified by informatics expert)
  - question_token_count: length effects
  - domain: specialty fixed effects

Judge:
  gpt-4o-mini with pre-specified prompt (registered below as JUDGE_SYS)
  Validation: 10% physician adjudication subsample (offline)

Dataset:
  313 hop-annotated MedAlign questions (all available, not subsampled)
  Distribution: hop=1 (n=112), hop=2 (n=47), hop=3 (n=43), hop=4 (n=111)

Output:
  results/main_experiment_raw.csv
  results/main_experiment_scores.csv
"""

import json, time, xml.etree.ElementTree as ET, os
from pathlib import Path
import modal
import numpy as np
import pandas as pd

# ── Paths (local only — data passed as JSON payload) ──────────────────────────

MEDAL_BASE = Path("/Users/sanjaybasu/waymark-local/data/medalign"
                  "/MedAlign_files/medalign_instructions_v1_3")
ANNOT_PATH = Path("/Users/sanjaybasu/waymark-local/notebooks"
                  "/ai-clinical-rtm/results/hop_annotations.csv")
OUT_DIR    = Path("/Users/sanjaybasu/waymark-local/notebooks"
                  "/ai-clinical-rtm/results")

# ── Load API keys locally ─────────────────────────────────────────────────────

def _load_env(path: str) -> dict:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    out[k] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    return out

_env = _load_env("/Users/sanjaybasu/waymark-local/notebooks"
                 "/rl_vs_llm_safety_v2/.env")

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or _env.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY")    or _env.get("OPENAI_API_KEY", "")
GOOGLE_KEY    = os.environ.get("GOOGLE_API_KEY")    or _env.get("GOOGLE_API_KEY", "")

# ── Pre-registered judge prompt (do not modify after pre-registration) ────────
# Registered on OSF prior to data collection.

JUDGE_SYS = """You are a clinical QA evaluator. Score model responses against clinician reference answers.

Return ONLY valid JSON — no other text:
{
  "correct": true or false,
  "error_type": "none" | "omission" | "hallucination" | "reasoning_error",
  "confidence": 0.0-1.0
}

Definitions:
- correct: model response substantially agrees with the clinician reference
- omission: model failed to answer, refused, gave empty/irrelevant response, or missed required info
- hallucination: model stated facts not in the EHR or contradicts medical knowledge
- reasoning_error: model had the right facts but reached the wrong conclusion
- confidence: your confidence in this scoring (0=unsure, 1=certain)"""

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("clinical-peng-v5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "anthropic>=0.49.0",
        "openai>=1.0",
        "google-genai>=0.8",
        "numpy>=1.26",
        "pandas>=2.0",
    )
)

# ── Remote evaluation function ────────────────────────────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "OPENAI_API_KEY":    OPENAI_KEY,
        "GOOGLE_API_KEY":    GOOGLE_KEY,
    })],
    timeout=600,
    retries=1,
    max_containers=3,      # 3 containers × 4 conditions = ≤12 concurrent API calls
)
def evaluate_question(payload: str) -> str:
    """
    Evaluate one question across all model conditions.

    payload: JSON with {instruction_id, question, hop_count, clinician_response,
                        ehr_text, ehr_char_count, question_token_count, domain}
    returns: JSON list of result records
    """
    import anthropic as _anthropic
    import openai as _openai
    from google import genai as _genai
    import json as _json
    import os as _os
    import time as _time

    record   = _json.loads(payload)
    question = record["question"]
    ehr_text = record["ehr_text"]
    clinician= record["clinician_response"]
    iid      = record["instruction_id"]
    hop      = record["hop_count"]
    ehr_chars= record["ehr_char_count"]
    q_tokens = record["question_token_count"]
    domain   = record["domain"]

    anth_client   = _anthropic.Anthropic(api_key=_os.environ["ANTHROPIC_API_KEY"])
    openai_client = _openai.OpenAI(api_key=_os.environ["OPENAI_API_KEY"])
    gemini_client = _genai.Client(api_key=_os.environ.get("GOOGLE_API_KEY", ""))

    def _retry(fn, max_attempts=5, base_sleep=2.0):
        """Retry with exponential backoff on 429/overload errors."""
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:
                if attempt < max_attempts - 1 and (
                    "429" in str(e) or "overload" in str(e).lower() or
                    "rate_limit" in str(e).lower()
                ):
                    _time.sleep(base_sleep * (2 ** attempt))
                    continue
                raise

    # ── Generators ──────────────────────────────────────────────────────────

    def claude_zeroshot():
        def _call():
            r = anth_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\nAnswer concisely."}]
            )
            return r.content[0].text.strip(), 0
        return _retry(_call)

    def claude_extended_thinking():
        """Same model as zero-shot, reasoning enabled. Primary within-model ablation."""
        def _call():
            r = anth_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 10000},
                messages=[{"role": "user", "content":
                    f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\nAnswer concisely."}]
            )
            return r
        r = _retry(_call)
        # Capture thinking token count as metacognitive proxy
        think_tokens = sum(
            len(b.thinking.split()) * 1.3  # rough token estimate
            for b in r.content if hasattr(b, "thinking") and b.type == "thinking"
        )
        # Extract final text block
        text_blocks = [b for b in r.content if b.type == "text"]
        text = text_blocks[0].text.strip() if text_blocks else ""
        return text, int(think_tokens)

    def gemini_standard():
        # google-genai SDK (new). Try 2.5 Flash then fallback to 2.0 Flash.
        for model_id in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                resp = gemini_client.models.generate_content(
                    model=model_id,
                    contents=f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\nAnswer concisely.",
                )
                return resp.text.strip(), 0
            except Exception:
                continue
        raise RuntimeError("All Gemini model IDs failed")

    def gpt54_standard():
        r = openai_client.chat.completions.create(
            model="gpt-5.4",
            max_completion_tokens=512,
            messages=[{"role": "user", "content":
                f"Patient EHR:\n{ehr_text}\n\nQuestion: {question}\n\nAnswer concisely."}]
        )
        return r.choices[0].message.content.strip(), 0

    # ── Judge ────────────────────────────────────────────────────────────────

    def judge(response: str) -> dict:
        """gpt-4o-mini judge with pre-registered prompt."""
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=120,
            messages=[
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user",   "content":
                    f"Question: {question}\nClinician reference: {clinician}\n"
                    f"Model response: {response}"}
            ]
        )
        raw = r.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        try:
            return _json.loads(raw)
        except Exception:
            return {"correct": False, "error_type": "error", "confidence": 0.0}

    # ── Evaluate all conditions ──────────────────────────────────────────────

    conditions = {
        "claude_zeroshot":          claude_zeroshot,
        "claude_extended_thinking": claude_extended_thinking,
        "gemini_standard":          gemini_standard,
        "gpt54_standard":           gpt54_standard,
    }

    results = []
    for cond, fn in conditions.items():
        try:
            resp, think_tokens = fn()
            sc = judge(resp)
            results.append({
                "instruction_id":      iid,
                "condition":           cond,
                "hop_count":           hop,
                "ehr_char_count":      ehr_chars,
                "question_token_count": q_tokens,
                "domain":              domain,
                "question":            question,
                "clinician_ref":       clinician,
                "model_response":      resp,
                "correct":             int(sc.get("correct", False)),
                "error_type":          sc.get("error_type", "error"),
                "judge_confidence":    sc.get("confidence", 0.0),
                "thinking_tokens":     think_tokens,
            })
        except Exception as e:
            results.append({
                "instruction_id":      iid,
                "condition":           cond,
                "hop_count":           hop,
                "ehr_char_count":      ehr_chars,
                "question_token_count": q_tokens,
                "domain":              domain,
                "question":            question,
                "clinician_ref":       clinician,
                "model_response":      f"ERROR: {e}",
                "correct":             0,
                "error_type":          "error",
                "judge_confidence":    0.0,
                "thinking_tokens":     0,
            })
        _time.sleep(1.5)   # rate limiting between conditions (313 workers × 4 calls)

    return _json.dumps(results)


# ── Local utilities ───────────────────────────────────────────────────────────

def extract_ehr(xml_path: Path, max_chars: int = 8000) -> tuple[str, int]:
    """Returns (ehr_text, full_char_count). full_char_count is the completeness signal."""
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
        full = "\n".join(segs)
        return full[:max_chars], len(full)
    except Exception as e:
        return f"[EHR error: {e}]", 0


def infer_domain(specialty: str) -> str:
    """Map MedAlign submitter_specialty to broad domain."""
    s = str(specialty).lower()
    for domain, keywords in {
        "cardiology":  ["cardiol", "heart", "cardiac"],
        "nephrology":  ["nephrol", "renal", "kidney"],
        "oncology":    ["oncol", "cancer", "tumor"],
        "pulmonary":   ["pulmon", "lung", "respir"],
        "neurology":   ["neurol", "neuro"],
        "endocrine":   ["endocrin", "diabet", "thyroid"],
        "primary_care":["primary", "general", "internal", "family"],
        "surgery":     ["surg"],
        "psychiatry":  ["psych"],
    }.items():
        if any(kw in s for kw in keywords):
            return domain
    return "other"


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    # ── Load annotations ──
    annot = pd.read_csv(ANNOT_PATH).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)
    print(f"Loaded {len(annot)} annotated questions")
    print("Hop distribution:", annot["hop_count"].value_counts().sort_index().to_dict())

    # ── Load MedAlign reference answers ──
    df_resp  = pd.read_csv(MEDAL_BASE / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map  = (df_resp.drop_duplicates("instruction_id")
                [["instruction_id", "filename", "clinician_response"]]
                .set_index("instruction_id"))

    sample = annot[annot["instruction_id"].isin(ref_map.index)].copy()
    print(f"Questions with reference answers: {len(sample)}")

    # ── Resume support ──
    raw_path = OUT_DIR / "main_experiment_raw.csv"
    if raw_path.exists():
        done_df  = pd.read_csv(raw_path)
        # Done key: questions already evaluated (all 4 conditions present)
        done_counts = done_df.groupby("instruction_id")["condition"].count()
        done_ids    = set(done_counts[done_counts >= 4].index.tolist())
        all_rows    = done_df.to_dict("records")
        print(f"Resuming: {len(done_ids)} questions fully evaluated")
    else:
        done_ids, all_rows = set(), []

    # ── Build payloads ──
    payloads = []
    skipped  = 0
    for _, row in sample.iterrows():
        iid = row["instruction_id"]
        if iid in done_ids:
            skipped += 1
            continue
        filename = str(ref_map.loc[iid, "filename"])
        ehr_text, ehr_char_count = extract_ehr(MEDAL_BASE / "ehrs" / filename)
        payloads.append(json.dumps({
            "instruction_id":       iid,
            "question":             row["question"],
            "hop_count":            int(row["hop_count"]),
            "clinician_response":   str(ref_map.loc[iid, "clinician_response"]),
            "ehr_text":             ehr_text,
            "ehr_char_count":       ehr_char_count,
            "question_token_count": len(row["question"].split()),
            "domain":               infer_domain(row.get("submitter_specialty", "")),
        }))

    print(f"\nDispatching {len(payloads)} questions to Modal "
          f"({skipped} already done, {len(payloads)} remaining)")
    print(f"Total evaluations: {len(payloads)} × 4 conditions = {len(payloads)*4}")
    print(f"Estimated time @ 8 concurrent workers: ~{len(payloads)//8 * 45}s")

    if not payloads:
        print("Nothing to do — all questions already evaluated.")
    else:
        n_done = 0
        for raw_out in evaluate_question.map(payloads, order_outputs=False):
            rows    = json.loads(raw_out)
            all_rows.extend(rows)
            n_done += 1
            iid_out  = rows[0]["instruction_id"] if rows else "?"
            hop_out  = rows[0]["hop_count"] if rows else "?"
            acc_out  = [r["correct"] for r in rows if r.get("model_response","").startswith("ERROR") is False]
            acc_str  = f"acc={sum(acc_out)}/{len(acc_out)}" if acc_out else "errors"
            print(f"  [{n_done:3d}/{len(payloads)}] {iid_out}  hop={hop_out}  {acc_str}")

            # Checkpoint every 20 questions
            if n_done % 20 == 0:
                pd.DataFrame(all_rows).to_csv(raw_path, index=False)
                print(f"    [checkpoint saved: {len(all_rows)} rows]")

    # ── Save raw ──
    df_out = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df_out)} raw rows → {raw_path}")

    # ── Aggregate: primary accuracy table ──
    agg = df_out.groupby(["condition", "hop_count"]).agg(
        n                    = ("correct", "count"),
        accuracy             = ("correct", "mean"),
        omission_rate        = ("error_type", lambda x: (x == "omission").mean()),
        hallucination_rate   = ("error_type", lambda x: (x == "hallucination").mean()),
        reasoning_error_rate = ("error_type", lambda x: (x == "reasoning_error").mean()),
        mean_thinking_tokens = ("thinking_tokens", "mean"),
        mean_ehr_completeness= ("ehr_char_count", "mean"),
        mean_judge_confidence= ("judge_confidence", "mean"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "main_experiment_scores.csv", index=False)
    print("\n=== Accuracy by hop count × condition ===")
    print(agg.to_string(index=False))

    # ── Variance metric: cross-model agreement-when-wrong ──
    # For each question where ≥2 models are wrong, measure whether wrong models
    # give the same error type (convergence = Peng homogenization signature)
    print("\n=== Cross-model agreement-when-wrong by hop count ===")
    pivot = df_out.pivot_table(
        index="instruction_id", columns="condition",
        values=["correct", "error_type"], aggfunc="first"
    )
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot.merge(
        annot[["instruction_id", "hop_count"]].drop_duplicates(), on="instruction_id"
    )

    correct_cols  = [c for c in pivot.columns if c.startswith("correct_")]
    errtype_cols  = [c for c in pivot.columns if c.startswith("error_type_")]

    def agreement_when_wrong(row):
        """Fraction of wrong-model pairs that share the same error type."""
        wrong_types = [row[c] for c in errtype_cols
                       if c.replace("error_type_", "correct_") in pivot.columns
                       and row.get(c.replace("error_type_", "correct_"), 1) == 0
                       and row[c] not in ("error", "none")]
        if len(wrong_types) < 2:
            return np.nan
        from itertools import combinations
        pairs = list(combinations(wrong_types, 2))
        return sum(a == b for a, b in pairs) / len(pairs)

    pivot["agreement_when_wrong"] = pivot.apply(agreement_when_wrong, axis=1)
    aww_by_hop = (pivot.groupby("hop_count")["agreement_when_wrong"]
                  .agg(["mean", "count", "sem"]).reset_index())
    aww_by_hop.columns = ["hop_count", "mean_agreement", "n_questions", "se_agreement"]
    aww_by_hop.to_csv(OUT_DIR / "main_experiment_variance.csv", index=False)
    print(aww_by_hop.to_string(index=False))
    print("\nPeng prediction: mean_agreement should INCREASE with hop count")

    # ── CoT benefit slope (primary interaction test) ──
    print("\n=== CoT benefit by hop count (pre-registered primary test) ===")
    zs  = agg[agg["condition"] == "claude_zeroshot"].set_index("hop_count")["accuracy"]
    et  = agg[agg["condition"] == "claude_extended_thinking"].set_index("hop_count")["accuracy"]
    benefit = (et - zs).dropna()
    print("Claude ET - ZeroShot (should increase with hop):")
    print(benefit.round(3))
    slope_direction = "✓ CONSISTENT" if (
        benefit.index.tolist() == sorted(benefit.index) and
        benefit.is_monotonic_increasing
    ) else "✗ NON-MONOTONIC"
    print(f"Monotonic increase: {slope_direction}")

    # ── Thinking token scaling (mechanistic Peng test) ──
    print("\n=== Thinking token scaling (metacognitive proxy) ===")
    tt = df_out[df_out["condition"] == "claude_extended_thinking"].groupby("hop_count")[
        "thinking_tokens"].mean()
    print("Mean thinking tokens by hop count (Peng predicts O(√n/H·d·p)):")
    print(tt.round(0))

    print(f"\n=== All results saved to {OUT_DIR} ===")
    print(f"  Raw data:        main_experiment_raw.csv")
    print(f"  Accuracy scores: main_experiment_scores.csv")
    print(f"  Variance metric: main_experiment_variance.csv")


if __name__ == "__main__":
    print("Run with: modal run experiments/05_main_experiment_modal.py")
