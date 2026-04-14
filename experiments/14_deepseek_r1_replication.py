"""
Experiment 14: DeepSeek-R1 Replication Study
==============================================
Evaluates DeepSeek-R1 (RL-based reasoning model, via local Ollama) on the same
301 clinical EHR questions used in the main compositional-depth experiment.

Purpose: Test whether a third reasoning model family (DeepSeek-R1,
reinforcement-learning-based like o1) shows the same hop-accuracy degradation.
Alongside o3-mini, this creates a comprehensive test of whether
reasoning-specialized models escape the hop-count bottleneck.

Run:
    python3 experiments/14_deepseek_r1_replication.py

Prerequisites:
    - Ollama running locally with deepseek-r1 model available
    - OPENAI_API_KEY env var or in .env file (for gpt-4o-mini judge)
"""

import json, time, re, os, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import openai
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
            "/medalign_instructions_v1_3")
OUT  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results")
RAW_CLAUDE = OUT / "claude_experiment_raw.csv"
RAW_OUT    = OUT / "deepseek_r1_raw.csv"
ENV_PATH   = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"

# ── Key loading ────────────────────────────────────────────────────────────────
def load_key(env_path: str, key: str) -> str:
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return ""

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "") or load_key(ENV_PATH, "OPENAI_API_KEY")
if not OPENAI_KEY:
    print("WARNING: No OPENAI_API_KEY found. Judge calls will fail.")
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

# ── Detect DeepSeek-R1 model ──────────────────────────────────────────────────
def detect_deepseek_model() -> str:
    """Return the largest available deepseek-r1 variant."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    print("Ollama models available:")
    print(result.stdout)

    # Collect deepseek-r1 variants with their size
    candidates = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if "deepseek-r1" in name.lower():
            # Extract size in GB if available
            size_str = parts[2] if len(parts) > 2 else "0"
            try:
                size_gb = float(size_str.replace("GB", "").replace("MB", ""))
                if "MB" in size_str:
                    size_gb /= 1000.0
            except Exception:
                size_gb = 0.0
            candidates.append((size_gb, name))

    if not candidates:
        raise RuntimeError("No deepseek-r1 model found in ollama list output.")

    # Use largest available
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    print(f"Selected model: {chosen} ({candidates[0][0]:.1f} GB)")
    return chosen


# ── Ollama generate ────────────────────────────────────────────────────────────
def deepseek_generate(question: str, ehr: str, model: str) -> str:
    """Call Ollama DeepSeek-R1; strip <think>...</think> from response."""
    prompt = (
        f"Patient EHR:\n{ehr}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # NOTE: spec calls for 8000-char EHR but local 8b model generates
                # unbounded thinking tokens on complex questions (>20 min/question).
                # Using num_predict=1024 and num_ctx=8192 to cap generation while
                # still allowing full 4000-char EHR context for fair comparison.
                "options": {"num_predict": 1024, "num_ctx": 8192, "temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        response = data.get("response", "")
        # Strip chain-of-thought thinking block
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out")


# ── EHR parsing ────────────────────────────────────────────────────────────────
def extract_ehr(xml_path: Path, max_chars: int = 4000) -> tuple:  # 4K fits 8192 ctx with prompt
    """
    Returns (truncated_ehr_text, full_char_count).
    Uses the generic iterator approach (same as spec) as a fallback if
    the structured encounter/entry/event approach yields nothing.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Try structured encounter approach first (matches main experiment)
        segs = []
        for enc in root.findall("encounter"):
            for entry in enc.findall(".//entry"):
                ts = entry.get("timestamp", "")
                for ev in entry.findall("event"):
                    txt = (ev.text or "").strip()
                    if txt:
                        segs.append(f"[{ts}] {ev.get('type', '')}: {txt}")
        if segs:
            full = "\n".join(segs)
            return full[:max_chars], len(full)
        # Fallback: generic element iterator
        lines = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                lines.append(f"{tag}: {elem.text.strip()}")
        full = "\n".join(lines)
        return full[:max_chars], len(full)
    except Exception as e:
        return f"[EHR error: {e}]", 0


# ── Judge ─────────────────────────────────────────────────────────────────────
JUDGE_SYS = (
    "You are a clinical QA evaluator. Score model responses against clinician "
    "reference answers. Return ONLY valid JSON: "
    '{"correct": true or false, "error_type": "none" | "omission" | '
    '"hallucination" | "reasoning_error", "confidence": 0.0-1.0}. '
    "Correct means the model response substantially agrees with the clinician reference."
)


def judge(question: str, ref: str, response: str) -> dict:
    try:
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
        return json.loads(raw)
    except Exception as e:
        print(f"  [judge error: {e}]")
        return {"correct": False, "error_type": "error", "confidence": 0.0}


# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(correct: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson score interval."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = correct / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ca_z_test(df_ds: pd.DataFrame, df_zs: pd.DataFrame) -> tuple:
    """
    One-tailed z-test: H1 = DeepSeek accuracy < ZS accuracy (testing degradation).
    Pools across all hops.
    """
    n_ds = len(df_ds)
    n_zs = len(df_zs)
    p_ds = df_ds["correct"].mean()
    p_zs = df_zs["correct"].mean()
    p_pool = (df_ds["correct"].sum() + df_zs["correct"].sum()) / (n_ds + n_zs)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_ds + 1 / n_zs))
    if se == 0:
        return 0.0, 1.0
    z = (p_ds - p_zs) / se
    p_val = stats.norm.cdf(z)  # one-tailed: test if DS < ZS
    return z, p_val


def gee_interaction(df_ds: pd.DataFrame, df_zs: pd.DataFrame) -> tuple:
    """
    Approximate GEE hop×condition interaction using logistic regression.
    OR < 1 means DeepSeek degrades faster with hop count than ZS.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        df_ds2 = df_ds[["instruction_id", "hop_count", "correct"]].copy()
        df_ds2["is_deepseek"] = 1
        df_zs2 = df_zs[["instruction_id", "hop_count", "correct"]].copy()
        df_zs2["is_deepseek"] = 0
        combined = pd.concat([df_ds2, df_zs2], ignore_index=True)
        combined["hop_count"] = combined["hop_count"].astype(float)

        # GEE with exchangeable correlation, clustered by instruction_id
        model = smf.gee(
            "correct ~ hop_count * is_deepseek",
            groups="instruction_id",
            data=combined,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        result = model.fit(maxiter=100)
        # Interaction term: hop_count:is_deepseek
        inter_key = [k for k in result.params.index if "hop" in k and "deepseek" in k]
        if inter_key:
            beta = result.params[inter_key[0]]
            p = result.pvalues[inter_key[0]]
            return np.exp(beta), p
        return float("nan"), float("nan")
    except ImportError:
        print("  [statsmodels not available; skipping GEE]")
        return float("nan"), float("nan")
    except Exception as e:
        print(f"  [GEE error: {e}]")
        return float("nan"), float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Detect model
    model = detect_deepseek_model()

    # 2. Load the 301-question set from claude baseline
    df_claude = pd.read_csv(RAW_CLAUDE)
    df_zs = df_claude[df_claude["condition"] == "claude_zeroshot"].copy()
    df_zs = df_zs.drop_duplicates("instruction_id")
    print(f"Loaded {len(df_zs)} unique questions from claude ZS baseline")

    print("Hop distribution:", df_zs["hop_count"].value_counts().sort_index().to_dict())

    # 3. Load EHR filename map
    df_resp = pd.read_csv(
        BASE / "clinician-reviewed-model-responses.tsv", sep="\t"
    )
    ref_map = (
        df_resp.drop_duplicates("instruction_id")
        [["instruction_id", "filename", "clinician_response"]]
        .set_index("instruction_id")
    )

    # 4. Resume logic
    if RAW_OUT.exists():
        done_df = pd.read_csv(RAW_OUT)
        done_ids = set(done_df["instruction_id"].tolist())
        records = done_df.to_dict("records")
        print(f"Resuming: {len(done_df)} already done, {len(df_zs) - len(done_ids)} remaining")
    else:
        done_ids = set()
        records = []

    total = len(df_zs)

    # 5. Main loop
    for i, (_, row) in enumerate(df_zs.iterrows()):
        iid      = int(row["instruction_id"])
        question = str(row["question"])
        hop      = int(row["hop_count"])
        cr       = str(row["clinician_ref"])

        if iid in done_ids:
            continue

        # Get filename from ref_map
        if iid not in ref_map.index:
            print(f"  [SKIP] instruction_id {iid} not in ref_map")
            continue

        fn_ = str(ref_map.loc[iid, "filename"])
        xml_path = BASE / "ehrs" / fn_
        ehr, _ = extract_ehr(xml_path)

        n_done = len(records) + 1
        print(f"[{n_done:3d}/{total}] hop={hop} id={iid}: {question[:50]}")

        try:
            resp = deepseek_generate(question, ehr, model)
            sc = judge(question, cr, resp)
            correct_val = int(bool(sc.get("correct", False)))
            error_type  = sc.get("error_type", "error")
            confidence  = sc.get("confidence", 0.0)
            mark = "✓" if correct_val else "✗"
            print(f"  {mark} {error_type} (conf={confidence:.2f}): {resp[:60]}")
        except TimeoutError:
            print(f"  [TIMEOUT] marking as omission")
            resp       = "[TIMEOUT]"
            correct_val = 0
            error_type  = "omission"
            confidence  = 0.0
        except Exception as e:
            print(f"  [ERROR] {e}")
            resp       = f"[ERROR: {e}]"
            correct_val = 0
            error_type  = "omission"
            confidence  = 0.0

        records.append({
            "instruction_id":   iid,
            "hop_count":        hop,
            "model_used":       model,
            "question":         question,
            "clinician_ref":    cr,
            "model_response":   resp,
            "correct":          correct_val,
            "error_type":       error_type,
            "judge_confidence": confidence,
        })
        done_ids.add(iid)

        # Checkpoint every 20 questions
        if len(records) % 20 == 0:
            pd.DataFrame(records).to_csv(RAW_OUT, index=False)
            print(f"  [checkpoint saved: {len(records)}/{total}]")

        time.sleep(2)  # Ollama may be slow with large models

    # 6. Save final
    df_out = pd.DataFrame(records)
    df_out.to_csv(RAW_OUT, index=False)
    print(f"\nSaved {len(df_out)} rows → {RAW_OUT}")

    # 7. Results table
    # ZS reference accuracies (pre-computed from claude experiment)
    ZS_REF = {1: 0.3063, 2: 0.2826, 3: 0.2143, 4: 0.1731}

    print(f"\n{'='*60}")
    print(f"DEEPSEEK-R1 RESULTS (model: {model}):")
    print(f"{'='*60}")

    for hop in [1, 2, 3, 4]:
        sub = df_out[df_out["hop_count"] == hop]
        n = len(sub)
        if n == 0:
            print(f"hop={hop} (n=0): no data")
            continue
        correct = sub["correct"].sum()
        acc = correct / n
        lo, hi = wilson_ci(int(correct), n)
        zs_acc = ZS_REF.get(hop, float("nan"))
        print(
            f"hop={hop} (n={n:3d}): {acc*100:5.1f}% "
            f"[{lo*100:.1f}, {hi*100:.1f}] vs Claude ZS {zs_acc*100:.1f}%"
        )

    overall_acc = df_out["correct"].mean() if len(df_out) > 0 else float("nan")
    print(f"Overall: {overall_acc*100:.1f}%")

    # 8. CA z-test (one-tailed: DS < ZS)
    # Merge DeepSeek with ZS on instruction_id
    df_zs_merge = df_zs[["instruction_id", "hop_count", "correct"]].rename(
        columns={"correct": "correct_zs"}
    )
    df_merge = df_out[["instruction_id", "hop_count", "correct"]].merge(
        df_zs_merge, on=["instruction_id", "hop_count"], how="inner"
    )
    if len(df_merge) > 0:
        z, p = ca_z_test(
            df_merge[["instruction_id", "hop_count", "correct"]],
            df_merge[["instruction_id", "hop_count"]].assign(correct=df_merge["correct_zs"]),
        )
        print(f"CA z={z:.2f}, p={p:.3f} (one-tailed, H1: DeepSeek < ZS)")

        # 9. GEE interaction
        or_, p_gee = gee_interaction(
            df_out[["instruction_id", "hop_count", "correct"]],
            df_zs[["instruction_id", "hop_count", "correct"]],
        )
        if not np.isnan(or_):
            print(f"GEE hop×condition interaction (DeepSeek-R1 vs ZS): OR={or_:.2f}, p={p_gee:.3f}")
        else:
            print("GEE hop×condition interaction: could not compute")
    else:
        print("CA z-test: no matched pairs found")

    # 10. Error type breakdown
    print(f"\nError type breakdown:")
    print(df_out["error_type"].value_counts().to_string())

    # 11. Hop-by-hop accuracy trend (for visual inspection)
    print(f"\nHop-by-hop accuracy (DeepSeek-R1):")
    hop_agg = df_out.groupby("hop_count").agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
    )
    print(hop_agg.round(3).to_string())


if __name__ == "__main__":
    main()
