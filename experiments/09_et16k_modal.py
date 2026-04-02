"""
Experiment 9: ET-16K on Modal (detached, parallel)
====================================================
Runs ET-16K (budget_tokens=16000) on all 301 MedAlign questions in parallel
on Modal cloud infrastructure.  Uses --detach so the job persists even if the
local client disconnects.  Results are saved to a Modal Volume, then downloaded
with a separate download step.

Launch:
    cd /Users/sanjaybasu/waymark-local
    modal run --detach notebooks/ai-clinical-rtm/experiments/09_et16k_modal.py

Monitor:
    modal app list                              # shows et16k-neurips-2026 status
    modal volume ls et16k-results              # shows files saved

Download results after completion:
    modal run notebooks/ai-clinical-rtm/experiments/09_et16k_modal.py::download

Output:
    notebooks/ai-clinical-rtm/results/claude_et16k_raw.csv

NOTE: App 'et16k-neurips-2026' — does NOT touch rlm-medical or concept-tri apps.
"""

import json
import os
import sys
import time
import xml.etree.ElementTree as ET_xml
from pathlib import Path

import modal
import pandas as pd

# ── Local paths ────────────────────────────────────────────────────────────────
BASE_LOCAL  = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
                   "/medalign_instructions_v1_3")
ANNOT_LOCAL = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                   "/results/hop_annotations.csv")
ORIG_LOCAL  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                   "/results/claude_experiment_raw.csv")
OUT_CSV     = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm"
                   "/results/claude_et16k_raw.csv")
ENV_PATH    = "/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/.env"

REMOTE_OUT  = "/results/claude_et16k_raw.csv"   # path inside volume

# ── Modal app + volume ────────────────────────────────────────────────────────
app    = modal.App("et16k-neurips-2026")
volume = modal.Volume.from_name("et16k-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("anthropic>=0.40.0", "openai>=1.30.0", "pandas")
)

JUDGE_SYS = (
    "You are a clinical QA evaluator. Score model responses against clinician "
    "reference answers.\nReturn ONLY valid JSON — no other text:\n"
    "{\"correct\": true or false, \"error_type\": \"none\" | \"omission\" | "
    "\"hallucination\" | \"reasoning_error\", \"confidence\": 0.0-1.0}\n"
    "Definitions:\n"
    "- correct: model response substantially agrees with the clinician reference\n"
    "- omission: model failed to answer, refused, gave empty/irrelevant response\n"
    "- hallucination: model stated facts not in the EHR or contradicts medical knowledge\n"
    "- reasoning_error: model had the right facts but reached the wrong conclusion\n"
    "- confidence: your confidence in this scoring (0=unsure, 1=certain)"
)


# ── Remote: ET-16K question runner ────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    timeout=1800,
    single_use_containers=True,
)
def run_et16k(row: dict) -> dict:
    import anthropic, openai, os

    anth_client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = openai.OpenAI(api_key=row["openai_key"])

    iid, question, hop, ehr = (
        row["instruction_id"], row["question"], int(row["hop_count"]), row["ehr_text"]
    )
    cr = row["clinician_ref"]

    # ET-16K call with exponential backoff
    resp, think_tokens = "", 0
    delay = 60
    for attempt in range(8):
        try:
            r = anth_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=17000,
                thinking={"type": "enabled", "budget_tokens": 16000},
                messages=[{"role": "user", "content":
                    f"Patient EHR:\n{ehr}\n\nQuestion: {question}\n\nAnswer concisely."}]
            )
            think_tokens = int(sum(
                len(b.thinking.split()) * 1.3
                for b in r.content if hasattr(b, "thinking") and b.type == "thinking"
            ))
            text_blocks = [b for b in r.content if b.type == "text"]
            resp = text_blocks[0].text.strip() if text_blocks else ""
            break
        except anthropic.RateLimitError:
            if attempt == 7:
                return {**_base(row), "model_response": "", "correct": 0,
                        "error_type": "error", "judge_confidence": 0.0, "thinking_tokens": 0}
            time.sleep(delay)
            delay = min(delay * 2, 600)
        except Exception as e:
            return {**_base(row), "model_response": f"[API error: {e}]",
                    "correct": 0, "error_type": "error", "judge_confidence": 0.0,
                    "thinking_tokens": 0}

    # Judge
    try:
        jr  = openai_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=120,
            messages=[
                {"role": "system", "content": JUDGE_SYS},
                {"role": "user",   "content":
                    f"Question: {question}\nClinician reference: {cr}\n"
                    f"Model response: {resp}"}
            ]
        )
        raw = jr.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        sc = json.loads(raw)
    except Exception:
        sc = {"correct": False, "error_type": "error", "confidence": 0.0}

    return {
        **_base(row),
        "model_response":   resp,
        "correct":          int(sc.get("correct", False)),
        "error_type":       sc.get("error_type", "error"),
        "judge_confidence": sc.get("confidence", 0.0),
        "thinking_tokens":  think_tokens,
    }


def _base(row):
    return {
        "instruction_id":       row["instruction_id"],
        "condition":            "claude_et16k",
        "hop_count":            row["hop_count"],
        "ehr_char_count":       row["ehr_chars"],
        "question_token_count": row["question_token_count"],
        "domain":               row["domain"],
        "question":             row["question"],
        "clinician_ref":        row["clinician_ref"],
    }


# ── Orchestrator: runs on Modal, saves to volume ───────────────────────────────
@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=21600,   # 6 hr hard cap for the whole job
    secrets=[modal.Secret.from_name("anthropic-secret")],
)
def orchestrate(rows: list, openai_key: str):
    """
    Runs in Modal (not locally).  Dispatches batches of 20 in parallel,
    saves checkpoint CSV to volume after each batch.
    Survives local client disconnect because this function runs on Modal.
    """
    import pandas as pd

    BATCH_SIZE  = 20
    BATCH_SLEEP = 20

    records = []
    errors  = 0

    for b_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[b_start : b_start + BATCH_SIZE]
        print(f"[Batch {b_start//BATCH_SIZE + 1}/{(len(rows)+BATCH_SIZE-1)//BATCH_SIZE}]"
              f"  q {b_start+1}–{min(b_start+BATCH_SIZE, len(rows))}")

        # inject openai key into each row (not stored in CSV)
        for r in batch:
            r["openai_key"] = openai_key

        batch_results = list(run_et16k.map(
            batch,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        ))

        for i, res in enumerate(batch_results):
            gi = b_start + i + 1
            if isinstance(res, Exception):
                print(f"  ERROR [{gi}]: {type(res).__name__}: {res}")
                errors += 1
            else:
                mark = "✓" if res.get("correct") else "✗"
                print(f"  [{gi:3d}] {mark} hop={res['hop_count']}"
                      f"  {res.get('error_type','?'):18s}"
                      f"  think={res.get('thinking_tokens',0):5d}"
                      f"  {res['question'][:35]}")
                records.append({k: v for k, v in res.items() if k != "openai_key"})

        # Checkpoint to volume
        df_ckpt = pd.DataFrame(records)
        df_ckpt.to_csv(REMOTE_OUT, index=False)
        volume.commit()
        print(f"  [checkpoint: {len(records)} ok, {errors} errors → volume]")

        if b_start + BATCH_SIZE < len(rows):
            time.sleep(BATCH_SLEEP)

    # Final save
    df_out = pd.DataFrame(records)
    df_out.to_csv(REMOTE_OUT, index=False)
    volume.commit()
    print(f"\nDone. {len(df_out)} rows saved to volume.")
    if len(df_out) > 0:
        print(f"Accuracy: {df_out['correct'].mean():.3f}")
        print(df_out.groupby("hop_count")["correct"].agg(["mean","count"]))
        print("Mean thinking tokens:")
        print(df_out.groupby("hop_count")["thinking_tokens"].mean())


# ── EHR parsing (local) ───────────────────────────────────────────────────────
def extract_ehr(xml_path, max_chars=8000):
    try:
        tree = ET_xml.parse(xml_path)
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


# ── Local entrypoint: prepare data, call orchestrate ─────────────────────────
@app.local_entrypoint()
def main():
    def load_key(path, key):
        with open(path) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"')
        return ""

    openai_key = os.environ.get("OPENAI_API_KEY", "") or load_key(ENV_PATH, "OPENAI_API_KEY")
    if not openai_key:
        sys.exit("ERROR: OPENAI_API_KEY not found")

    annot     = pd.read_csv(ANNOT_LOCAL).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)
    orig      = pd.read_csv(ORIG_LOCAL)
    exp6_ids  = set(orig["instruction_id"].unique())

    df_resp = pd.read_csv(BASE_LOCAL / "clinician-reviewed-model-responses.tsv", sep="\t")
    ref_map = (df_resp.drop_duplicates("instruction_id")
               [["instruction_id", "filename", "clinician_response"]]
               .set_index("instruction_id"))

    sample = (annot[annot["instruction_id"].isin(exp6_ids)]
              .drop_duplicates(subset=["instruction_id"]).copy())
    print(f"Total unique questions: {len(sample)}")

    # Parse EHRs locally before sending to Modal
    print("Parsing EHR files locally...")
    rows = []
    for _, r in sample.iterrows():
        iid = r["instruction_id"]
        if iid not in ref_map.index:
            continue
        fn_        = str(ref_map.loc[iid, "filename"])
        ehr_text, ehr_chars = extract_ehr(BASE_LOCAL / "ehrs" / fn_)
        rows.append({
            "instruction_id":       iid,
            "question":             str(r["question"]),
            "hop_count":            int(r["hop_count"]),
            "ehr_text":             ehr_text,
            "ehr_chars":            ehr_chars,
            "question_token_count": len(str(r["question"]).split()),
            "clinician_ref":        str(ref_map.loc[iid, "clinician_response"]),
            "domain":               str(r.get("submitter_specialty", "")),
            "openai_key":           "",   # filled in orchestrate()
        })

    print(f"Dispatching {len(rows)} questions to Modal orchestrator...")
    print("Use 'modal run --detach' — orchestrator runs on Modal and survives client disconnect.")
    # orchestrate() runs on Modal; safe to disconnect locally after this returns
    orchestrate.remote(rows, openai_key)
    print("Orchestrator launched. Download results with:")
    print("  modal run notebooks/ai-clinical-rtm/experiments/09_et16k_modal.py::download")


def download_results():
    """
    Download completed results from Modal Volume to local CSV.
    Run directly: python3 09_et16k_modal.py download
    """
    import io
    vol = modal.Volume.from_name("et16k-results")
    try:
        data = b"".join(vol.read_file(REMOTE_OUT))
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        OUT_CSV.write_bytes(data)
        df = pd.read_csv(io.BytesIO(data))
        print(f"Downloaded {len(df)} rows → {OUT_CSV}")
        print(f"Accuracy: {df['correct'].mean():.3f}")
        print(df.groupby("hop_count")["correct"].agg(["mean","count"]))
    except Exception as e:
        print(f"Download failed: {e}")
        print("Check if job is still running: modal app list")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "download":
    download_results()
