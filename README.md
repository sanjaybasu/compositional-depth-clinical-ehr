# Compositional Reasoning Depth Predicts Clinical AI Failure

Code and paper for: *Compositional Reasoning Depth Predicts Clinical AI Failure: Empirical Evidence Consistent with the Peng Impossibility Theorem in Electronic Health Record Question Answering*

Under review, 2026

---

## Overview

We introduce a **hop-count taxonomy** — the number of distinct inferential steps required to answer a clinical question from an EHR — as a principled, theoretically grounded predictor of clinical AI failure. Motivated by the Peng impossibility theorem for transformer self-attention, we annotate 313 clinician-generated EHR questions from MedAlign across four hop levels and evaluate accuracy under zero-shot and extended thinking conditions using `claude-sonnet-4-6`, with independent replication on `gpt-4o`.

**Key findings:**
- Monotone accuracy decline with hop count across both model families (Claude: OR per hop = 0.72, p = 0.008; GPT-4o: OR = 0.58, p < 0.001)
- Extended thinking does not significantly flatten the accuracy-depth curve at 3K, 16K, or explicit-CoT conditions (all interaction p > 0.05)
- Thinking-token usage scales with hop count (r = 0.31, p < 0.0001), consistent with O(k) computational requirement
- Structured decomposition converts hallucination to omission errors — a clinically favorable safety trade-off
- EHR truncation ruled out as confounder by answerability audit and 32K-character ablation

---

## Repository Structure

```
paper/                  LaTeX source, bibliography, compiled PDF
figures/                Publication-quality figures (PDF + PNG)
experiments/            Experiment scripts (numbered in execution order)
  01_hop_annotation.py         Hop-count annotation using LLM
  01b_hop_annotation_reliability.py  Inter-rater reliability
  06_main_experiment_local.py  Main experiment (ZS + ET conditions)
  07_analysis_and_figures.py   Statistical analysis and figure generation
  08_et16k_explicit_cot_rerun.py  ExplicitCoT and ET-16K conditions
results/
  analysis_stats.json          Aggregated statistics (no raw EHR content)
```

---

## Reproducing the Experiments

### Requirements

```bash
pip install anthropic openai statsmodels scipy numpy pandas matplotlib seaborn
```

### Data

This study uses the [MedAlign](https://arxiv.org/abs/2308.14089) dataset (Fleming et al., AAAI 2024), which is available with registration at PhysioNet ([physionet.org](https://physionet.org)). The hop-annotated question subset and derived evaluation results (without raw EHR content) will be released upon acceptance.

### Running the analysis

```bash
# Hop annotation
python experiments/01_hop_annotation.py

# Main experiment (requires ANTHROPIC_API_KEY)
python experiments/06_main_experiment_local.py

# GPT-4o replication (requires OPENAI_API_KEY)
# python experiments/04_sota_comparison.py

# Statistical analysis and figures
python experiments/07_analysis_and_figures.py
```

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{basu2026compositional,
  title={{Compositional Reasoning Depth Predicts Clinical AI Failure:
         Empirical Evidence Consistent with the Peng Impossibility Theorem
         in Electronic Health Record Question Answering}},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

---

## License

Code: MIT License. Data usage governed by the MedAlign data use agreement (Stanford STARR-OMOP).
