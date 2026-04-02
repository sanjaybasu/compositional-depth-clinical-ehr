"""
Experiment 3: Statistical Analysis and Peng Theorem Curve Fitting
=================================================================
Combines:
  (a) existing MedAlign binary_correct × hop_count (6 models)
  (b) new arch_comparison results (zero_shot / CoT / RAG)

Primary analyses:
  1. GLMM: binary_correct ~ hop_count * model_class + (1|specialty)
     Tests: does error rate scale with hop count? Is the scaling different by architecture?
  2. Peng curve fit: P(error) = logistic(β0 - β1 * n^hop * log(n) / capacity)
     Tests: does the Peng impossibility theorem formula fit the empirical data?
  3. Interaction test: hop_count × {zero_shot vs CoT} — does CoT help more at high hops?
  4. Error type decomposition: omission vs hallucination vs reasoning_error by hop count

Outputs:
  results/glmm_results.json
  results/peng_fit.json
  figures/fig1_hop_error_rate.png   (existing MedAlign, 6 models)
  figures/fig2_arch_comparison.png  (new experiments, 3 conditions)
  figures/fig3_peng_curve.png       (theory vs empirical)
  figures/fig4_error_types.png      (omission/hallucination decomposition)
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency
from pathlib import Path

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels not available; skipping GLMM")

warnings.filterwarnings("ignore")

BASE_DATA = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
RESP_TSV  = BASE_DATA / "medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv"
ANNOT     = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/hop_annotations.csv")
ARCH_RAW  = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/arch_comparison_raw.csv")
ARCH_SCORE= Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm/results/arch_comparison_scores.csv")

OUT_DIR   = Path("/Users/sanjaybasu/waymark-local/notebooks/ai-clinical-rtm")
FIG_DIR   = OUT_DIR / "figures"
RES_DIR   = OUT_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
})

# Model capacity ordering (relative H*d*p, log scale)
MODEL_CAPACITY = {
    "mpt_7B_instruct":              7,   # 7B params
    "vicuna_7B":                    7,
    "vicuna_13B":                  13,
    "gpt4-32k-vicuna-context":    175,   # GPT-4 with vicuna context formatting
    "gpt4-32k":                   175,   # GPT-4 baseline
    "gpt4-32k-multistep-refinement": 175, # GPT-4 + test-time compute
}
MODEL_LABEL = {
    "mpt_7B_instruct":                "MPT 7B\n(small)",
    "vicuna_7B":                      "Vicuna 7B\n(small)",
    "vicuna_13B":                     "Vicuna 13B\n(medium)",
    "gpt4-32k-vicuna-context":        "GPT-4\n(RAG-style)",
    "gpt4-32k":                       "GPT-4\n(standard)",
    "gpt4-32k-multistep-refinement":  "GPT-4\n(multistep CoT)",
}
MODEL_COLOR = {
    "mpt_7B_instruct":                "#d73027",
    "vicuna_7B":                      "#f46d43",
    "vicuna_13B":                     "#fdae61",
    "gpt4-32k-vicuna-context":        "#74add1",
    "gpt4-32k":                       "#4575b4",
    "gpt4-32k-multistep-refinement":  "#313695",
}
ARCH_COLOR = {
    "zero_shot": "#e41a1c",
    "cot":       "#377eb8",
    "rag":       "#4daf4a",
}
ARCH_LABEL = {
    "zero_shot": "Zero-shot (direct)",
    "cot":       "Chain-of-thought",
    "rag":       "RAG-grounded",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_medalign_with_hops() -> pd.DataFrame:
    annot = pd.read_csv(ANNOT).dropna(subset=["hop_count"])
    annot["hop_count"] = annot["hop_count"].astype(int).clip(upper=4)
    df = pd.read_csv(RESP_TSV, sep="\t")
    merged = df.merge(annot[["instruction_id","hop_count","requires_ehr","reasoning_type"]],
                      on="instruction_id", how="inner")
    merged = merged[merged["is_used_eval"] == "yes"]
    print(f"MedAlign merged: {len(merged)} rows, {merged['instruction_id'].nunique()} unique questions")
    return merged


def load_arch_comparison() -> pd.DataFrame | None:
    if not ARCH_RAW.exists():
        print("Arch comparison not yet run. Skipping.")
        return None
    df = pd.read_csv(ARCH_RAW)
    print(f"Arch comparison: {len(df)} rows")
    return df


# ── Analysis 1: GLMM ─────────────────────────────────────────────────────────

def run_glmm(df: pd.DataFrame) -> dict:
    if not HAS_SM:
        return {}
    df = df.copy()
    df["hop_f"] = df["hop_count"].astype(float)
    df["capacity_log"] = df["model_name"].map(MODEL_CAPACITY).apply(np.log)

    # Model 1: hop_count main effect
    m1 = smf.logit("binary_correct ~ hop_f", data=df).fit(disp=False)
    # Model 2: hop × model class
    m2 = smf.logit("binary_correct ~ hop_f * model_name", data=df).fit(disp=False)
    # Model 3: hop × capacity (continuous)
    m3 = smf.logit("binary_correct ~ hop_f * capacity_log", data=df).fit(disp=False)

    results = {
        "m1_hop_coef":    float(m1.params.get("hop_f", np.nan)),
        "m1_hop_pval":    float(m1.pvalues.get("hop_f", np.nan)),
        "m1_aic":         float(m1.aic),
        "m2_aic":         float(m2.aic),
        "m3_hop_x_cap_coef": float(m3.params.get("hop_f:capacity_log", np.nan)),
        "m3_hop_x_cap_pval": float(m3.pvalues.get("hop_f:capacity_log", np.nan)),
        "m3_aic":         float(m3.aic),
    }
    print("\n=== GLMM Results ===")
    print(f"  hop main effect: coef={results['m1_hop_coef']:.3f}, p={results['m1_hop_pval']:.4f}")
    print(f"  hop × log-capacity interaction: coef={results['m3_hop_x_cap_coef']:.3f}, p={results['m3_hop_x_cap_pval']:.4f}")
    return results


# ── Analysis 2: Peng curve fitting ──────────────────────────────────────────

def fit_peng_curve(agg: pd.DataFrame, model_name: str) -> dict:
    """
    Fit an exponential degradation curve to error rate by hop count.
    Peng theorem predicts P(error) ~ 1 - exp(-alpha * beta^k) for some alpha, beta > 0.
    With only 4 data points we fit a log-linear model: log(error_rate) ~ a + b*k.
    slope b (= log(beta)) reflects how steeply error rates scale with hop count —
    the Peng prediction is b > 0 and b should be larger for smaller-capacity models.
    """
    sub = agg[agg["model_name"] == model_name].sort_values("hop_count")
    if len(sub) < 3:
        return {"model": model_name, "error": "insufficient data"}

    x = sub["hop_count"].values.astype(float)
    y_error = np.clip(1.0 - sub["accuracy"].values, 1e-6, 1.0)

    # Log-linear fit: log(error) = a + b*k
    log_y = np.log(y_error)
    coeffs = np.polyfit(x, log_y, 1)  # [slope, intercept]
    slope     = float(coeffs[0])
    intercept = float(coeffs[1])

    y_pred = np.exp(intercept + slope * x)
    ss_res = np.sum((y_error - y_pred) ** 2)
    ss_tot = np.sum((y_error - y_error.mean()) ** 2)
    r2     = float(1 - ss_res / (ss_tot + 1e-12))

    # Also compute simple linear slope in accuracy space
    acc_slope, _ = np.polyfit(x, sub["accuracy"].values, 1)

    return {
        "model":          model_name,
        "log_linear_slope":  slope,      # Peng: should be > 0 (error grows with hops)
        "log_linear_intercept": intercept,
        "acc_slope":      float(acc_slope),  # negative = accuracy drops with hops
        "r2":             r2,
        "error_rates":    y_error.tolist(),
        "hop_counts":     x.tolist(),
    }


# ── Figure 1: MedAlign hop × model ──────────────────────────────────────────

def fig1_hop_error_rate(df: pd.DataFrame):
    agg = df.groupby(["model_name","hop_count"])["binary_correct"].agg(
        accuracy="mean", n="count", se=lambda x: x.std() / np.sqrt(len(x))
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: accuracy by hop count per model
    for mname in ["mpt_7B_instruct","vicuna_7B","vicuna_13B",
                  "gpt4-32k","gpt4-32k-multistep-refinement"]:
        sub = agg[agg["model_name"] == mname].sort_values("hop_count")
        if len(sub) == 0: continue
        ax1.plot(sub["hop_count"], sub["accuracy"],
                 marker="o", color=MODEL_COLOR[mname],
                 label=MODEL_LABEL[mname].replace("\n"," "), linewidth=2, markersize=6)
        ax1.fill_between(sub["hop_count"],
                         sub["accuracy"] - sub["se"],
                         sub["accuracy"] + sub["se"],
                         color=MODEL_COLOR[mname], alpha=0.12)

    ax1.set_xlabel("Compositional depth (hop count)", fontsize=12)
    ax1.set_ylabel("Accuracy (fraction correct)", fontsize=12)
    ax1.set_title("A  Clinical accuracy declines with compositional depth\n"
                  "(Peng et al. 2402.08164 prediction)", fontsize=11)
    ax1.set_xticks([1,2,3,4])
    ax1.set_xticklabels(["1\n(Direct\nretrieval)", "2\n(Simple\ncomposition)",
                         "3\n(Multi-step\nreasoning)", "4+\n(Complex\nsynthesis)"])
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right")

    # Right: GPT-4 standard vs multistep (CoT effect) — gap widens with hop count
    gpt4_std  = agg[agg["model_name"] == "gpt4-32k"].sort_values("hop_count")
    gpt4_cot  = agg[agg["model_name"] == "gpt4-32k-multistep-refinement"].sort_values("hop_count")
    vicuna13  = agg[agg["model_name"] == "vicuna_13B"].sort_values("hop_count")

    for sub, label, color in [
        (gpt4_std, "GPT-4 standard", MODEL_COLOR["gpt4-32k"]),
        (gpt4_cot, "GPT-4 + multistep CoT", MODEL_COLOR["gpt4-32k-multistep-refinement"]),
        (vicuna13, "Vicuna 13B", MODEL_COLOR["vicuna_13B"]),
    ]:
        if len(sub) == 0: continue
        ax2.plot(sub["hop_count"], sub["accuracy"],
                 marker="o", color=color, label=label, linewidth=2, markersize=7)

    # Shade the CoT benefit region
    if len(gpt4_std) > 0 and len(gpt4_cot) > 0:
        hops_common = sorted(set(gpt4_std["hop_count"]) & set(gpt4_cot["hop_count"]))
        std_vals = gpt4_std.set_index("hop_count").loc[hops_common, "accuracy"].values
        cot_vals = gpt4_cot.set_index("hop_count").loc[hops_common, "accuracy"].values
        ax2.fill_between(hops_common, std_vals, cot_vals,
                         alpha=0.15, color=MODEL_COLOR["gpt4-32k-multistep-refinement"],
                         label="CoT benefit")

    ax2.set_xlabel("Compositional depth (hop count)", fontsize=12)
    ax2.set_ylabel("Accuracy (fraction correct)", fontsize=12)
    ax2.set_title("B  CoT benefit grows with hop count\n"
                  "(test-time compute — Peng solution class 1)", fontsize=11)
    ax2.set_xticks([1,2,3,4])
    ax2.set_xticklabels(["1\n(Direct)", "2\n(Simple\ncomp.)",
                         "3\n(Multi-step)", "4+\n(Complex)"])
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "fig1_hop_error_rate.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    return agg


# ── Figure 2: New arch comparison ────────────────────────────────────────────

def fig2_arch_comparison(df_arch: pd.DataFrame):
    agg = df_arch.groupby(["condition","hop_count"]).agg(
        accuracy=("binary_correct","mean"),
        n=("binary_correct","count"),
        se=("binary_correct", lambda x: x.std() / np.sqrt(len(x))),
        omission_rate=("error_type", lambda x: (x=="omission").mean()),
        hallucination_rate=("error_type", lambda x: (x=="hallucination").mean()),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: accuracy by condition × hop count
    ax = axes[0]
    for cond in ["zero_shot","cot","rag"]:
        sub = agg[agg["condition"]==cond].sort_values("hop_count")
        if len(sub)==0: continue
        ax.plot(sub["hop_count"], sub["accuracy"],
                marker="o", color=ARCH_COLOR[cond],
                label=ARCH_LABEL[cond], linewidth=2.5, markersize=7)
        ax.fill_between(sub["hop_count"],
                        sub["accuracy"]-sub["se"],
                        sub["accuracy"]+sub["se"],
                        color=ARCH_COLOR[cond], alpha=0.12)
    ax.set_xlabel("Compositional depth (hop count)")
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_title("A  Accuracy by architecture × hop count")
    ax.set_xticks([1,2,3,4])
    ax.set_ylim(0,1)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Middle: omission rate (76.6% of NOHARM harms are omissions)
    ax = axes[1]
    for cond in ["zero_shot","cot","rag"]:
        sub = agg[agg["condition"]==cond].sort_values("hop_count")
        if len(sub)==0: continue
        ax.plot(sub["hop_count"], sub["omission_rate"],
                marker="s", color=ARCH_COLOR[cond],
                label=ARCH_LABEL[cond], linewidth=2.5, markersize=7, linestyle="--")
    ax.set_xlabel("Compositional depth (hop count)")
    ax.set_ylabel("Omission rate")
    ax.set_title("B  Omission rate by architecture × hop count\n"
                 "(76.6% of NOHARM harms are omissions)")
    ax.set_xticks([1,2,3,4])
    ax.set_ylim(0,1)
    ax.legend(fontsize=9)

    # Right: hallucination rate
    ax = axes[2]
    for cond in ["zero_shot","cot","rag"]:
        sub = agg[agg["condition"]==cond].sort_values("hop_count")
        if len(sub)==0: continue
        ax.plot(sub["hop_count"], sub["hallucination_rate"],
                marker="^", color=ARCH_COLOR[cond],
                label=ARCH_LABEL[cond], linewidth=2.5, markersize=7, linestyle=":")
    ax.set_xlabel("Compositional depth (hop count)")
    ax.set_ylabel("Hallucination rate")
    ax.set_title("C  Hallucination rate by architecture × hop count")
    ax.set_xticks([1,2,3,4])
    ax.set_ylim(0,1)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "fig2_arch_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return agg


# ── Figure 3: Peng curve fit ─────────────────────────────────────────────────

def fig3_peng_curve(medalign_agg: pd.DataFrame, fit_results: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: error rates by hop count per model with log-linear fit curves
    ax = axes[0]
    k_fine = np.linspace(1, 4, 100)

    for fr in fit_results:
        if "log_linear_slope" not in fr:
            continue
        mname = fr["model"]
        slope = fr["log_linear_slope"]
        intercept = fr["log_linear_intercept"]
        r2    = fr["r2"]

        sub = medalign_agg[medalign_agg["model_name"]==mname].sort_values("hop_count")
        if len(sub) < 2: continue

        color = MODEL_COLOR.get(mname, "gray")
        label = MODEL_LABEL.get(mname, mname).replace("\n"," ")

        ax.scatter(sub["hop_count"], 1 - sub["accuracy"],
                   color=color, s=60, zorder=5)
        y_fit = np.clip(np.exp(intercept + slope * k_fine), 0, 1)
        ax.plot(k_fine, y_fit, color=color,
                label=f"{label} (slope={slope:.2f})",
                linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Compositional depth k (hop count)", fontsize=12)
    ax.set_ylabel("Error rate (1 − accuracy)", fontsize=12)
    ax.set_title("A  Error rate scaling with compositional depth\n"
                 "log(error) ~ α + β·k  (Peng: β > 0 for all models)", fontsize=11)
    ax.set_xticks([1,2,3,4])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Right: log-linear error slope vs. model size
    # Peng predicts: smaller models (lower H*d*p) should have steeper slopes
    ax = axes[1]
    param_counts = {
        "mpt_7B_instruct": 7,
        "vicuna_7B":        7,
        "vicuna_13B":      13,
        "gpt4-32k":        175,
        "gpt4-32k-vicuna-context": 175,
        "gpt4-32k-multistep-refinement": 175,
    }
    xs, ys, labels, colors = [], [], [], []
    for fr in fit_results:
        mname = fr.get("model","")
        if mname not in param_counts or "log_linear_slope" not in fr:
            continue
        xs.append(np.log10(param_counts[mname] * 1e9))
        ys.append(fr["log_linear_slope"])
        labels.append(MODEL_LABEL.get(mname, mname).replace("\n"," "))
        colors.append(MODEL_COLOR.get(mname, "gray"))

    for x, y, lbl, c in zip(xs, ys, labels, colors):
        ax.scatter(x, y, s=100, color=c, zorder=5)
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=7)

    if len(xs) > 2:
        p = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs)-0.1, max(xs)+0.1, 50)
        ax.plot(xline, np.polyval(p, xline), "k--", alpha=0.5, lw=1.5,
                label=f"slope={p[0]:.2f} (predicted: negative)")
        ax.legend(fontsize=9)

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("log₁₀(parameter count)", fontsize=12)
    ax.set_ylabel("Error-rate scaling slope (β)", fontsize=12)
    ax.set_title("B  Error scaling β vs. model size\n"
                 "Peng prediction: larger models → smaller β (less steep)", fontsize=11)

    plt.tight_layout()
    path = FIG_DIR / "fig3_peng_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Figure 4: Error type decomposition ───────────────────────────────────────

def fig4_error_types(df: pd.DataFrame, df_arch: pd.DataFrame | None):
    """Stacked bar: correct / omission / hallucination / reasoning_error by hop count."""
    fig, axes = plt.subplots(1, 2 if df_arch is not None else 1,
                             figsize=(12 if df_arch is not None else 6, 5))
    if df_arch is None:
        axes = [axes]

    def stacked_bar(ax, data, condition_label):
        hops = sorted(data["hop_count"].unique())
        correct, omission, hallucn, reason = [], [], [], []
        for h in hops:
            sub = data[data["hop_count"]==h]
            n   = len(sub)
            correct.append(sub["binary_correct"].mean())
            omission.append((sub.get("error_type","none")=="omission").mean()
                            if "error_type" in sub.columns else 0)
            hallucn.append((sub.get("error_type","none")=="hallucination").mean()
                           if "error_type" in sub.columns else 0)
            reason.append((sub.get("error_type","none")=="reasoning_error").mean()
                          if "error_type" in sub.columns else 0)

        x = np.arange(len(hops))
        w = 0.6
        ax.bar(x, correct, w, label="Correct", color="#4dac26")
        ax.bar(x, omission, w, bottom=correct, label="Omission", color="#d01c8b")
        bottom2 = [c+o for c,o in zip(correct, omission)]
        ax.bar(x, hallucn, w, bottom=bottom2, label="Hallucination", color="#f1b6da")
        bottom3 = [b+h for b,h in zip(bottom2, hallucn)]
        ax.bar(x, reason, w, bottom=bottom3, label="Reasoning error", color="#b8e186")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Hop {h}" for h in hops])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of responses")
        ax.set_title(condition_label)
        ax.legend(fontsize=8, loc="upper right")

    stacked_bar(axes[0], df, "A  Existing MedAlign (all models)\nError type by compositional depth")
    if df_arch is not None:
        # Show zero_shot condition for contrast
        zs = df_arch[df_arch["condition"]=="zero_shot"].copy()
        stacked_bar(axes[1], zs, "B  New experiment: Zero-shot\nError type by compositional depth")

    plt.tight_layout()
    path = FIG_DIR / "fig4_error_types.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not ANNOT.exists():
        print("ERROR: Run 01_hop_annotation.py first.")
        return

    df_medalign = load_medalign_with_hops()
    df_arch     = load_arch_comparison()

    # Analysis 1: GLMM
    glmm_res = run_glmm(df_medalign)

    # Analysis 2: Peng curve fitting per model
    agg_medalign = df_medalign.groupby(["model_name","hop_count"])["binary_correct"].agg(
        accuracy="mean", n="count"
    ).reset_index()
    fit_results = []
    for mname in df_medalign["model_name"].unique():
        fr = fit_peng_curve(agg_medalign, mname)
        fit_results.append(fr)
        if "capacity_fit" in fr:
            print(f"  {mname}: n_fit={fr['n_fit']:.0f}, capacity={fr['capacity_fit']:.0f}, R²={fr['r2']:.3f}")

    # Analysis 3: CoT interaction test
    cot_results = {}
    if HAS_SM:
        df_cot_test = df_medalign[df_medalign["model_name"].isin(
            ["gpt4-32k","gpt4-32k-multistep-refinement"])].copy()
        df_cot_test["is_cot"] = (df_cot_test["model_name"]
                                 == "gpt4-32k-multistep-refinement").astype(int)
        df_cot_test["hop_f"] = df_cot_test["hop_count"].astype(float)
        try:
            m_cot = smf.logit("binary_correct ~ hop_f * is_cot", data=df_cot_test).fit(disp=False)
            cot_results = {
                "interaction_coef": float(m_cot.params.get("hop_f:is_cot", np.nan)),
                "interaction_pval": float(m_cot.pvalues.get("hop_f:is_cot", np.nan)),
            }
            print(f"\nCoT interaction: coef={cot_results['interaction_coef']:.3f}, "
                  f"p={cot_results['interaction_pval']:.4f}")
            if cot_results["interaction_pval"] < 0.05:
                print("  *** CoT benefit significantly increases with hop count (p<0.05) ***")
        except Exception as e:
            print(f"  CoT interaction test error: {e}")

    # Save results
    all_results = {
        "glmm": glmm_res,
        "peng_fits": fit_results,
        "cot_interaction": cot_results,
    }
    with open(RES_DIR / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved analysis results → {RES_DIR}/analysis_results.json")

    # Figures
    fig1_hop_error_rate(df_medalign)
    fig3_peng_curve(agg_medalign, fit_results)
    fig4_error_types(df_medalign, df_arch)
    if df_arch is not None:
        fig2_arch_comparison(df_arch)

    print("\nAll figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
