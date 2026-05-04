"""
Comprehensive statistical analysis: Self-Learning vs Social-Learning
across all experimental configurations.
"""

import os
import json
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, shapiro

BASE_EXP = "/Users/nyuad/Desktop/capstone/experiments"
BASE_OLD = "/Users/nyuad/Desktop/capstone/old"


def load_final_scores(exp_dir: str) -> list[float]:
    """Load per-run mean final agent score from an experiment directory."""
    run_scores = []
    if not os.path.isdir(exp_dir):
        return run_scores
    for run in sorted(os.listdir(exp_dir)):
        if not run.startswith("run_"):
            continue
        exp_file = os.path.join(exp_dir, run, "experiment.json")
        if not os.path.exists(exp_file):
            continue
        with open(exp_file) as f:
            data = json.load(f)
        agents = data.get("agents", {})
        agent_finals = []
        for agent_id, agent_data in agents.items():
            scores = agent_data.get("scores", {})
            if scores:
                last_key = max(scores.keys(), key=lambda k: int(k))
                agent_finals.append(scores[last_key])
        if agent_finals:
            run_scores.append(np.mean(agent_finals))
    return run_scores


def load_metadata(exp_dir: str) -> dict:
    """Load metadata from the first available run."""
    if not os.path.isdir(exp_dir):
        return {}
    for run in sorted(os.listdir(exp_dir)):
        if not run.startswith("run_"):
            continue
        meta_file = os.path.join(exp_dir, run, "metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                return json.load(f)
    return {}


def stat_test(self_scores: list, social_scores: list) -> dict:
    """Run statistical tests comparing two groups."""
    if len(self_scores) < 2 or len(social_scores) < 2:
        return {"error": "insufficient data"}

    s = np.array(self_scores)
    so = np.array(social_scores)

    # Normality (Shapiro-Wilk, only if n >= 3)
    try:
        sw_self = shapiro(s).pvalue if len(s) >= 3 else None
        sw_social = shapiro(so).pvalue if len(so) >= 3 else None
    except Exception:
        sw_self = sw_social = None

    # Welch's t-test
    t_stat, t_p = ttest_ind(s, so, equal_var=False)

    # Mann-Whitney U (non-parametric)
    try:
        mw_u, mw_p = mannwhitneyu(s, so, alternative="two-sided")
    except Exception:
        mw_u, mw_p = np.nan, np.nan

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.std(s, ddof=1) ** 2 + np.std(so, ddof=1) ** 2) / 2)
    cohens_d = (np.mean(s) - np.mean(so)) / pooled_std if pooled_std > 0 else 0.0

    # Effect size: rank-biserial correlation (for Mann-Whitney)
    n1, n2 = len(s), len(so)
    r_rb = 1 - (2 * mw_u) / (n1 * n2) if not np.isnan(mw_u) else np.nan

    winner = (
        "self"
        if np.mean(s) > np.mean(so)
        else ("social" if np.mean(so) > np.mean(s) else "tie")
    )

    return {
        "self_mean": np.mean(s),
        "self_std": np.std(s, ddof=1),
        "self_median": np.median(s),
        "self_n": n1,
        "social_mean": np.mean(so),
        "social_std": np.std(so, ddof=1),
        "social_median": np.median(so),
        "social_n": n2,
        "t_stat": t_stat,
        "t_p": t_p,
        "mw_u": mw_u,
        "mw_p": mw_p,
        "cohens_d": cohens_d,
        "r_rb": r_rb,
        "sw_self_p": sw_self,
        "sw_social_p": sw_social,
        "winner": winner,
    }


def significance_label(p: float) -> str:
    if np.isnan(p):
        return "?"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ── Configuration registry ──────────────────────────────────────────────────
# Each entry: (label, category, info_ratio, n_agents, self_dir, social_dir)
CONFIGS = [
    # Baseline
    (
        "Baseline",
        "Baseline",
        "N/A",
        10,
        f"{BASE_EXP}/baseline_self",
        f"{BASE_EXP}/baseline_social",
    ),
    # Old exponential swap pool (20%)
    (
        "Old Exp Swap 20%",
        "Exp Swap (old)",
        "20%",
        10,
        f"{BASE_OLD}/exponential_swap_pool_self",
        f"{BASE_OLD}/exponential_swap_pool_social",
    ),
    # Old exponential swap pool (50%)
    (
        "Old Exp Swap 50%",
        "Exp Swap (old)",
        "50%",
        10,
        f"{BASE_OLD}/exponential_swap_pool_self_50",
        f"{BASE_OLD}/exponential_swap_pool_social_50",
    ),
    # Old exponential swap pool (80%)
    (
        "Old Exp Swap 80%",
        "Exp Swap (old)",
        "80%",
        10,
        f"{BASE_OLD}/exponential_swap_pool_self_80",
        f"{BASE_OLD}/exponential_swap_pool_social_80",
    ),
    # Exponential one-time pool (20%, 10 agents)
    (
        "Exp One-Time 20%",
        "Exp One-Time",
        "20%",
        10,
        f"{BASE_EXP}/exponential_one_time_pool_self",
        f"{BASE_EXP}/exponential_one_time_pool_social",
    ),
    # Exponential one-time pool (10%, 10 agents)
    (
        "Exp One-Time 10%",
        "Exp One-Time",
        "10%",
        10,
        f"{BASE_EXP}/exponential_one_time_pool_self_10_percent",
        f"{BASE_EXP}/exponential_one_time_pool_social_10_percent",
    ),
    # Exponential one-time pool (10%, 25 agents)
    (
        "Exp One-Time 10% (25 ag)",
        "Exp One-Time",
        "10%",
        25,
        f"{BASE_EXP}/exponential_one_time_pool_self_10_percent_25_agents",
        f"{BASE_EXP}/exponential_one_time_pool_social_10_percent_25_agents",
    ),
    # Exponential one-time pool (30%, 25 agents)
    (
        "Exp One-Time 30% (25 ag)",
        "Exp One-Time",
        "30%",
        25,
        f"{BASE_EXP}/exponential_one_time_pool_self_30_percent_25_agents",
        f"{BASE_EXP}/exponential_one_time_pool_social_30_percent_25_agents",
    ),
    # Decay
    (
        "Decay",
        "Decay",
        "75% vis.",
        25,
        f"{BASE_EXP}/decay_self_learning",
        f"{BASE_EXP}/decay_social_learning",
    ),
    # Movement
    (
        "Movement",
        "Movement",
        "N/A",
        10,
        f"{BASE_EXP}/movement_self_learning",
        f"{BASE_EXP}/movement_social_learning",
    ),
    # Dynamic pool (10 agents)
    (
        "Dynamic Pool",
        "Dynamic Pool",
        "~50% dyn",
        10,
        f"{BASE_EXP}/dynamic_pool_self_10",
        f"{BASE_EXP}/dynamic_pool_social_10",
    ),
    # Constant ratio pool (20%)
    (
        "Constant Ratio 20%",
        "Constant Ratio",
        "20%",
        10,
        f"{BASE_EXP}/constant_ratio_pool_self",
        f"{BASE_EXP}/constant_ratio_pool_social",
    ),
]

# ── Run analysis ─────────────────────────────────────────────────────────────

results = []
for label, category, info_ratio, n_agents, self_dir, social_dir in CONFIGS:
    self_scores = load_final_scores(self_dir)
    social_scores = load_final_scores(social_dir)
    meta = load_metadata(self_dir)

    ground_truth_snippets = meta.get("ground_truth_snippet_count", "?")
    num_subjects = meta.get("num_subject_agents", "?")

    res = stat_test(self_scores, social_scores)
    res.update(
        {
            "label": label,
            "category": category,
            "info_ratio": info_ratio,
            "n_agents": n_agents,
            "gt_snippets": ground_truth_snippets,
            "num_subjects": num_subjects,
            "self_runs": len(self_scores),
            "social_runs": len(social_scores),
        }
    )
    results.append(res)


# ── Pretty-print table ────────────────────────────────────────────────────────

SEP = "-" * 185

header = (
    f"{'Config':<28} {'Category':<16} {'Info%':<9} {'Agents':<7} {'Subj':<5} "
    f"{'n_S':<4} {'n_So':<4} "
    f"{'Self μ':<8} {'Self σ':<8} "
    f"{'Soc μ':<8} {'Soc σ':<8} "
    f"{'Δμ':<8} "
    f"{'t-stat':<9} {'t-p':<8} {'Sig':<5} "
    f"{'MW-p':<8} {'MW-Sig':<7} "
    f"{'Cohen d':<9} {'r_rb':<7} "
    f"{'Winner':<8}"
)

print("\n" + "=" * 185)
print("COMPREHENSIVE STATISTICAL ANALYSIS: Self-Learning vs Social-Learning")
print("Metric: final mean cosine-bm25 score per run (averaged over knowledge agents)")
print("Tests: Welch's t-test (parametric) + Mann-Whitney U (non-parametric, two-sided)")
print("Effect: Cohen's d (standardised mean diff) | r_rb (rank-biserial correlation)")
print("Sig: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
print("=" * 185)
print(header)
print(SEP)

for r in results:
    if "error" in r:
        print(f"{r['label']:<28}  {'INSUFFICIENT DATA':}")
        continue

    delta = r["self_mean"] - r["social_mean"]
    sig_t = significance_label(r["t_p"])
    sig_mw = significance_label(r["mw_p"])

    row = (
        f"{r['label']:<28} {r['category']:<16} {r['info_ratio']:<9} {r['n_agents']:<7} "
        f"{str(r['num_subjects']):<5} "
        f"{r['self_runs']:<4} {r['social_runs']:<4} "
        f"{r['self_mean']:>7.4f}  {r['self_std']:>7.4f}  "
        f"{r['social_mean']:>7.4f}  {r['social_std']:>7.4f}  "
        f"{delta:>+7.4f}  "
        f"{r['t_stat']:>8.3f}  {r['t_p']:>7.4f}  {sig_t:<5} "
        f"{r['mw_p']:>7.4f}  {sig_mw:<7} "
        f"{r['cohens_d']:>8.4f}  {r['r_rb']:>6.4f}  "
        f"{r['winner']:<8}"
    )
    print(row)

print(SEP)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nSUMMARY")
print("-" * 60)
for r in results:
    if "error" in r:
        continue
    sig_t = significance_label(r["t_p"])
    sig_mw = significance_label(r["mw_p"])
    delta = r["self_mean"] - r["social_mean"]
    mag = abs(r["cohens_d"])
    size_label = (
        "large" if mag >= 0.8 else "medium" if mag >= 0.5 else "small" if mag >= 0.2 else "negligible"
    )
    print(
        f"  {r['label']:<30}  winner={r['winner']:<7}  Δμ={delta:+.4f}  "
        f"t-sig={sig_t:<4}  MW-sig={sig_mw:<4}  effect={size_label} (d={r['cohens_d']:.3f})"
    )

print("\nNormality check (Shapiro-Wilk p-values; p<0.05 suggests non-normal distribution):")
print("-" * 60)
for r in results:
    if "error" in r:
        continue
    sw_s = f"{r['sw_self_p']:.4f}" if r["sw_self_p"] is not None else "N/A"
    sw_so = f"{r['sw_social_p']:.4f}" if r["sw_social_p"] is not None else "N/A"
    print(f"  {r['label']:<30}  SW-self={sw_s:<8}  SW-social={sw_so:<8}")
