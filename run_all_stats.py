"""
run_all_stats.py
~~~~~~~~~~~~~~~~
Run comprehensive statistical tests (endpoint, AUC, time-to-threshold) across
ALL self-vs-social experiment config pairs and print a unified summary table.

Usage
-----
    uv run python run_all_stats.py [--alpha 0.05] [--thresholds 0.5 0.8] [--output results.json]

The NLI model is loaded once and shared across all config pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from process_experiment import _load_nli_model
from statistical_tests import perform_statistical_tests, print_stats_report

BASE = Path(__file__).parent
EXP = BASE / "experiments"
OLD = BASE / "old"

# ---------------------------------------------------------------------------
# Config registry
# (label, category, info_ratio_str, n_agents, n_subjects, self_dir, social_dir)
# ---------------------------------------------------------------------------
CONFIG_PAIRS = [
    # ── Baseline ──────────────────────────────────────────────────────────
    (
        "Baseline",
        "Baseline",
        "100%",
        10,
        10,
        EXP / "baseline_self",
        EXP / "baseline_social",
    ),
    # ── Old exponential swap pool ──────────────────────────────────────────
    (
        "Old Exp Swap 20%",
        "Exp Swap (old)",
        "20%",
        10,
        10,
        OLD / "exponential_swap_pool_self",
        OLD / "exponential_swap_pool_social",
    ),
    (
        "Old Exp Swap 50%",
        "Exp Swap (old)",
        "50%",
        10,
        10,
        OLD / "exponential_swap_pool_self_50",
        OLD / "exponential_swap_pool_social_50",
    ),
    (
        "Old Exp Swap 80%",
        "Exp Swap (old)",
        "80%",
        10,
        10,
        OLD / "exponential_swap_pool_self_80",
        OLD / "exponential_swap_pool_social_80",
    ),
    # ── Exponential one-time pool ──────────────────────────────────────────
    (
        "Exp One-Time 20%",
        "Exp One-Time",
        "20%",
        10,
        10,
        EXP / "exponential_one_time_pool_self",
        EXP / "exponential_one_time_pool_social",
    ),
    (
        "Exp One-Time 10%",
        "Exp One-Time",
        "10%",
        10,
        10,
        EXP / "exponential_one_time_pool_self_10_percent",
        EXP / "exponential_one_time_pool_social_10_percent",
    ),
    (
        "Exp One-Time 10% (25ag)",
        "Exp One-Time",
        "10%",
        25,
        10,
        EXP / "exponential_one_time_pool_self_10_percent_25_agents",
        EXP / "exponential_one_time_pool_social_10_percent_25_agents",
    ),
    (
        "Exp One-Time 30% (25ag)",
        "Exp One-Time",
        "30%",
        25,
        10,
        EXP / "exponential_one_time_pool_self_30_percent_25_agents",
        EXP / "exponential_one_time_pool_social_30_percent_25_agents",
    ),
    # ── Decay ─────────────────────────────────────────────────────────────
    (
        "Decay",
        "Decay",
        "75% vis.",
        25,
        40,
        EXP / "decay_self_learning",
        EXP / "decay_social_learning",
    ),
    # ── Movement ──────────────────────────────────────────────────────────
    (
        "Movement",
        "Movement",
        "100%",
        10,
        10,
        EXP / "movement_self_learning",
        EXP / "movement_social_learning",
    ),
    # ── Dynamic pool ──────────────────────────────────────────────────────
    (
        "Dynamic Pool",
        "Dynamic Pool",
        "~50% dyn",
        10,
        10,
        EXP / "dynamic_pool_self_10",
        EXP / "dynamic_pool_social_10",
    ),
    # ── Constant ratio pool ────────────────────────────────────────────────
    (
        "Constant Ratio 20%",
        "Const. Ratio",
        "20%",
        10,
        10,
        EXP / "constant_ratio_pool_self",
        EXP / "constant_ratio_pool_social",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(p: float | None, alpha: float) -> str:
    if p is None:
        return "n/a"
    if p < alpha / 50:
        return "***"
    if p < alpha / 5:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


def _fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "n/a"


def _winner(self_val: float, social_val: float) -> str:
    if self_val > social_val:
        return "SELF"
    if social_val > self_val:
        return "SOCIAL"
    return "TIE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive self vs social statistical tests.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--output", "-o", default=None, help="Save JSON results to this path.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-config detailed reports.")
    args = parser.parse_args()

    nli_model = _load_nli_model()
    cache: dict = {}
    alpha = args.alpha
    thresholds = args.thresholds

    all_results: list[dict] = []
    skipped: list[str] = []

    for label, category, info_ratio, n_agents, n_subjects, self_dir, social_dir in CONFIG_PAIRS:
        if not self_dir.exists():
            print(f"[SKIP] {label}: self dir not found ({self_dir})")
            skipped.append(label)
            continue
        if not social_dir.exists():
            print(f"[SKIP] {label}: social dir not found ({social_dir})")
            skipped.append(label)
            continue

        print(f"\n{'='*60}\nRunning: {label}\n{'='*60}")
        try:
            res = perform_statistical_tests(
                self_folder=self_dir,
                social_folder=social_dir,
                nli_model=nli_model,
                cache=cache,
                thresholds=thresholds,
                alpha=alpha,
            )
        except Exception as exc:
            print(f"  [ERROR] {label}: {exc}")
            skipped.append(label)
            continue

        res["_label"] = label
        res["_category"] = category
        res["_info_ratio"] = info_ratio
        res["_n_agents"] = n_agents
        res["_n_subjects"] = n_subjects
        all_results.append(res)

        if args.verbose:
            print_stats_report(res)

    # ── Summary table ───────────────────────────────────────────────────────
    W = 220
    SEP = "─" * W
    DSEP = "═" * W

    print(f"\n\n{DSEP}")
    print("  COMPREHENSIVE STATISTICAL SUMMARY: Self-Learning vs Social-Learning")
    print(f"  Metric: NLI coverage score (fraction of ground-truth claims entailed by agent summaries)")
    print(f"  Tests:  Endpoint & AUC → Mann-Whitney U + Welch's t  |  Effect → Cohen's d")
    print(f"  Sig:    *** p<{alpha/50:.4g}  ** p<{alpha/5:.4g}  * p<{alpha}  ns = not significant  (α={alpha})")
    print(DSEP)

    thr_labels = [f"T@{t:.0%}" for t in thresholds]

    hdr1 = (
        f"{'Config':<26} {'Category':<14} {'Info%':<10} {'Agents':<7} {'Subj':<5}"
        f" {'n_S':>4} {'n_So':>4}"
        f"   {'── ENDPOINT ──────────────────────────────────':}"
        f"   {'── AUC ────────────────────────':}"
    )
    hdr2 = (
        f"{'':26} {'':14} {'':10} {'':7} {'':5}"
        f" {'':4} {'':4}"
        f"   {'Self μ':>7} {'Soc μ':>7} {'Δ':>7}"
        f"  {'MWU-p':>7} {'Sig':>4}  {'t-p':>7} {'Sig':>4}  {'d':>6} {'Effect':<11} {'Winner':<8}"
        f"   {'Self':>12} {'Social':>12} {'MWU-p':>7} {'Sig':>4}"
    )
    if thresholds:
        thr_section = "   " + "  ".join(
            f"{'── ' + lbl + ' ───────────────────────────────────────────────':}"
            for lbl in thr_labels
        )
        hdr1 += thr_section
        thr_sub = "   " + "  ".join(
            f"{'S-med':>7} {'So-med':>7} {'S-reach':>8} {'So-reach':>9} {'MWU-p':>7} {'Sig':>4}"
            for _ in thresholds
        )
        hdr2 += thr_sub

    print(hdr1)
    print(hdr2)
    print(SEP)

    for r in all_results:
        ep = r["endpoint"]
        auc = r["auc"]
        ttt = r["time_to_threshold"]

        self_mean = ep["self_mean"]
        soc_mean = ep["social_mean"]
        delta = self_mean - soc_mean

        mwu_ep = ep["mann_whitney"]
        wt_ep = ep["welch_t"]
        d = ep["cohens_d"]
        effect = ep["effect_label"]
        winner = _winner(self_mean, soc_mean)

        mwu_auc = auc["mann_whitney"]

        row = (
            f"{r['_label']:<26} {r['_category']:<14} {r['_info_ratio']:<10}"
            f" {r['_n_agents']:<7} {r['_n_subjects']:<5}"
            f" {r['n_self']:>4} {r['n_social']:>4}"
            f"   {self_mean:>7.4f} {soc_mean:>7.4f} {delta:>+7.4f}"
            f"  {_fmt(mwu_ep['p']):>7} {_sig(mwu_ep['p'], alpha):>4}"
            f"  {_fmt(wt_ep['p']):>7} {_sig(wt_ep['p'], alpha):>4}"
            f"  {d:>6.3f} {effect:<11} {winner:<8}"
            f"   {auc['self_mean']:>12.1f} {auc['social_mean']:>12.1f}"
            f" {_fmt(mwu_auc['p']):>7} {_sig(mwu_auc['p'], alpha):>4}"
        )

        for thr in thresholds:
            td = ttt.get(thr, {})
            s_med = f"{td['self_median']:.0f}s" if td.get("self_median") is not None else "never"
            so_med = f"{td['social_median']:.0f}s" if td.get("social_median") is not None else "never"
            s_reach = f"{td.get('n_self_reached', 0)}/{td.get('n_self_total', r['n_self'])}"
            so_reach = f"{td.get('n_social_reached', 0)}/{td.get('n_social_total', r['n_social'])}"
            mwu_t = td.get("mann_whitney") or {}
            row += (
                f"   {s_med:>7} {so_med:>7} {s_reach:>8} {so_reach:>9}"
                f" {_fmt(mwu_t.get('p')):>7} {_sig(mwu_t.get('p'), alpha):>4}"
            )

        print(row)

    print(SEP)

    # ── Mini-summary ────────────────────────────────────────────────────────
    print(f"\n  QUICK VERDICT (endpoint score, α={alpha})")
    print("  " + "─" * 75)
    for r in all_results:
        ep = r["endpoint"]
        delta = ep["self_mean"] - ep["social_mean"]
        winner = _winner(ep["self_mean"], ep["social_mean"])
        sig_t = _sig(ep["welch_t"]["p"], alpha)
        sig_mw = _sig(ep["mann_whitney"]["p"], alpha)
        print(
            f"  {r['_label']:<28}  winner={winner:<7}  Δμ={delta:+.4f}"
            f"  t={sig_t:<4}  MWU={sig_mw:<4}  effect={ep['effect_label']} (d={ep['cohens_d']:.3f})"
        )

    if skipped:
        print(f"\n  Skipped ({len(skipped)}): {', '.join(skipped)}")

    print(f"\n{DSEP}\n")

    # ── Optional JSON output ─────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
