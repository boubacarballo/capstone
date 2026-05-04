from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

from process_experiment import _load_nli_model, process_run
from process_experiment_configs import discover_runs


# ---------------------------------------------------------------------------
# Per-run mean extraction
# ---------------------------------------------------------------------------


def extract_per_run_means(
    config_folder: Path,
    nli_model,
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Process all runs in *config_folder* and return per-run mean trajectories.

    The experimental unit for statistical testing is the run, not the
    individual agent.  This function averages over agents within each run,
    producing one mean trajectory per run instead of flattening runs and
    agents together.

    Parameters
    ----------
    config_folder :
        Path to an experiment configuration directory containing ``run_*``
        subdirectories.
    nli_model :
        Loaded CrossEncoder NLI model.
    cache :
        Shared mutable cache to avoid recomputing identical summary scores.

    Returns
    -------
    common_timestamps : np.ndarray, shape (n_common_ts,)
    per_run_means : np.ndarray, shape (n_runs, n_common_ts)
        Each row is the mean coverage score across agents for one run.
    label : str
        ``config_folder.name``
    """
    if cache is None:
        cache = {}

    run_paths = discover_runs(config_folder)
    n_runs = len(run_paths)
    print(f"\n[{config_folder.name}] Found {n_runs} run(s).")

    run_results: list[tuple[list[int], np.ndarray]] = []
    for run_path in tqdm(run_paths, desc=config_folder.name, unit="run"):
        timestamps, all_scores, _agent_ids, _label = process_run(
            run_path, nli_model, cache
        )
        run_results.append((timestamps, all_scores))

    # Align all runs to the intersection of their timestamps
    ts_sets = [set(ts_list) for ts_list, _ in run_results]
    common_ts_set: set[int] = ts_sets[0].intersection(*ts_sets[1:])
    common_timestamps = np.array(sorted(common_ts_set), dtype=int)

    if common_timestamps.size == 0:
        raise ValueError(
            f"No common timestamps found across runs in '{config_folder}'."
        )

    # Build (n_runs, n_agents, n_common_ts) volume
    stacked: list[np.ndarray] = []
    for ts_list, all_scores in run_results:
        ts_to_idx = {t: i for i, t in enumerate(ts_list)}
        col_indices = [ts_to_idx[t] for t in common_timestamps]
        stacked.append(all_scores[:, col_indices])

    max_agents = max(m.shape[0] for m in stacked)
    padded: list[np.ndarray] = []
    for m in stacked:
        if m.shape[0] < max_agents:
            pad = np.zeros((max_agents - m.shape[0], m.shape[1]))
            m = np.vstack([m, pad])
        padded.append(m)

    volume = np.stack(padded, axis=0)  # (n_runs, n_agents, n_common_ts)

    # Average over agents → one trajectory per run
    per_run_means = volume.mean(axis=1)  # (n_runs, n_common_ts)

    return common_timestamps, per_run_means, config_folder.name


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled std) between two 1-D arrays."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _effect_label(d: float) -> str:
    """Map an absolute Cohen's d value to a descriptive label."""
    d = abs(d)
    if d < 0.20:
        return "negligible"
    if d < 0.50:
        return "small"
    if d < 0.80:
        return "medium"
    if d < 1.20:
        return "large"
    return "huge"


def _sig_stars(p: float, alpha: float) -> str:
    if p < alpha / 50:
        return "***"
    if p < alpha / 5:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


def _mwu(a: np.ndarray, b: np.ndarray) -> dict:
    """Run a two-sided Mann-Whitney U test and return a result dict."""
    if len(a) < 2 or len(b) < 2:
        return {"U": None, "p": None, "significant": False}
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(result.statistic), "p": float(result.pvalue), "significant": None}


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    """Run a two-sided Welch t-test and return a result dict."""
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "p": None, "significant": False}
    result = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(result.statistic), "p": float(result.pvalue), "significant": None}


def _threshold_times(
    per_run_means: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    For each run return the first timestamp at which the mean score reaches
    *threshold*.  Runs that never reach the threshold are represented as
    ``np.nan``.
    """
    times = []
    for run in per_run_means:
        idx = np.argmax(run >= threshold)
        if run[idx] >= threshold:
            times.append(float(timestamps[idx]))
        else:
            times.append(float("nan"))
    return np.array(times)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def perform_statistical_tests(
    self_folder: Path | str,
    social_folder: Path | str,
    nli_model=None,
    cache: dict | None = None,
    thresholds: list[float] | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compare self-learning vs social-learning on a given experiment config pair.

    The experimental unit is the per-run mean coverage score (averaged over
    agents within a run).  Three families of tests are run:

    1. **Endpoint** — final mean score per run (Mann-Whitney U, Welch's t,
       Cohen's d).
    2. **AUC** — area under the mean-score trajectory per run, computed with
       the trapezoidal rule (Mann-Whitney U).
    3. **Time-to-threshold** — for each threshold, the first timestamp at
       which a run's mean score reaches that value (Mann-Whitney U).

    Parameters
    ----------
    self_folder :
        Path to the self-learning config folder (contains ``run_*`` subdirs).
    social_folder :
        Path to the social-learning config folder (contains ``run_*`` subdirs).
    nli_model :
        Pre-loaded CrossEncoder model.  Loaded automatically when ``None``.
    cache :
        Shared mutable cache passed through to ``process_run``.
    thresholds :
        Coverage thresholds for the time-to-threshold test.
        Defaults to ``[0.5, 0.8]``.
    alpha :
        Significance level used to set the ``significant`` flag.
        Defaults to ``0.05``.

    Returns
    -------
    dict with keys ``config_name``, ``n_self``, ``n_social``,
    ``endpoint``, ``auc``, ``time_to_threshold``.
    """
    if thresholds is None:
        thresholds = [0.5, 0.8]

    self_folder = Path(self_folder)
    social_folder = Path(social_folder)

    if nli_model is None:
        nli_model = _load_nli_model()
    if cache is None:
        cache = {}

    ts_self, prm_self, label_self = extract_per_run_means(
        self_folder, nli_model, cache
    )
    ts_social, prm_social, label_social = extract_per_run_means(
        social_folder, nli_model, cache
    )

    # Align both sets of per-run means to a shared timestamp grid
    common_ts = np.array(
        sorted(set(ts_self.tolist()) & set(ts_social.tolist())), dtype=int
    )
    if common_ts.size == 0:
        raise ValueError(
            "No common timestamps between self and social folders. "
            "Cannot run statistical tests."
        )

    self_idx = [np.where(ts_self == t)[0][0] for t in common_ts]
    social_idx = [np.where(ts_social == t)[0][0] for t in common_ts]

    prm_self = prm_self[:, self_idx]
    prm_social = prm_social[:, social_idx]

    n_self = prm_self.shape[0]
    n_social = prm_social.shape[0]

    # ------------------------------------------------------------------
    # 1. Endpoint: final mean score per run
    # ------------------------------------------------------------------
    final_self = prm_self[:, -1]
    final_social = prm_social[:, -1]

    mwu_ep = _mwu(final_self, final_social)
    welch_ep = _welch(final_self, final_social)
    d = _cohens_d(final_self, final_social)

    if mwu_ep["p"] is not None:
        mwu_ep["significant"] = mwu_ep["p"] < alpha
    if welch_ep["p"] is not None:
        welch_ep["significant"] = welch_ep["p"] < alpha

    endpoint = {
        "self_mean": float(final_self.mean()),
        "social_mean": float(final_social.mean()),
        "mann_whitney": mwu_ep,
        "welch_t": welch_ep,
        "cohens_d": d,
        "effect_label": _effect_label(d),
    }

    # ------------------------------------------------------------------
    # 2. AUC: trapezoidal area under each run's mean trajectory
    # ------------------------------------------------------------------
    ts_float = common_ts.astype(float)
    auc_self = np.array([np.trapezoid(row, ts_float) for row in prm_self])
    auc_social = np.array([np.trapezoid(row, ts_float) for row in prm_social])

    mwu_auc = _mwu(auc_self, auc_social)
    if mwu_auc["p"] is not None:
        mwu_auc["significant"] = mwu_auc["p"] < alpha

    auc = {
        "self_mean": float(auc_self.mean()),
        "social_mean": float(auc_social.mean()),
        "mann_whitney": mwu_auc,
    }

    # ------------------------------------------------------------------
    # 3. Time-to-threshold
    # ------------------------------------------------------------------
    time_to_threshold: dict[float, dict] = {}
    for thr in thresholds:
        times_self = _threshold_times(prm_self, common_ts, thr)
        times_social = _threshold_times(prm_social, common_ts, thr)

        reached_self = times_self[~np.isnan(times_self)]
        reached_social = times_social[~np.isnan(times_social)]

        if len(reached_self) >= 2 and len(reached_social) >= 2:
            mwu_thr = _mwu(reached_self, reached_social)
            mwu_thr["significant"] = mwu_thr["p"] < alpha
        else:
            mwu_thr = None

        time_to_threshold[thr] = {
            "self_median": float(np.nanmedian(times_self)) if len(reached_self) > 0 else None,
            "social_median": float(np.nanmedian(times_social)) if len(reached_social) > 0 else None,
            "n_self_reached": int(len(reached_self)),
            "n_social_reached": int(len(reached_social)),
            "n_self_total": n_self,
            "n_social_total": n_social,
            "mann_whitney": mwu_thr,
        }

    return {
        "config_name": f"{label_self} vs {label_social}",
        "self_label": label_self,
        "social_label": label_social,
        "n_self": n_self,
        "n_social": n_social,
        "alpha": alpha,
        "endpoint": endpoint,
        "auc": auc,
        "time_to_threshold": time_to_threshold,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_stats_report(results: dict) -> None:
    """Pretty-print a statistical comparison report to stdout."""
    alpha = results.get("alpha", 0.05)

    def stars(p: float | None) -> str:
        if p is None:
            return "n/a"
        return _sig_stars(p, alpha)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Statistical Comparison: {results['config_name']}")
    print(f"  Experimental unit: per-run mean  "
          f"(n_self={results['n_self']}, n_social={results['n_social']})")
    print(sep)

    # Endpoint
    ep = results["endpoint"]
    mwu = ep["mann_whitney"]
    wt = ep["welch_t"]
    print("\n  Endpoint score  (final timestamp)")
    print(f"    Self mean:    {ep['self_mean']:.3f}  |  "
          f"Social mean:  {ep['social_mean']:.3f}")
    if mwu["p"] is not None:
        print(f"    Mann-Whitney: U={mwu['U']:.1f}, p={mwu['p']:.4f}  {stars(mwu['p'])}")
    if wt["p"] is not None:
        print(f"    Welch t-test: t={wt['t']:.2f}, p={wt['p']:.4f}  {stars(wt['p'])}")
    print(f"    Cohen's d:    {ep['cohens_d']:.2f}  ({ep['effect_label']})")

    # AUC
    auc = results["auc"]
    mwu_a = auc["mann_whitney"]
    print("\n  AUC  (area under mean-score trajectory)")
    print(f"    Self mean:    {auc['self_mean']:.1f}  |  "
          f"Social mean:  {auc['social_mean']:.1f}")
    if mwu_a["p"] is not None:
        print(f"    Mann-Whitney: U={mwu_a['U']:.1f}, p={mwu_a['p']:.4f}  {stars(mwu_a['p'])}")

    # Time-to-threshold
    for thr, tdata in results["time_to_threshold"].items():
        print(f"\n  Time to {thr:.0%} coverage")
        s_med = f"{tdata['self_median']:.0f}s" if tdata["self_median"] is not None else "never"
        so_med = f"{tdata['social_median']:.0f}s" if tdata["social_median"] is not None else "never"
        print(f"    Self:   {s_med}  "
              f"({tdata['n_self_reached']}/{tdata['n_self_total']} runs reached)")
        print(f"    Social: {so_med}  "
              f"({tdata['n_social_reached']}/{tdata['n_social_total']} runs reached)")
        mwu_t = tdata["mann_whitney"]
        if mwu_t is not None and mwu_t["p"] is not None:
            print(f"    Mann-Whitney: U={mwu_t['U']:.1f}, p={mwu_t['p']:.4f}  {stars(mwu_t['p'])}")
        else:
            print("    Mann-Whitney: insufficient data")

    print(f"\n  Significance (α={alpha}): * p<{alpha}  "
          f"** p<{alpha/5:.3g}  *** p<{alpha/50:.3g}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare self-learning vs social-learning experiment configs "
            "using Mann-Whitney U, Welch's t-test, Cohen's d, AUC, and "
            "time-to-threshold statistical tests."
        )
    )
    parser.add_argument(
        "self_folder",
        metavar="SELF_FOLDER",
        help="Path to the self-learning config folder (e.g. experiments/baseline_self).",
    )
    parser.add_argument(
        "social_folder",
        metavar="SOCIAL_FOLDER",
        help="Path to the social-learning config folder (e.g. experiments/baseline_social).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.8],
        metavar="T",
        help="Coverage thresholds for time-to-threshold tests. Default: 0.5 0.8.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Significance level. Default: 0.05.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help=(
            "Save the results to a JSON file at this path. "
            "If omitted, only the text report is printed."
        ),
    )

    args = parser.parse_args()

    results = perform_statistical_tests(
        self_folder=args.self_folder,
        social_folder=args.social_folder,
        thresholds=args.thresholds,
        alpha=args.alpha,
    )

    print_stats_report(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved → {out_path}")
