from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from process_experiment import _load_nli_model, process_run


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def discover_runs(config_folder: Path) -> list[Path]:
    """
    Return sorted list of run subdirectories inside *config_folder*.

    A valid run directory must be named ``run_*`` and contain both
    ``experiment.json`` and ``metadata.json``.
    """
    runs = sorted(
        p
        for p in config_folder.iterdir()
        if p.is_dir()
        and p.name.startswith("run_")
        and (p / "experiment.json").exists()
        and (p / "metadata.json").exists()
    )
    if not runs:
        raise FileNotFoundError(
            f"No valid run directories found in '{config_folder}'. "
            "Expected subdirectories named 'run_*' containing "
            "'experiment.json' and 'metadata.json'."
        )
    return runs


# ---------------------------------------------------------------------------
# Config-level aggregation
# ---------------------------------------------------------------------------


def aggregate_config_runs(
    config_folder: Path,
    nli_model,
    cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """
    Process all runs in *config_folder* and return aggregate statistics.

    Each run's per-agent scores are computed via :func:`process_run`.  The
    resulting matrices are aligned to the common timestamp intersection, then
    stacked into a ``(n_runs, n_agents, n_common_ts)`` array from which the
    mean and std over *all* runs and agents are derived.

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
    mean_scores : np.ndarray, shape (n_common_ts,)
    std_scores : np.ndarray, shape (n_common_ts,)
    label : str  — ``config_folder.name``
    n_runs : int — number of runs processed
    """
    if cache is None:
        cache = {}

    run_paths = discover_runs(config_folder)
    n_runs = len(run_paths)
    print(f"\n[{config_folder.name}] Found {n_runs} run(s).")

    # Collect per-run results
    run_results: list[tuple[list[int], np.ndarray]] = []
    for run_path in tqdm(run_paths, desc=config_folder.name, unit="run"):
        timestamps, all_scores, _agent_ids, _label = process_run(
            run_path, nli_model, cache
        )
        run_results.append((timestamps, all_scores))

    # Build common timestamp intersection
    ts_sets = [set(ts_list) for ts_list, _ in run_results]
    common_ts_set: set[int] = ts_sets[0].intersection(*ts_sets[1:])
    common_timestamps = np.array(sorted(common_ts_set), dtype=int)

    if common_timestamps.size == 0:
        raise ValueError(
            f"No common timestamps found across runs in '{config_folder}'. "
            "Cannot aggregate."
        )

    # Stack scores aligned to common timestamps
    # Shape: (n_runs, n_agents, n_common_ts)
    stacked: list[np.ndarray] = []
    for ts_list, all_scores in run_results:
        ts_to_idx = {t: i for i, t in enumerate(ts_list)}
        col_indices = [ts_to_idx[t] for t in common_timestamps]
        stacked.append(all_scores[:, col_indices])

    # Pad agent axis to the maximum agent count if runs differ
    max_agents = max(m.shape[0] for m in stacked)
    padded: list[np.ndarray] = []
    for m in stacked:
        if m.shape[0] < max_agents:
            pad = np.zeros((max_agents - m.shape[0], m.shape[1]))
            m = np.vstack([m, pad])
        padded.append(m)

    volume = np.stack(padded, axis=0)  # (n_runs, n_agents, n_common_ts)

    # Flatten runs × agents, then compute stats across that joint axis
    flat = volume.reshape(-1, common_timestamps.size)  # (n_runs*n_agents, n_ts)
    mean_scores = flat.mean(axis=0)
    std_scores = flat.std(axis=0)

    return common_timestamps, mean_scores, std_scores, config_folder.name, n_runs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_configs(
    config_results: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, int]],
    output_path: Path | str | None = None,
    title: str = "Average Entailment Coverage Score Over Time",
    style: str = "line",
) -> plt.Figure:
    """
    Plot averaged coverage curves for one or more experiment configurations.

    Parameters
    ----------
    config_results :
        List of ``(common_timestamps, mean_scores, std_scores, label, n_runs)``
        tuples as returned by :func:`aggregate_config_runs`.
    output_path :
        If provided, the figure is saved here; otherwise ``plt.show()`` is
        called.
    title :
        Overall figure title.
    style : ``"line"`` | ``"area"``
        ``"line"`` (default) — bold mean line with a light ±1σ shaded band.
        ``"area"`` — each config is rendered as a filled area from 0 up to the
        mean, with the mean line drawn on top.  Multiple configs are layered
        with transparency so all fills remain visible.

    Returns
    -------
    matplotlib Figure
    """
    if style not in ("line", "area"):
        raise ValueError(f"style must be 'line' or 'area', got '{style}'")

    fig, ax = plt.subplots(figsize=(12, 5))
    color_cycle = plt.get_cmap("tab10").colors

    for idx, (timestamps, mean_scores, std_scores, label, n_runs) in enumerate(
        config_results
    ):
        color = color_cycle[idx % len(color_cycle)]
        ts = timestamps.astype(float)

        legend_label = f"{label}  (n={n_runs} runs)"

        if style == "area":
            pass  # handled in two-pass block below
        else:
            ax.plot(ts, mean_scores, linewidth=2.5, color=color, label=legend_label, zorder=3)
            ax.fill_between(
                ts,
                mean_scores - std_scores,
                mean_scores + std_scores,
                color=color,
                alpha=0.18,
                zorder=2,
            )

    if style == "area":
        # Two-pass rendering so every line stays visible regardless of area sizes.
        # Pass 1 — fills only, drawn largest-area-first so bigger fills go to
        #           the back and smaller fills layer on top.
        sorted_by_area = sorted(
            enumerate(config_results),
            key=lambda x: float(x[1][1].mean()),  # mean of mean_scores
            reverse=True,  # largest first → lowest zorder
        )
        for z_offset, (idx, (timestamps, mean_scores, _std, _label, _n)) in enumerate(
            sorted_by_area
        ):
            color = color_cycle[idx % len(color_cycle)]
            ts = timestamps.astype(float)
            ax.fill_between(
                ts,
                0,
                mean_scores,
                color=color,
                alpha=0.50,
                zorder=2 + z_offset,
            )

        # Pass 2 — lines only, all drawn above every fill
        for idx, (timestamps, mean_scores, _std, label, n_runs) in enumerate(
            config_results
        ):
            color = color_cycle[idx % len(color_cycle)]
            ts = timestamps.astype(float)
            legend_label = f"{label}  (n={n_runs} runs)"
            ax.plot(
                ts,
                mean_scores,
                linewidth=2.0,
                color=color,
                label=legend_label,
                zorder=2 + len(config_results) + idx,
            )

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Coverage score  (entailed claims / total claims)", fontsize=11)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
    ax.set_title(title, fontsize=13, pad=8)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {output_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def main(
    config_folders: list[str | Path],
    output_path: str | Path | None = None,
    title: str | None = None,
    nli_model=None,
    style: str = "line",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str, int]]:
    """
    Process one or more experiment configuration folders and plot averages.

    Parameters
    ----------
    config_folders :
        List of paths to experiment configuration directories.  Each must
        contain ``run_*`` subdirectories with ``experiment.json`` and
        ``metadata.json``.
    output_path :
        Path to save the output plot (PNG).  Defaults to
        ``<config_folder>/average_coverage.png`` when a single config is given.
    title :
        Custom plot title.
    nli_model :
        Pre-loaded CrossEncoder model.  Loaded automatically when ``None``.
    style : ``"line"`` | ``"area"``
        Plot style passed to :func:`plot_configs`.

    Returns
    -------
    list of ``(common_timestamps, mean_scores, std_scores, label, n_runs)``
    """
    config_folders = [Path(f) for f in config_folders]

    if nli_model is None:
        nli_model = _load_nli_model()

    cache: dict = {}
    config_results = []

    for folder in config_folders:
        result = aggregate_config_runs(folder, nli_model, cache)
        config_results.append(result)

    out = output_path
    if out is None and len(config_folders) == 1:
        out = config_folders[0] / "average_coverage.png"

    plot_configs(
        config_results,
        output_path=out,
        title=title or "Average Entailment Coverage Score Over Time",
        style=style,
    )

    return config_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate NLI-based entailment coverage scores across all runs "
            "in one or more experiment configuration folders and plot the "
            "average score over time."
        )
    )
    parser.add_argument(
        "config_folders",
        nargs="+",
        metavar="CONFIG_FOLDER",
        help=(
            "One or more experiment configuration folder paths, each "
            "containing run_* subdirectories "
            "(e.g. experiments/dynamic_pool_self_10/)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        default=None,
        help=(
            "Output path for the plot image (PNG). "
            "Defaults to <config_folder>/average_coverage.png for a single "
            "config folder."
        ),
    )
    parser.add_argument(
        "--title",
        "-t",
        metavar="TITLE",
        default=None,
        help="Custom title for the plot.",
    )
    parser.add_argument(
        "--style",
        "-s",
        choices=["line", "area"],
        default="line",
        help=(
            "Plot style. 'line' (default) draws a bold mean line with a ±1σ "
            "shaded band. 'area' fills from 0 up to the mean for each config, "
            "layering configs with transparency."
        ),
    )
    args = parser.parse_args()
    main(
        args.config_folders,
        output_path=args.output,
        title=args.title,
        style=args.style,
    )
