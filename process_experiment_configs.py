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
    # --- axis labels ---
    xlabel: str = "Time (seconds)",
    ylabel: str = "Coverage",
    # --- axis limits ---
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (-0.02, 1.08),
    # --- figure ---
    figsize: tuple[float, float] = (14, 8),
    dpi: int = 150,
    # --- fonts ---
    title_fontsize: int = 28,
    label_fontsize: int = 36,
    tick_fontsize: int = 32,
    legend_fontsize: int = 32,
    # --- legend ---
    legend_loc: str = "lower right",
    show_legend: bool = True,
    legend_labels: list[str] | None = None,
    n_agents: list[int] | None = None,
    # --- grid ---
    grid: bool = True,
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
    xlabel :
        X-axis label text.
    ylabel :
        Y-axis label text.
    xlim :
        ``(x_min, x_max)`` override for the x-axis.  When ``None`` the range
        spans all timestamps.
    ylim :
        ``(y_min, y_max)`` for the y-axis.
    figsize :
        Figure width and height in inches, e.g. ``(16, 6)``.
    dpi :
        Resolution used when saving the figure.
    title_fontsize :
        Font size for the figure title.
    label_fontsize :
        Font size for axis labels.
    tick_fontsize :
        Font size for axis tick labels.
    legend_fontsize :
        Font size for legend text.
    legend_loc :
        Matplotlib location string for the legend (e.g. ``"upper left"``).
    show_legend :
        Set to ``False`` to suppress the legend entirely.
    legend_labels :
        Custom legend entry text, one string per config in *config_results*.
        When ``None`` labels are auto-generated as ``"<name>  (n=N runs)"``.
    n_agents :
        Optional list of agent counts, one per config.  When provided, each
        legend label gains a ``" – A agents"`` suffix regardless of whether
        ``legend_labels`` is set, e.g. ``"Social learning – 25 agents"``.
    grid :
        Set to ``False`` to hide the background grid.

    Returns
    -------
    matplotlib Figure
    """
    if style not in ("line", "area"):
        raise ValueError(f"style must be 'line' or 'area', got '{style}'")

    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = plt.get_cmap("tab10").colors

    for idx, (timestamps, mean_scores, std_scores, label, n_runs) in enumerate(
        config_results
    ):
        color = color_cycle[idx % len(color_cycle)]
        ts = timestamps.astype(float)

        base_label = (
            legend_labels[idx]
            if legend_labels is not None and idx < len(legend_labels)
            else f"{label}  (n={n_runs} runs)"
        )
        agents_suffix = (
            f" – {n_agents[idx]} agents"
            if n_agents is not None and idx < len(n_agents)
            else ""
        )
        legend_label = f"{base_label}{agents_suffix}"

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
            base_label = (
                legend_labels[idx]
                if legend_labels is not None and idx < len(legend_labels)
                else f"{label}  (n={n_runs} runs)"
            )
            agents_suffix = (
                f" – {n_agents[idx]} agents"
                if n_agents is not None and idx < len(n_agents)
                else ""
            )
            legend_label = f"{base_label}{agents_suffix}"
            ax.plot(
                ts,
                mean_scores,
                linewidth=2.0,
                color=color,
                label=legend_label,
                zorder=2 + len(config_results) + idx,
            )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if grid:
        ax.grid(True, alpha=0.25, linestyle="--")
    if show_legend:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.8)
    ax.set_title(title, fontsize=title_fontsize, pad=20)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
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
    # --- axis labels ---
    xlabel: str = "Time (seconds)",
    ylabel: str = "Coverage",
    # --- axis limits ---
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (-0.02, 1.08),
    # --- figure ---
    figsize: tuple[float, float] = (14, 8),
    dpi: int = 150,
    # --- fonts ---
    title_fontsize: int = 28,
    label_fontsize: int = 36,
    tick_fontsize: int = 32,
    legend_fontsize: int = 32,
    # --- legend ---
    legend_loc: str = "lower right",
    show_legend: bool = True,
    legend_labels: list[str] | None = None,
    n_agents: list[int] | None = None,
    # --- grid ---
    grid: bool = True,
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
        Path to save the output plot (PDF).  Defaults to
        ``<config_folder>/average_coverage.pdf`` when a single config is given.
    title :
        Custom plot title.
    nli_model :
        Pre-loaded CrossEncoder model.  Loaded automatically when ``None``.
    style : ``"line"`` | ``"area"``
        Plot style passed to :func:`plot_configs`.
    xlabel :
        X-axis label text.
    ylabel :
        Y-axis label text.
    xlim :
        ``(x_min, x_max)`` override for the x-axis.
    ylim :
        ``(y_min, y_max)`` for the y-axis.
    figsize :
        Figure width and height in inches.
    dpi :
        Resolution used when saving the figure.
    title_fontsize :
        Font size for the figure title.
    label_fontsize :
        Font size for axis labels.
    tick_fontsize :
        Font size for axis tick labels.
    legend_fontsize :
        Font size for legend text.
    legend_loc :
        Matplotlib legend location string.
    show_legend :
        Set to ``False`` to suppress the legend entirely.
    legend_labels :
        Custom legend entry text, one string per config folder.  When
        ``None`` labels are auto-generated as ``"<name>  (n=N runs)"``.
    n_agents :
        Optional list of agent counts, one per config folder.  Always appended
        to each legend label as ``" – A agents"``, even when ``legend_labels``
        is provided explicitly.
    grid :
        Set to ``False`` to hide the background grid.

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
        out = config_folders[0] / "average_coverage.pdf"

    plot_configs(
        config_results,
        output_path=out,
        title=title or "Average Entailment Coverage Score Over Time",
        style=style,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        dpi=dpi,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        legend_loc=legend_loc,
        show_legend=show_legend,
        legend_labels=legend_labels,
        n_agents=n_agents,
        grid=grid,
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
        "--output", "-o",
        metavar="PATH",
        default=None,
        help=(
            "Output path for the plot (PDF). "
            "Defaults to <config_folder>/average_coverage.pdf for a single "
            "config folder."
        ),
    )
    parser.add_argument(
        "--title", "-t",
        metavar="TITLE",
        default=None,
        help="Custom title for the plot.",
    )
    parser.add_argument(
        "--style", "-s",
        choices=["line", "area"],
        default="line",
        help=(
            "Plot style. 'line' (default) draws a bold mean line with a ±1σ "
            "shaded band. 'area' fills from 0 up to the mean for each config, "
            "layering configs with transparency."
        ),
    )
    # ---- axis labels ----
    parser.add_argument(
        "--xlabel",
        metavar="LABEL",
        default="Time (seconds)",
        help="X-axis label. Default: 'Time (seconds)'.",
    )
    parser.add_argument(
        "--ylabel",
        metavar="LABEL",
        default="Coverage",
        help="Y-axis label.",
    )
    # ---- axis limits ----
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("X_MIN", "X_MAX"),
        default=None,
        help="X-axis limits, e.g. --xlim 0 3600.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("Y_MIN", "Y_MAX"),
        default=[-0.02, 1.08],
        help="Y-axis limits. Default: -0.02 1.08.",
    )
    # ---- figure ----
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=[14, 8],
        help="Figure size in inches, e.g. --figsize 16 6. Default: 14 8.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution (DPI) for the saved image. Default: 150.",
    )
    # ---- fonts ----
    parser.add_argument(
        "--title-fontsize",
        type=int,
        default=28,
        metavar="PT",
        help="Font size for the figure title. Default: 28.",
    )
    parser.add_argument(
        "--label-fontsize",
        type=int,
        default=36,
        metavar="PT",
        help="Font size for axis labels. Default: 36.",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=int,
        default=32,
        metavar="PT",
        help="Font size for axis tick labels. Default: 32.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=int,
        default=32,
        metavar="PT",
        help="Font size for legend text. Default: 32.",
    )
    # ---- legend ----
    parser.add_argument(
        "--legend-loc",
        default="lower right",
        metavar="LOC",
        help=(
            "Legend location. Valid values: 'upper right', 'upper left', "
            "'lower left', 'lower right', 'center left', 'center right', "
            "'upper center', 'lower center', 'center'. Default: 'lower right'."
        ),
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        default=False,
        help="Hide the legend.",
    )
    # ---- grid ----
    parser.add_argument(
        "--no-grid",
        action="store_true",
        default=False,
        help="Hide the background grid.",
    )
    # ---- legend labels ----
    parser.add_argument(
        "--legend-labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help=(
            "Custom legend labels, one per config folder (in the same order). "
            "Example: --legend-labels 'Self learning' 'Social learning'"
        ),
    )
    parser.add_argument(
        "--n-agents",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Number of agents, one value per config folder (in the same order). "
            "Appended to auto-generated legend labels when --legend-labels is "
            "not set. Example: --n-agents 10 25"
        ),
    )

    args = parser.parse_args()
    main(
        args.config_folders,
        output_path=args.output,
        title=args.title,
        style=args.style,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim),
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        legend_loc=args.legend_loc,
        show_legend=not args.no_legend,
        legend_labels=args.legend_labels,
        n_agents=args.n_agents,
        grid=not args.no_grid,
    )
