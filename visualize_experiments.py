import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from metrics import compute_final_score
from runtime_config import get_ground_truth_bundle
from prompts import GROUND_TRUTH_LIBRARY

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

def load_experiment_data(filepath):
    """Load experiment data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_run(run_path):
    """Load experiment summaries and metadata from a run directory."""
    run_dir = Path(run_path)
    data_path = run_dir / "experiment.json"
    metadata_path = run_dir / "metadata.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Experiment data not found at {data_path}")

    data = load_experiment_data(data_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as meta_file:
            metadata = json.load(meta_file)
    return data, metadata


def list_run_directories(
    base_dir="experiments",
    *,
    profile=None,
    cohort=None,
    knowledge_agents=None,
    ground_truth_key=None,
):
    """Return run directories matching the provided filters."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    profiles = [profile] if profile else [p.name for p in base_path.iterdir() if p.is_dir()]
    matching_runs = []

    for profile_name in profiles:
        profile_dir = base_path / profile_name
        if not profile_dir.is_dir():
            continue

        cohorts = [cohort] if cohort else [c.name for c in profile_dir.iterdir() if c.is_dir()]
        for cohort_name in cohorts:
            cohort_dir = profile_dir / cohort_name
            if not cohort_dir.is_dir():
                continue

            agent_dirs = sorted([d for d in cohort_dir.iterdir() if d.is_dir() and d.name.startswith("ka-")])
            for agent_dir in agent_dirs:
                if knowledge_agents is not None and agent_dir.name != f"ka-{knowledge_agents}":
                    continue

                scenario_dirs = [d for d in agent_dir.iterdir() if d.is_dir()]
                for scenario_dir in scenario_dirs:
                    if ground_truth_key and scenario_dir.name != ground_truth_key:
                        continue

                    run_dirs = [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
                    matching_runs.extend(sorted(run_dirs))

    return sorted(matching_runs)


def compute_group_scores(group_input, ground_truth):
    """Compute averaged scores for a group of runs or pre-loaded experiment data."""
    if isinstance(group_input, dict):
        return compute_scores_over_time(group_input, ground_truth)

    if isinstance(group_input, (str, Path)):
        group_paths = [Path(group_input)]
    else:
        group_paths = [Path(p) for p in group_input]

    step_scores: dict[int, list[float]] = defaultdict(list)

    for path in group_paths:
        data_path = path
        if path.is_dir():
            data_path = path / "experiment.json"
        if not data_path.exists():
            print(f"⚠️ Skipping missing experiment data at {data_path}")
            continue
        try:
            run_data = load_experiment_data(data_path)
        except Exception as exc:
            print(f"⚠️ Failed to load {data_path}: {exc}")
            continue

        steps, scores = compute_scores_over_time(run_data, ground_truth)
        for step, score in zip(steps, scores):
            step_scores[step].append(score)

    aggregated_steps = sorted(step_scores.keys())
    aggregated_scores = [float(np.mean(step_scores[step])) for step in aggregated_steps]
    return aggregated_steps, aggregated_scores

def compute_scores_over_time(experiment_data, ground_truth):
    """
    Compute average scores at each observation step.
    
    Parameters
    ----------
    experiment_data : dict
        Dictionary with keys as step numbers (strings) and values as lists of summaries
    ground_truth : str
        Ground truth text to compare against
    
    Returns
    -------
    steps : list
        List of observation step numbers
    avg_scores : list
        List of average scores at each step
    """
    steps = []
    avg_scores = []
    
    # Sort steps numerically
    step_numbers = sorted([int(k) for k in experiment_data.keys()])
    
    print(f"  Processing {len(step_numbers)} observation steps...")
    for i, step in enumerate(step_numbers):
        if (i + 1) % 2 == 0 or i == 0 or i == len(step_numbers) - 1:
            print(f"  Step {step} ({i+1}/{len(step_numbers)})...", end='\r')
        step_key = str(step)
        summaries = experiment_data[step_key]
        
        if not summaries:
            continue
        
        # Compute scores for all summaries at this step
        scores = []
        for summary in summaries:
            if summary and summary.strip():  # Skip empty summaries
                try:
                    # Suppress print statements from compute_final_score
                    with suppress_stdout():
                        score = compute_final_score(summary, ground_truth)
                    scores.append(score)
                except Exception as e:
                    print(f"Error computing score for step {step}: {e}")
                    continue
        
        if scores:
            avg_score = np.mean(scores)
            steps.append(step)
            avg_scores.append(avg_score)
    
    print()  # New line after progress updates
    return steps, avg_scores

def plot_comparison(
    group_a,
    group_b,
    ground_truth=None,
    output_path="experiments/comparison_plot.png",
    label_a="Group A",
    label_b="Group B",
):
    """
    Create a comparison plot of experiments 25 and 27.
    
    Parameters
    ----------
    exp25_data : dict
        Experiment 25 data (social learning enabled)
    exp27_data : dict
        Experiment 27 data (social learning disabled)
    ground_truth : str
        Ground truth text
    output_path : str
        Path to save the plot
    """
    # Compute scores for both experiments
    active_ground_truth = ground_truth or get_ground_truth_bundle().get("text", "")
    steps_25, scores_25 = compute_group_scores(group_a, active_ground_truth)
    steps_27, scores_27 = compute_group_scores(group_b, active_ground_truth)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot experiment 25 (with communication) - orange, dashed, 'x' markers
    plt.plot(steps_25, scores_25, 
             color='orange', 
             linestyle='--', 
             marker='x', 
             markersize=8,
             linewidth=2,
             label=label_a)
    
    # Plot experiment 27 (no communication) - blue, solid, circular markers
    plt.plot(steps_27, scores_27, 
             color='blue', 
             linestyle='-', 
             marker='o', 
             markersize=8,
             linewidth=2,
             label=label_b)
    
    # Customize the plot
    plt.xlabel('Observation Step', fontsize=12)
    plt.ylabel('Average Best-Match Similarity Score', fontsize=12)
    plt.title('Comparison of Average Semantic Learning (Average Raw Fact Score)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(left=-0.5)
    plt.ylim(0.0, 1.0)
    
    # Set y-axis ticks
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    
    # Set x-axis ticks based on the data range
    max_step = max(max(steps_25) if steps_25 else 0, max(steps_27) if steps_27 else 0)
    if max_step <= 10:
        plt.xticks(range(0, max_step + 1, 2))
    else:
        plt.xticks(range(0, max_step + 1, max(1, max_step // 5)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare experiment groups and plot learning progress.")
    parser.add_argument("--group-a", nargs="+", help="Run directories or experiment JSON files for group A.")
    parser.add_argument("--group-b", nargs="+", help="Run directories or experiment JSON files for group B.")
    parser.add_argument("--label-a", default="Group A", help="Legend label for group A.")
    parser.add_argument("--label-b", default="Group B", help="Legend label for group B.")
    parser.add_argument("--ground-truth", default=None, help="Override ground-truth text or key. Defaults to active profile ground truth.")
    parser.add_argument("--output", default="experiments/comparison_plot.png", help="Path for the output plot image.")
    parser.add_argument("--use-key", action="store_true", help="Interpret --ground-truth as a ground-truth key instead of raw text.")

    args = parser.parse_args()

    if not args.group_a or not args.group_b:
        parser.error("Both --group-a and --group-b must be provided.")

    active_gt = args.ground_truth
    if args.use_key and args.ground_truth:
        bundle = GROUND_TRUTH_LIBRARY.get(args.ground_truth)
        if bundle:
            active_gt = bundle.get("text", "")
        else:
            print(f"⚠️ Unknown ground-truth key '{args.ground_truth}'. Falling back to active configuration.")
            active_gt = get_ground_truth_bundle().get("text", "")
    elif not args.ground_truth:
        active_gt = get_ground_truth_bundle().get("text", "")

    print("Computing scores and creating plot...")
    plot_comparison(
        args.group_a,
        args.group_b,
        active_gt,
        output_path=args.output,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    print("Done!")

