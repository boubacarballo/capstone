"""Parallel parameter sweep over experiment configs.

Builds an ExperimentMatrix of custom fields (e.g. learning_mode), expands it into
unique ExperimentConfig instances, and runs them in parallel with multiprocessing.Pool.

Here the *matrix* values are the source of truth — unlike `experiment.py`, which stays
yaml-driven. Each config carries its own learning_mode/experiment, which Environment and
run_simulation honor over the yaml defaults.

Requirements:
- `is_headless: true` in configs.yaml — Pool runs HeadlessSimulation; multiple on-screen
  windows are not possible.
- If using Ollama, `ollama serve` must be running. Each worker fires async LLM calls, so
  consider limiting `Pool(processes=N)` to avoid overwhelming the backend.

Usage:
    uv run run_matrix.py
"""

from __future__ import annotations

import logging
from multiprocessing import Pool

from config import ExperimentConfig, ExperimentMatrix
from experiment import run_simulation
from logging_setup import setup_logging
from runtime_config import get_runtime_settings
from vi import Window

logger = logging.getLogger(__name__)


def run_one(config: ExperimentConfig) -> str:
    """Worker: run a single config to completion and return its run directory path.

    Must be a top-level function so it is picklable for Pool's 'spawn' start method.
    `get_runtime_settings()` is re-evaluated fresh in each worker process (its lru_cache
    is per-process); the per-config fields on `config` override the yaml-derived values.
    """
    settings = get_runtime_settings()
    simulation = run_simulation(config, settings)
    setup_logging(simulation.run_dir / "run.log")
    try:
        simulation.run()
    finally:
        simulation.stop()
        if not getattr(simulation, "_experiment_saved", False):
            simulation.save_experiment_data()
    return str(simulation.run_dir)


if __name__ == "__main__":
    settings = get_runtime_settings()
    env_width = settings["environment"]["width"]
    env_height = settings["environment"]["height"]

    matrix = ExperimentMatrix(
        window=Window(env_width, env_height),
        fps_limit=settings.get("fps"),
        learning_mode=["self", "social"],
        experiment=settings["active_experiment"],
    )

    configs = matrix.to_configs(ExperimentConfig)
    # Distinct, reproducible seed per config (matrix assigns 1-indexed ids).
    for config in configs:
        config.seed = config.id

    logger.info("Launching %d configs in parallel", len(configs))
    with Pool() as pool:
        run_dirs = pool.map(run_one, configs)

    print("\n".join(run_dirs))
