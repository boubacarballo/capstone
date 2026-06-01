# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run a single experiment (uses active_profile in configs/configs.yaml)
uv run experiment.py

# Run the active profile 10 times sequentially
bash run.sh

# Post-processing / analysis (examples)
uv run plot_swarm_experiment_averages.py
uv run plot_multi_experiment_averages.py
uv run statistical_analysis.py
uv run run_all_stats.py
```

## Architecture

The simulation is a 2D multi-agent world (built on [Violet](https://github.com/m-rots/violet)) where LLM-enabled agents explore an environment, read snippets from stationary/mobile `SubjectAgent`s, and optionally share summaries peer-to-peer. The core question is whether social learning beats self-learning at reconstructing a hidden ground-truth narrative.

**Data flow:**
1. `experiment.py` reads `configs/configs.yaml` via `runtime_config.py` → resolves the active profile into a flat settings dict.
2. `Environment` (subclass of Violet's `Simulation`) spawns `knowledgeAgent`s and `SubjectAgent`s, then drives the main tick loop.
3. Each tick: agents run a proximity FSM (`sensors.py`) → submit async LLM tasks (`llm.py`) → poll for completed summaries → update internal state.
4. At each snapshot interval, `Environment.record_snapshot()` scores every agent's summary against the ground truth (`metrics.py`) and appends results to `experiment.json`.
5. Run output lands in `experiments/<profile>/run_NNNN/`.

**Key design decisions:**
- `runtime_config.py` uses `@lru_cache` on every getter — config is frozen after the first call. Restarting the process is required to pick up config changes.
- `swarm_overrides` in `configs.yaml` can override `swarm_type` without touching the profile definition; this is the intended way to quickly flip self↔social without duplicating config.
- LLM calls are fully async (`llm.py`): `submit_*()` enqueues a task, `poll()` drains completed results. Blocking the event loop inside a tick is avoided by design.
- The metric is set globally in `configs.yaml` under `metric:`, not per-profile; changing it mid-experiment family will make runs incomparable.
- Ground-truth scenarios live in `constants.py` (`GROUND_TRUTH_LIBRARY`) and `story_registry.py`. The profile's `ground_truth_key` selects which one is active.

## Code Style

- **No leading underscores on non-private functions.** Only use `_` prefix for genuinely private helpers or Python dunder methods — not as a general naming convention.
- **No duplication.** Extract repeated logic into a named function or variable rather than copying it.
- **Descriptive names, no abbreviations.** Names should say what they represent: `num_agents` not `n_ag`, `ground_truth_bundle` not `gt_b`.
- **One responsibility per function.** If a function handles multiple concerns, split it.

## Configuration

Active profile is set in `configs/configs.yaml`:
```yaml
experiments:
  active_profile: baseline_social
```

LLM backend is set in `.env` (copy from `.env.example`). When using Ollama, `ollama serve` must be running before launching the experiment.

## HPC (NYUAD Jubail / SLURM)

Scripts live in `hpc/`. Submit with `sbatch hpc/submit_experiment.slurm` (25-task job array, one A100 per task). Each task starts its own `ollama serve` on a unique port to avoid collisions on shared nodes.
