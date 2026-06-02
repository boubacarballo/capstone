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
3. Each tick: agents run a proximity FSM (`sensors.py`) → submit async LLM tasks (`llm.py`) → check completed futures → update internal state.
4. At each snapshot interval, `Environment.record_snapshot()` scores every agent's summary against the ground truth (`metrics.py`) and appends results to `experiment.json`.
5. Run output lands in `experiments/<profile>/run_NNNN/`.

**Key design decisions:**
- `runtime_config.py` uses `@lru_cache` on every getter — config is frozen after the first call. Restarting the process is required to pick up config changes.
- `swarm_overrides` in `configs.yaml` can override `swarm_type` without touching the profile definition; this is the intended way to quickly flip self↔social without duplicating config.
- LLM calls are fully async (`llm.py`): `submit_*()` returns a `Future` stored directly on the agent; each tick checks `future.done()` and reads the result. Blocking the event loop inside a tick is avoided by design.
- The metric is set globally in `configs.yaml` under `metric:`, not per-profile; changing it mid-experiment family will make runs incomparable.
- Ground-truth scenarios live in `prompts.py` (`GROUND_TRUTH_LIBRARY`). The profile's `ground_truth_key` selects which one is active.

## Configuration

Active profile is set in `configs/configs.yaml`:
```yaml
experiments:
  active_profile: baseline_social
```

LLM backend is set in `.env` (copy from `.env.example`). Supported providers: `ollama` (default, local) and `openai` (cloud). When using Ollama, `ollama serve` must be running before launching the experiment.

