# LLM-Augmented Swarm Intelligence: Knowledge Acquisition in Multi-Agent Systems

A research simulation that pits **self-learning** swarms against **social-learning** swarms and measures how quickly and how accurately a group of LLM-enabled agents can collectively reconstruct a hidden narrative from distributed information sources.

Built on top of [Violet](https://github.com/m-rots/violet), a lightweight 2D agent simulation framework.

---

## What This Project Does

Agents roam a 2D environment populated by **subject agents** — stationary or mobile NPCs that each carry one snippet of a larger ground-truth narrative (e.g., a career fair scene). When a knowledge agent steps near a subject it reads that snippet, passes it to an LLM, and updates its internal summary. Periodically, agents that are near each other can exchange summaries and fuse them with the LLM.

At fixed time intervals the simulation records each agent's current summary and scores it against the ground truth using a combined semantic + lexical metric. The experiment ends after a configured number of snapshots, and all results are persisted to disk.

The central research question is:

> Does social learning (agents sharing summaries peer-to-peer) lead to faster or more accurate collective knowledge acquisition than self-learning (agents only accumulating their own observations)?

---

## Architecture Overview

```
experiment.py          ← entry point; wires everything together
├── agents.py          ← knowledgeAgent: FSM-driven; senses, shares, summarises
├── subjects.py        ← SubjectAgent: carries one ground-truth snippet
├── environment.py     ← Environment (extends Simulation): snapshot loop, scoring,
│                         information-teleportation modes, output persistence
├── sensors.py         ← Sensor (proximity reads) + Actuator (movement)
├── llm.py             ← async LLM wrapper: Ollama / OpenAI / vLLM
├── metrics.py         ← cosine, BM25, BERTScore, NLI scorers + heatmap renderer
├── constants.py       ← system prompts + ground-truth library (3 difficulty tiers)
├── story_registry.py  ← alternative "Lost Artifact" narrative scenario
├── runtime_config.py  ← reads configs.yaml and resolves the active profile
└── configs/
    └── configs.yaml   ← all experiment profiles and global settings
```

Output from each run lands in `experiments/<profile>/run_NNNN/`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.13+ | enforced by `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | fast package manager / venv runner |
| An LLM backend | Ollama (local), vLLM (local server), or OpenAI (cloud) — see below |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd capstone

# 2. Create the virtual environment and install dependencies
uv sync

# 3. Copy the environment template and fill in your LLM credentials
cp .env.example .env
```

---

## Configuration

### LLM Backend (`.env`)

Choose one backend and edit `.env` accordingly:

```dotenv
# ── Ollama (local, default) ──────────────────────────────
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e4b

# ── OpenAI (cloud) ───────────────────────────────────────
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o

# ── vLLM (local server, OpenAI-compatible) ───────────────
# LLM_PROVIDER=vllm
# VLLM_BASE_URL=http://localhost:8000/v1
# VLLM_API_KEY=EMPTY
# LLM_MODEL=google/gemma-3-4b-it

# Parallel LLM threads (tune to your hardware)
LLM_MAX_WORKERS=200
```

When using **Ollama**, start the server before running:

```bash
ollama serve          # in one terminal
ollama pull gemma4:e4b
```

### Experiment Profile (`configs/configs.yaml`)

Set `experiments.active_profile` to the profile you want to run:

```yaml
experiments:
  active_profile: baseline_social   # ← change this
  num_snapshots: 400
  snapshot_interval_seconds: 3.0
```

Key profile settings:

| Key | Description |
|---|---|
| `swarm_type` | `self_learning` or `social_learning` |
| `num_knowledge_agents` | Number of roaming agents |
| `ground_truth_key` | `ground_truth_1` (10 snippets), `ground_truth_2` (20), `ground_truth_3` (40) |
| `environment.width/height` | World dimensions in pixels |
| `information_teleportation` | Controls how subjects appear/disappear (see below) |
| `movement` | Whether subjects move around the environment |

Available built-in profiles (all have `_self` and `_social` variants):

| Profile | Description |
|---|---|
| `baseline_self / baseline_social` | All subjects always visible |
| `decay_self_learning / decay_social_learning` | Subjects expire exponentially (no reappearance) |
| `movement_self_learning / movement_social_learning` | Subjects wander the environment |
| `dynamic_pool_self / dynamic_pool_social` | Subjects pop in/out; snippets recycled to a pool |
| `constant_ratio_pool_self / constant_ratio_pool_social` | Fixed percentage of subjects visible at all times |
| `exponential_swap_pool_self_* / _social_*` | Single Poisson timer swaps one active ↔ one inactive subject (20 / 50 / 80 % active variants) |
| `exponential_one_time_pool_self_* / _social_*` | Like swap pool but each subject can only appear once |

### Metric (`configs/configs.yaml`)

```yaml
metric: cosine-bm25   # default; options: cosine-bm25 | cosine-bert | bert-score | nli
```

| Metric | What it measures |
|---|---|
| `cosine-bm25` | 0.7 × semantic cosine similarity + 0.3 × BM25 (lexical) |
| `cosine-bert` | Cosine similarity via SentenceTransformers (`all-mpnet-base-v2`) |
| `bert-score` | BERT F1 score (`roberta-large`) |
| `nli` | NLI entailment probability (SummaC-ZS style, `roberta-large-mnli`) |

---

## Running an Experiment

```bash
# Single run (uses active_profile in configs.yaml)
uv run experiment.py

# 10 sequential runs of the same profile
bash run.sh

# Watch the live score plot (enable in configs.yaml first)
# visualization.live_plot.enabled: true
uv run experiment.py
```

Output is written to `experiments/<profile>/run_NNNN/`:

```
experiments/
└── baseline_social/
    └── run_0001/
        ├── experiment.json        # per-agent scores and summaries at every snapshot
        ├── metadata.json          # full config snapshot for reproducibility
        ├── scores_over_time.png   # average score trajectory
        ├── live_plot.png          # live plot capture (if enabled)
        └── similarity_matrices/
            ├── agent_1_cosine.png # cosine heatmap: summary sentences vs GT facts
            ├── agent_1_bm25.png   # BM25 heatmap
            └── ...
```

---

## How the Simulation Works

### Agent Lifecycle (per tick)

1. **Proximity scan** — the agent lists all nearby knowledge agents and subject agents.
2. **Subject encounter (object FSM)** — if a visible subject is nearby and the agent is not busy with an LLM task, it reads the subject's snippet into its private context buffer `p`.  It then submits an async LLM task to integrate that snippet into its current summary.
3. **Peer encounter (surrounding FSM)** — if another knowledge agent is nearby and social learning is enabled, the agent exchanges its most recent summary with that peer and submits an async LLM fusion task.
4. **LLM polling** — completed async tasks are drained; their results update the agent's `t_summary` buffer.

### LLM Fusion Prompts

Two prompt templates govern knowledge integration:

- **Private info prompt** — merges a newly observed snippet with the agent's existing summary.
- **Interaction prompt** — merges a peer's summary with the agent's existing summary.

Both prompts enforce lossless merging: every detail already in the prior summary must be preserved or made more specific; nothing is dropped unless directly contradicted.

### Information Teleportation Modes

These modes control whether and how subjects appear/disappear, letting the simulation study knowledge acquisition under partial observability:

| Mode | Behaviour |
|---|---|
| *(none)* | All subjects always visible |
| `decay` | Each subject is assigned an independent exponential lifetime; permanently disappears when it expires |
| `dynamic_pool` | Subjects expire exponentially and new ones appear from a pool of unseen snippets |
| `constant_ratio_pool` | A fixed fraction of all subjects is always visible; expired subjects are immediately replaced |
| `exponential_swap_pool` | A single Poisson timer fires and swaps one active subject for one inactive one |
| `exponential_one_time_pool` | Like swap pool but retired subjects can never re-enter the environment |

---

## Analysing Results

After one or more runs, use the post-processing scripts:

```bash
# Plot average scores for a single swarm experiment (across multiple runs)
uv run plot_swarm_experiment_averages.py

# Compare two or more experiment profiles side by side
uv run plot_multi_experiment_averages.py

# Compare self-learning vs social-learning within a profile family
uv run plot_swarm_group_scores.py

# Single agent vs full swarm comparison
uv run plot_single_agent_vs_swarm_self.py
uv run plot_single_agent_vs_swarm_social.py

# Statistical significance tests across runs
uv run statistical_analysis.py
uv run statistical_tests.py
uv run run_all_stats.py
```

The `analysis/` folder contains additional profiling scripts for aggregating scores across experiment families.

---

## Running on HPC (NYUAD Jubail / SLURM)

The `hpc/` folder contains scripts for running batched experiments on a SLURM cluster with NVIDIA A100 GPUs.

```bash
# One-time environment setup on the cluster
bash hpc/setup_env.sh

# Submit a job array (25 parallel runs, each on 1 A100 + 64 CPUs)
sbatch hpc/submit_experiment.slurm
```

The SLURM script:
- Loads the `ollama/0.12.6` and `python/3.13` modules
- Starts an `ollama serve` instance on a unique port per array task (to avoid collisions when multiple tasks land on the same node)
- Pulls `gemma4:e4b` to `/scratch/<user>/.ollama/` on first use
- Runs `python experiment.py` and cleans up Ollama on exit

Logs are written to `hpc/logs/<job_id>_<task_id>.{out,err}`.

---

## Project File Reference

| File | Purpose |
|---|---|
| `experiment.py` | Main entry point; configures and launches the simulation |
| `agents.py` | `knowledgeAgent` — the core exploring/learning agent |
| `subjects.py` | `SubjectAgent` — passive information carrier |
| `environment.py` | `Environment` — simulation loop, scoring, persistence, teleportation |
| `sensors.py` | `Sensor` (proximity/border reads) and `Actuator` (movement) |
| `llm.py` | Async LLM wrapper supporting Ollama, OpenAI, and vLLM |
| `metrics.py` | Cosine, BM25, BERTScore, NLI scorers; heatmap renderer |
| `constants.py` | System prompts (v1/v2/v3) and ground-truth library |
| `story_registry.py` | Alternative "Lost Artifact of Eldoria" narrative |
| `runtime_config.py` | Resolves active profile and returns flat settings dict |
| `helpers.py` | YAML loading utilities |
| `visualize.py` | `LivePlot` — real-time score plot during the simulation |
| `world.py` | Seeded environment builder for obstacles and sites |
| `communication.py` | Low-level inter-agent message utilities |
| `configs/configs.yaml` | All experiment profiles and global settings |
| `run.sh` | Runs the active profile 10 times sequentially |
| `hpc/submit_experiment.slurm` | SLURM job array script |
| `hpc/setup_env.sh` | Cluster environment bootstrap |
| `process_experiment.py` | Post-processes a single run directory |
| `plot_*.py` | Various comparison and visualisation scripts |
| `statistical_*.py` | Statistical significance tests |

---

## Extending the System

**New experiment profile** — add a block under `experiments.profiles` in `configs/configs.yaml` and set `active_profile` to its key.

**New ground-truth scenario** — add an entry to `GROUND_TRUTH_LIBRARY` in `constants.py`. Each entry needs `name`, `snippets` (list of strings), `text`, `summary`, and `facts`.

**New LLM backend** — extend the `LLM` class in `llm.py`. The public interface is `submit_interaction()`, `submit_private_info()`, and `poll()`.

**New metric** — add a branch in `environment.py → record_snapshot()` and implement the scorer in `metrics.py`.

**New information teleportation mode** — add an `initialize_*` and `*_update` method pair to `Environment`, register the mode string in `experiment.py` and `environment.py → run()`.

---

## Acknowledgements

- [Violet](https://github.com/m-rots/violet) — the underlying 2D agent simulation framework.
- [SentenceTransformers](https://www.sbert.net/) — semantic embeddings (`all-mpnet-base-v2`).
- [BERTScore](https://github.com/Tiiiger/bert_score) — reference-based text quality evaluation.
- [Ollama](https://ollama.com/) — local LLM serving.
- NYUAD HPC (Jubail) — compute resources for large-scale runs.

---

## License

MIT. Please respect the licenses of any third-party models you use (Gemma, GPT-4o, etc.).
