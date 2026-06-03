# LLM-Augmented Swarm Intelligence (LASI): Knowledge Acquisition in Multi-Agent Systems

A research simulation that studies how groups of LLM-enabled agents collectively reconstruct a hidden narrative from distributed information sources — and whether sharing knowledge with peers helps or hurts that process.

Built on top of [Violet](https://github.com/m-rots/violet), a lightweight 2D agent simulation framework.

---

## Research Questions

This project investigates two nested questions:

1. **Self vs. social learning** — Does an agent swarm that shares summaries peer-to-peer reconstruct a hidden ground-truth narrative faster and more accurately than a swarm where each agent only accumulates its own observations?

2. **Effect of information scarcity** — How does that self/social gap change when information sources appear and disappear over time, rather than remaining permanently available?

The experiment profiles are treatments on these two variables. You run the same profile twice — once in `self` mode, once in `social` mode — and compare the resulting coverage trajectories.

---

## How the Simulation Works

Agents roam a 2D grid populated by **subject agents** — stationary NPCs each carrying one snippet of a larger ground-truth narrative (a career fair scene). When a knowledge agent steps near a subject, it reads that snippet and submits it to an LLM, which integrates it into the agent's running summary. In social learning mode, agents that come near each other also exchange summaries and fuse them via LLM.

At fixed time intervals, the simulation scores every agent's summary against the ground truth using NLI entailment (`cross-encoder/nli-deberta-v3-large`). A claim is "covered" if at least one sentence in the agent's summary entails it. The experiment ends after a configured number of snapshots.

### Agent lifecycle (per tick)

1. **Proximity scan** — the agent lists all nearby knowledge agents and subject agents.
2. **Subject encounter** — if a visible subject is nearby and the agent is not busy with an LLM task, it reads the subject's snippet and submits an async LLM task to integrate it into its summary.
3. **Peer encounter** — if social learning is enabled and another knowledge agent is nearby, the agent exchanges its most recent summary with that peer and submits an async LLM fusion task.
4. **LLM polling** — completed async tasks are drained; their results update the agent's summary buffer.

### LLM fusion

Two prompt templates govern knowledge integration:

- **Private info prompt** — merges a newly observed snippet with the agent's existing summary.
- **Interaction prompt** — merges a peer's summary with the agent's existing summary.

Both enforce lossless merging: every detail in the prior summary must appear in the output with equal or greater specificity. Details are only dropped if directly contradicted by new information.

### Information teleportation

The three experiment profiles control how visible subject agents are over time:

| Profile | Behaviour |
|---|---|
| `baseline` | All subjects are always visible. Full information available throughout. |
| `dynamic_pool_with_reemission` | Only a fraction of subjects are visible at any time. A Poisson timer fires and swaps one active subject for one inactive one. Retired subjects can re-enter later. |
| `dynamic_pool_without_reemission` | Same as above, but each subject can only ever appear once. Once retired, it is gone permanently. |

The `active_ratio` and `mean_swap_time` parameters in `configs.yaml` control what fraction of subjects are visible and how frequently swaps occur.

---

## Architecture

```
experiment.py          ← entry point; wires everything together
├── agents.py          ← knowledgeAgent: FSM-driven; senses, shares, summarises
├── subjects.py        ← SubjectAgent: carries one ground-truth snippet
├── environment.py     ← Environment: snapshot loop, scoring, teleportation, persistence
├── sensors.py         ← Sensor (proximity reads) + Actuator (movement)
├── llm.py             ← async LLM wrapper: Ollama / OpenAI
├── prompts.py         ← system prompt + ground-truth library (3 difficulty tiers)
├── runtime_config.py  ← reads configs.yaml and resolves the active experiment
└── configs/
    └── configs.yaml   ← all experiment profiles and global settings
```

Output from each run lands in `experiments/<profile>__<mode>/run_NNNN/`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.13+ | enforced by `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | package manager / venv runner |
| An LLM backend | Ollama (local, default) or OpenAI (cloud) — see below |

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

### LLM backend (`.env`)

```dotenv
# ── Ollama (local, default) ──────────────────────────────
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e4b

# ── OpenAI (cloud) ───────────────────────────────────────
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o

# Parallel LLM threads (tune to your hardware)
LLM_MAX_WORKERS=200
```

When using **Ollama**, start the server before running:

```bash
ollama serve          # in one terminal
ollama pull gemma4:e4b
```

### Experiment profile (`configs/configs.yaml`)

Two top-level keys control what runs:

```yaml
active_experiment: baseline       # which profile to use
learning_mode: social             # self | social
```

`learning_mode` is a global toggle — it is not embedded in the profile. To compare self vs. social on the same experiment, run the simulation twice, flipping this value between runs.

**Available profiles:**

| Profile | Description |
|---|---|
| `baseline` | All subjects always visible throughout the run |
| `dynamic_pool_with_reemission` | Subjects rotate in and out via a Poisson timer; retired subjects can re-enter |
| `dynamic_pool_without_reemission` | Same rotation, but each subject can only appear once |

**Global defaults** (override per-profile if needed):

```yaml
defaults:
  num_knowledge_agents: 10
  ground_truth_key: ground_truth_1   # ground_truth_1 | ground_truth_2 | ground_truth_3
  environment:
    width: 750
    height: 750
  num_snapshots: 25
  snapshot_interval_seconds: 2.0     # total sim time = num_snapshots × snapshot_interval_seconds
```

**Teleportation parameters** (for the two pool profiles):

```yaml
information_teleportation:
  active_ratio: 0.2        # fraction of subjects visible at any time
  mean_swap_time: 10.0     # average seconds between Poisson swap events
```

### Ground-truth scenarios

| Key | Name | Snippets | Complexity |
|---|---|---|---|
| `ground_truth_1` | career_fair_low | 10 | Low — broad scene overview |
| `ground_truth_2` | career_fair_medium | 20 | Medium — adds demos, workshops, swag |
| `ground_truth_3` | career_fair_high | 40 | High — full production detail |

All three describe the same career fair at increasing levels of detail. Use a lower key for faster iteration and a higher key to test reconstruction under information overload.

---

## Running an Experiment

### Single run

```bash
uv run experiment.py
```

### Compare self vs. social learning

```bash
# Step 1 — run in self-learning mode
# In configs.yaml: learning_mode: self
uv run experiment.py

# Step 2 — run in social-learning mode
# In configs.yaml: learning_mode: social
uv run experiment.py
```

### Multiple sequential runs of the same profile

```bash
bash run.sh        # defaults to 10 runs
bash run.sh 25     # run a custom number of times
```

### Output structure

```
experiments/
└── baseline__social/
    └── run_0001/
        ├── run.json        # per-agent scores and summaries at every snapshot
        ├── metadata.json   # full config snapshot for reproducibility
        └── run.log         # structured log of the run
```

---

## Analysing Results

### Coverage over time (`data_processing/process_experiment.py`)

Re-scores every agent snapshot using `cross-encoder/nli-deberta-v3-large` and plots how well agents' summaries cover the ground-truth claims over time. A claim is counted as covered when at least one sentence in the agent's summary entails it.

```bash
# Score and plot a single run
uv run data_processing/process_experiment.py experiments/baseline__social/run_0001

# Compare two runs side by side
uv run data_processing/process_experiment.py \
    experiments/baseline__self/run_0001 \
    experiments/baseline__social/run_0001

# Overlay runs on a single axes with custom labels
uv run data_processing/process_experiment.py \
    experiments/baseline__self/run_0001 \
    experiments/baseline__social/run_0001 \
    --overlay \
    --legend-labels "Self" "Social"
```

**Key flags:**

| Flag | Description |
|---|---|
| `--output PATH` | Where to save the PNG |
| `--overlay` | All runs on one axes instead of side-by-side subplots |
| `--full-range` | Show the full x-axis; default clips at the plateau |
| `--title TEXT` | Custom plot title |
| `--legend-labels L1 L2 …` | Custom labels, one per run folder |
| `--xlim X_MIN X_MAX` | Override x-axis range |
| `--ylim Y_MIN Y_MAX` | Override y-axis range |
| `--figsize W H` | Figure size in inches |
| `--no-legend` | Hide the legend |
| `--no-grid` | Hide the background grid |

> The NLI model (~1.5 GB) is downloaded from HuggingFace on first use and cached locally.

### Aggregate across configs (`data_processing/process_experiment_configs.py`)

Aggregates and plots results across multiple experiment configurations for cross-condition comparisons.

```bash
uv run data_processing/process_experiment_configs.py
```

---

## Extending the System

### New experiment profile

Add a block under `experiments` in `configs/configs.yaml`:

```yaml
experiments:
  my_profile:
    information_teleportation:
      enabled: true
      mode: dynamic_pool_with_reemission
      active_ratio: 0.5
      mean_swap_time: 5.0
```

Then set `active_experiment: my_profile`.

### New ground-truth scenario

Add an entry to `GROUND_TRUTH_LIBRARY` in `prompts.py`:

```python
GROUND_TRUTH_LIBRARY["my_scenario"] = {
    "name": "my_scenario",
    "snippets": [...],   # list of strings, one per subject agent
    "text": "...",       # full narrative text
    "summary": "...",    # condensed version
    "facts": "...",      # scoring facts as a single string
}
```

Then set `ground_truth_key: my_scenario` in `configs.yaml`.

### New LLM provider

Extend the `LLM` class in `llm.py`. The public interface is:

- `submit_interaction(summary, received_info)` → `Future[str]`
- `submit_private_info(summary, private_info)` → `Future[str]`

### New teleportation mode

Add an `initialize_*` and `*_update` method pair to `Environment` in `environment.py`, then register the mode string in `experiment.py` and `environment.py → run()`.

---

## Project File Reference

| File | Purpose |
|---|---|
| `experiment.py` | Entry point; configures and launches the simulation |
| `agents.py` | `knowledgeAgent` — exploring/learning agent with proximity FSM |
| `subjects.py` | `SubjectAgent` — stationary information carrier |
| `environment.py` | Simulation loop, snapshot scoring, teleportation, persistence |
| `sensors.py` | `Sensor` (proximity reads) and `Actuator` (movement) |
| `llm.py` | Async LLM wrapper supporting Ollama and OpenAI |
| `prompts.py` | System prompt and ground-truth library |
| `runtime_config.py` | Resolves active profile and returns flat settings dict |
| `context.py` | Agent context buffer utilities |
| `helpers.py` | YAML loading utilities |
| `logging_setup.py` | Structured logging configuration |
| `configs/configs.yaml` | All experiment profiles and global settings |
| `run.sh` | Runs the active profile 10 times sequentially |
| `data_processing/process_experiment.py` | NLI coverage scoring and plotting for one or more runs |
| `data_processing/process_experiment_configs.py` | Aggregate plotting across multiple configurations |

---

## Acknowledgements

- [Violet](https://github.com/m-rots/violet) — the underlying 2D agent simulation framework.
- [SentenceTransformers](https://www.sbert.net/) — semantic embeddings.
- [Ollama](https://ollama.com/) — local LLM serving.
