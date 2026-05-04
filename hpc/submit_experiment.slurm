#!/bin/bash
#SBATCH --job-name=capstone_exp
#SBATCH --output=hpc/logs/%A_%a.out
#SBATCH --error=hpc/logs/%A_%a.err
#SBATCH --array=1-25
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bab9755@nyu.edu

# ─────────────────────────────────────────────────────────
# 0.  Environment bootstrap
# ─────────────────────────────────────────────────────────
echo "======================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Array task    : $SLURM_ARRAY_TASK_ID  / $SLURM_ARRAY_TASK_MAX"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "Working dir   : $SLURM_SUBMIT_DIR"
echo "======================================================"

PROJECT_DIR="$SLURM_SUBMIT_DIR/.."
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

module purge
module load ollama/0.12.6
module load python/3.13              # adjust version name if needed

source .venv/bin/activate

# ─────────────────────────────────────────────────────────
# 1.  Ollama server setup
#
#     Each array task gets its own port to avoid collisions
#     when multiple tasks land on the same compute node.
#     Models are stored in /scratch to avoid home-dir quota.
# ─────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}"
OLLAMA_MODEL_DIR="${SCRATCH_DIR}/.ollama"
mkdir -p "${OLLAMA_MODEL_DIR}/models"

# Unique port: base 11434 + task ID offset
OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST="http://127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_MODELS="${OLLAMA_MODEL_DIR}/models"

# Allow multiple concurrent requests (tune to your GPU VRAM)
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1

echo "--- Starting ollama serve on port ${OLLAMA_PORT} ---"
ollama serve &
OLLAMA_PID=$!

# Ensure ollama is killed when this job exits (success or failure)
trap "echo 'Shutting down ollama (pid ${OLLAMA_PID})'; kill ${OLLAMA_PID} 2>/dev/null; wait ${OLLAMA_PID} 2>/dev/null" EXIT

# Wait until the server is accepting connections
echo "--- Waiting for ollama server to be ready ---"
for i in $(seq 1 30); do
    if ollama list &>/dev/null; then
        echo "--- Ollama ready after ${i}s ---"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: ollama did not start within 30s"
        exit 1
    fi
    sleep 1
done

# Pull the model only if it isn't already cached in SCRATCH
MODEL="gemma4:e4b"
if ! ollama list | grep -q "${MODEL}"; then
    echo "--- Pulling ${MODEL} (first time only, cached to ${OLLAMA_MODELS}) ---"
    ollama pull "${MODEL}"
else
    echo "--- ${MODEL} already cached, skipping pull ---"
fi

# ─────────────────────────────────────────────────────────
# 2.  Provider configuration
# ─────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# Force ollama provider and correct model regardless of .env
export LLM_PROVIDER=ollama
export LLM_MODEL="${MODEL}"

export MPLBACKEND=Agg    # no display on HPC nodes

# ─────────────────────────────────────────────────────────
# 3.  Run the experiment
# ─────────────────────────────────────────────────────────
echo "--- Starting experiment run ${SLURM_ARRAY_TASK_ID} ---"
python experiment.py
EXIT_CODE=$?

echo "--- Finished run ${SLURM_ARRAY_TASK_ID} (exit code: ${EXIT_CODE}) ---"
echo "End time: $(date)"
exit $EXIT_CODE
