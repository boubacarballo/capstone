#!/bin/bash
# Run this script ONCE on the login node to set up the Python environment.
# Usage:  bash hpc/setup_env.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> Setting up environment in: $PROJECT_DIR"

# 1. Load modules
module purge
module load ollama/0.12.6
module load python/3.13    # adjust version name if needed

# 2. Create a virtual environment in .venv/
python -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install all project dependencies
pip install --upgrade pip
pip install -r <(pip-compile pyproject.toml --quiet --output-file /dev/stdout 2>/dev/null) || \
    pip install .        # fallback: plain pip install from pyproject.toml

# 4. Pre-download NLTK data and HuggingFace models used by the metrics
python - <<'PYEOF'
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Trigger a HuggingFace model download so it is cached before the job runs
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PYEOF

# 5. Create log directory for SLURM output files
mkdir -p hpc/logs

echo ""
echo "==> Setup complete. Virtual environment is at: $PROJECT_DIR/.venv"
echo "==> Submit jobs with:  sbatch hpc/submit_experiment.slurm"
