#!/bin/bash

#SBATCH --account=chmdsh_project
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=Bootstrap_MACE

set -euo pipefail

PACKAGE_ROOT="$1"
CONFIG="$2"

cd "$PACKAGE_ROOT"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env

PYTHON="$(command -v python)"

echo "=============================================="
echo "MACE BOOTSTRAP TRAINING"
echo "=============================================="
echo "Python: $PYTHON"
echo "Config: $CONFIG"
echo ""

echo "===== GPU CHECK ====="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "===================="

echo ""
echo "Testing PyTorch..."

"$PYTHON" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available. MACE training will not run.")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
PY

echo ""
echo "Starting bootstrap MACE training..."

exec "$PYTHON" "$PACKAGE_ROOT/bootstrap_train.py" \
    --config "$CONFIG"
