# Variables partagées par les scripts (chargées avec `source`).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
GWT_DATA_DIR="${GWT_DATA_DIR:-$REPO_ROOT/data}"
GWT_DATASET_DIR="${GWT_DATASET_DIR:-$GWT_DATA_DIR/simple_shapes_dataset}"
GWT_EVALSET_DIR="${GWT_EVALSET_DIR:-$GWT_DATA_DIR/evaluation_set}"
GWT_PRETRAINED_DIR="${GWT_PRETRAINED_DIR:-$REPO_ROOT/checkpoints/pretrained}"
GWT_RUNS_DIR="${GWT_RUNS_DIR:-$REPO_ROOT/checkpoints/runs}"
GWT_RESULTS_DIR="${GWT_RESULTS_DIR:-$REPO_ROOT/results}"
SHIMMER_SSD_FORK="${SHIMMER_SSD_FORK:-git+https://github.com/AEmanuelli/shimmer-ssd.git@8e615310cab3488e5e79a4ad8b898bde52104aca}"
export GWT_DATA_DIR GWT_DATASET_DIR GWT_EVALSET_DIR GWT_PRETRAINED_DIR GWT_RUNS_DIR GWT_RESULTS_DIR
