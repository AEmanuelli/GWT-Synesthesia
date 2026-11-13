# Variables partagées par les scripts (chargées avec `source`).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
GWT_DATA_DIR="${GWT_DATA_DIR:-$REPO_ROOT/data}"
GWT_DATASET_DIR="${GWT_DATASET_DIR:-$GWT_DATA_DIR/simple_shapes_dataset}"
GWT_EVALSET_DIR="${GWT_EVALSET_DIR:-$GWT_DATA_DIR/evaluation_set}"
GWT_PRETRAINED_DIR="${GWT_PRETRAINED_DIR:-$REPO_ROOT/checkpoints/pretrained}"
GWT_RUNS_DIR="${GWT_RUNS_DIR:-$REPO_ROOT/checkpoints/runs}"
GWT_RESULTS_DIR="${GWT_RESULTS_DIR:-$REPO_ROOT/results}"
SHIMMER_SSD_FORK="${SHIMMER_SSD_FORK:-git+https://github.com/AEmanuelli/shimmer-ssd.git@21f8271f163ec72950e147ebe1e192d467ef7f2a}"
export GWT_DATA_DIR GWT_DATASET_DIR GWT_EVALSET_DIR GWT_PRETRAINED_DIR GWT_RUNS_DIR GWT_RESULTS_DIR
