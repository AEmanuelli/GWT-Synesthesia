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

# Garde-fou : si un .venv existe, le `python` utilisé doit être le sien
# (un venv déplacé ou non activé fait retomber sur un autre Python, sans les dépendances).
if [ -d "$REPO_ROOT/.venv" ] && [ "${GWT_SKIP_VENV_CHECK:-0}" != 1 ]; then
    _py_prefix="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
    if [ "$(cd "$REPO_ROOT/.venv" && pwd -P)" != "$(cd "${_py_prefix:-/nonexistent}" 2>/dev/null && pwd -P)" ]; then
        echo "ERREUR : le Python actif ($(command -v python || echo introuvable)) n'est pas celui de $REPO_ROOT/.venv."
        echo "  - environnement non activé :  source .venv/bin/activate"
        echo "  - dossier déplacé depuis la création du venv : rm -rf .venv && bash scripts/setup_env.sh"
        exit 1
    fi
fi
