#!/usr/bin/env bash
# Étape 1 — Crée un environnement virtuel Python 3.11 et installe les dépendances.
# Usage : bash scripts/setup_env.sh            (crée .venv/)
set -euo pipefail
GWT_SKIP_VENV_CHECK=1  # le venv est (re)créé ici
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3.11}"
command -v "$PYTHON" >/dev/null || { echo "Python 3.11 requis (variable PYTHON=...)"; exit 1; }

"$PYTHON" -m venv --clear .venv  # --clear : repart de zéro (ex. venv déplacé)
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Fork shimmer-ssd de l'auteur (sans dépendances : elles sont figées dans requirements.txt)
pip install --no-deps "$SHIMMER_SSD_FORK"

python - <<'PY'
from shimmer_ssd.config import DomainModuleVariant
assert hasattr(DomainModuleVariant, "attr_legacy_no_color"), (
    "shimmer-ssd installé n'est PAS la version modifiée (attr_legacy_no_color manquant). "
    "Relancer scripts/setup_env.sh.")
print("OK : environnement prêt.")
PY
echo "Activez-le avec : source .venv/bin/activate"
