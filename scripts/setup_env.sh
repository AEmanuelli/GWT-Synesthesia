#!/usr/bin/env bash
# Étape 1 — Crée un environnement virtuel Python 3.11 et installe les dépendances.
# Usage : bash scripts/setup_env.sh            (crée .venv/)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3.11}"
command -v "$PYTHON" >/dev/null || { echo "Python 3.11 requis (variable PYTHON=...)"; exit 1; }

"$PYTHON" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Si shimmer a aussi été modifié localement, SHIMMER_FORK=git+https://... le remplace
if [ -n "${SHIMMER_FORK:-}" ]; then pip install --no-deps --force-reinstall "$SHIMMER_FORK"; fi
# Version modifiée de shimmer-ssd (sans dépendances : elles sont figées ci-dessus)
pip install --no-deps "$SHIMMER_SSD_FORK"

python - <<'PY'
from shimmer_ssd.config import DomainModuleVariant
assert hasattr(DomainModuleVariant, "attr_legacy_no_color"), (
    "shimmer-ssd installé n'est PAS la version modifiée (attr_legacy_no_color manquant). "
    "Voir README §2.")
print("OK : environnement prêt.")
PY
echo "Activez-le avec : source .venv/bin/activate"
