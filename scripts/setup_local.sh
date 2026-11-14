#!/usr/bin/env bash
# Prépare tout sur une machine où des fichiers du projet existent déjà :
#   1. installe l'environnement Python (seulement s'il manque ou ne fonctionne pas) ;
#   2. retrouve sur le disque le VAE, les 20 modèles du mémoire et les jeux de test,
#      et les range dans le dépôt (plan affiché, confirmation demandée) ;
#   3. vérifie si le GPU est utilisable.
# Usage : bash scripts/setup_local.sh [dossiers où chercher, ~ par défaut]
set -euo pipefail
GWT_SKIP_VENV_CHECK=1
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

echo "== 1. Environnement Python"
if .venv/bin/python -c "import shimmer_ssd, cv2, torch, yaml" 2>/dev/null; then
    echo "Environnement déjà fonctionnel (.venv)."
else
    bash scripts/setup_env.sh
fi

echo; echo "== 2. Fichiers déjà présents sur le disque"
.venv/bin/python scripts/adopt_existing_files.py "$@"

echo; echo "== 3. GPU"
.venv/bin/python - <<'PY'
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    ok = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
    print(f"GPU détecté : {torch.cuda.get_device_name(0)} (capacité {cap[0]}.{cap[1]})"
          + ("" if ok else " — non pris en charge par cette version de PyTorch, le CPU sera utilisé"))
else:
    print("Pas de GPU utilisable : le calcul se fera sur CPU.")
PY

echo; echo "Terminé. Dans chaque nouveau terminal : source .venv/bin/activate"
