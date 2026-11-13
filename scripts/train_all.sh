#!/usr/bin/env bash
# Étape 3 — Entraîne les 5 configurations x 3 graines (Vide Sémantique)
# + 1 Modèle Complet de contrôle par configuration = 20 runs de 300 000 pas.
# Usage : bash scripts/train_all.sh [experiment ...]   (défaut : les 5)
#         MAX_STEPS=2000 bash scripts/train_all.sh base_params   (test rapide)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

if [ $# -gt 0 ]; then EXPERIMENTS=("$@")
else EXPERIMENTS=(base_params high_translation high_contrastive high_cycles high_alpha); fi
EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")

for exp in "${EXPERIMENTS[@]}"; do
    alpha=1; [ "$exp" = "high_alpha" ] && alpha=5
    for seed in 0 1 2; do
        if [ -e "$GWT_RUNS_DIR/${exp}_seed${seed}/checkpoints/last.ckpt" ]; then
            echo "== $exp seed $seed déjà entraîné, on passe"; continue
        fi
        echo "== $exp | seed $seed | vide sémantique"
        python SSD_eval/SSD_Train.py --experiment "$exp" --seed "$seed" --attr-alpha "$alpha" ${EXTRA[@]+"${EXTRA[@]}"}
    done
    if [ ! -e "$GWT_RUNS_DIR/${exp}_control/checkpoints/last.ckpt" ]; then
        echo "== $exp | contrôle (Modèle Complet)"
        python SSD_eval/SSD_Train.py --experiment "$exp" --seed 0 --attr-alpha "$alpha" --with-colors ${EXTRA[@]+"${EXTRA[@]}"}
    fi
done
