#!/usr/bin/env bash
# Étape 4 — Critères de Régularité et d'Arbitraire sur les modèles entraînés.
# Usage : bash scripts/evaluate_all.sh [args supplémentaires pour SSD_eval_arbitrary.py]
#   ex. bash scripts/evaluate_all.sh --use-original-names   (poids publiés du stage)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

# 4a. Modèles "Vide Sémantique" : régularité par instance + arbitraire entre les 3 graines
python SSD_eval/SSD_eval_arbitrary.py --all --dataset-csv "$GWT_EVALSET_DIR/attributes.csv" "$@"

# 4b. Modèles Complets (contrôles) : régularité seule
for exp in base_params high_translation high_contrastive high_cycles high_alpha; do
    ckpt="$GWT_RUNS_DIR/${exp}_control/checkpoints/last.ckpt"
    if [ -e "$ckpt" ]; then
        python SSD_eval/SSD_eval_regularity.py --checkpoint "$ckpt" --with-colors \
            --name "${exp}_control" --dataset-csv "$GWT_EVALSET_DIR/attributes.csv"
    else
        echo "Contrôle absent, ignoré : $ckpt"
    fi
done
echo "Résultats dans $GWT_RESULTS_DIR/"
