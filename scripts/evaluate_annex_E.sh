#!/usr/bin/env bash
# Annexe E du mémoire — « Mais où est encodée la saturation ? »
# Refait l'évaluation Régularité + Arbitraire avec un bruit gaussien de 0.01 (graine 42,
# identique pour toutes les instances) ajouté au one-hot de la forme sur le chemin de
# traduction ("Traduction Bruitée", tableau E.1). Les chemins demi-cycle et cycle complet
# ne sont pas affectés. À comparer aux résultats sans bruit (scripts/evaluate_all.sh).
#
# Usage : bash scripts/evaluate_annex_E.sh [--use-original-names] [autres options]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"
python SSD_eval/SSD_eval_arbitrary.py --all \
    --shape-noise 0.01 --shape-noise-seed 42 \
    --dataset-csv "$GWT_EVALSET_DIR/attributes.csv" "$@"
echo "Résultats dans $GWT_RESULTS_DIR/arbitrary_shape_noise_0.01/"
