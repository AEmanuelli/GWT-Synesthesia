#!/usr/bin/env bash
# Saturation (S) et luminosité (L) des couleurs reconstruites (prolongement de l'annexe E).
# Refait l'évaluation du chapitre 3 avec S et L en plus de H, R, G, B (option --hls), sans bruit
# puis avec le bruit de l'annexe E (traduction bruitée), pour les 15 modèles « vide sémantique »
# et les Modèles Complets disponibles. Les résultats du chapitre 3 ne sont pas touchés.
#
# --low-memory : images reconstruites gardées en uint8 après l'extraction des couleurs (même
# résultat, environ 4 Go de mémoire vive au lieu de 9 par modèle).
#
# Usage : bash scripts/evaluate_saturation.sh [--use-original-names] [autres options]
# Sorties : results/saturation/arbitrary/, arbitrary_shape_noise_0.01/, regularity/
# Puis :    python experiments/saturation_analysis.py
#           python experiments/make_tables.py results/saturation/arbitrary --channel S
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

USE_ORIGINAL=0
for arg in "$@"; do if [ "$arg" = "--use-original-names" ]; then USE_ORIGINAL=1; fi; done

for noise in 0 0.01; do
    # 1. Modèles « vide sémantique » : régularité par instance + arbitraire entre les 3 graines
    python SSD_eval/SSD_eval_arbitrary.py --all --hls --low-memory --shape-noise "$noise" --shape-noise-seed 42 \
        --dataset-csv "$GWT_EVALSET_DIR/attributes.csv" "$@"

    # 2. Modèles Complets (contrôles)
    suffix=""; if [ "$noise" != 0 ]; then suffix="_shape_noise_$noise"; fi
    python - "$USE_ORIGINAL" <<'PY' | while IFS='|' read -r exp ckpt; do
import sys, yaml
sys.path.insert(0, "SSD_eval")
import paths
original = sys.argv[1] == "1"
for name, exp in yaml.safe_load(open(paths.THESIS_RUNS_FILE))["experiments"].items():
    c = exp["control"]
    run_dir = c["original_run_name"] if original else c["run_dir"]
    ckpt = c.get("original_ckpt", c["ckpt"]) if original else c["ckpt"]
    print(f"{name}|{paths.gw_checkpoint(run_dir, ckpt)}")
PY
        if [ -e "$ckpt" ]; then
            python SSD_eval/SSD_eval_regularity.py --checkpoint "$ckpt" --with-colors --hls --low-memory \
                --shape-noise "$noise" --shape-noise-seed 42 \
                --name "${exp}_control${suffix}" --dataset-csv "$GWT_EVALSET_DIR/attributes.csv" < /dev/null
        else
            echo "Contrôle absent, ignoré : $ckpt"
        fi
    done
done
echo "Résultats dans $GWT_RESULTS_DIR/saturation/"
