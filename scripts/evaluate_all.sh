#!/usr/bin/env bash
# Évaluation du chapitre 3 : critères de Régularité et d'Arbitraire.
# Usage : bash scripts/evaluate_all.sh [options de SSD_eval_arbitrary.py]
#   ex. bash scripts/evaluate_all.sh --use-original-names   (poids publiés sur Zenodo)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

USE_ORIGINAL=0
for arg in "$@"; do if [ "$arg" = "--use-original-names" ]; then USE_ORIGINAL=1; fi; done

# 1. Modèles « vide sémantique » : régularité par instance + arbitraire entre les 3 graines
python SSD_eval/SSD_eval_arbitrary.py --all --dataset-csv "$GWT_EVALSET_DIR/attributes.csv" "$@"

# 2. Modèles Complets (contrôles) : régularité seule
#    Chemin du checkpoint lu dans experiments/thesis_runs.yaml (noms d'origine ou nouveaux noms)
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
        python SSD_eval/SSD_eval_regularity.py --checkpoint "$ckpt" --with-colors \
            --name "${exp}_control" --dataset-csv "$GWT_EVALSET_DIR/attributes.csv" < /dev/null
    else
        echo "Contrôle absent, ignoré : $ckpt"
    fi
done
echo "Résultats dans $GWT_RESULTS_DIR/"
