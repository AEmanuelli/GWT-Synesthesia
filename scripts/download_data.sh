#!/usr/bin/env bash
# Étape 2 — Télécharge le Simple Shapes Dataset, les modules de domaine pré-entraînés,
# pré-calcule les latents du VAE visuel et génère le jeu de test structuré.
# Usage : bash scripts/download_data.sh [--eval-only] [--quick-evalset]
#   --eval-only      : seulement ce qu'il faut pour évaluer des poids existants
#                      (VAE pré-entraîné + jeu de test), sans le dataset d'entraînement
#   --quick-evalset  : mini-jeu de test (192 images) pour vérifier le pipeline
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

EVAL_ONLY=0; QUICK=0
for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=1 ;;
        --quick-evalset) QUICK=1 ;;
        *) echo "Option inconnue : $arg"; exit 1 ;;
    esac
done

# 2a. Simple Shapes Dataset (Devillers et al., Zenodo 8112838)
if [ "$EVAL_ONLY" = 1 ]; then
    echo "--eval-only : dataset d'entraînement ignoré"
elif [ ! -d "$GWT_DATASET_DIR" ]; then
    # `shapesd download` extrait toujours dans <path>/simple_shapes_dataset
    if [ "$(basename "$GWT_DATASET_DIR")" != "simple_shapes_dataset" ]; then
        echo "GWT_DATASET_DIR doit se terminer par /simple_shapes_dataset (reçu : $GWT_DATASET_DIR)"; exit 1
    fi
    mkdir -p "$(dirname "$GWT_DATASET_DIR")"
    shapesd download --path "$(dirname "$GWT_DATASET_DIR")"
else
    echo "Dataset déjà présent : $GWT_DATASET_DIR"
fi

# 2b. Modules de domaine pré-entraînés (VAE visuel gelé) — Zenodo 14747474
if [ ! -f "$GWT_PRETRAINED_DIR/domain_v.ckpt" ]; then
    mkdir -p "$GWT_PRETRAINED_DIR"
    tmp="$(mktemp -d)"
    curl -L --fail -o "$tmp/ckpt.tar.gz" \
        "https://zenodo.org/records/14747474/files/simple_shapes_checkpoints.tar.gz"
    tar -xzf "$tmp/ckpt.tar.gz" -C "$tmp"
    for f in domain_v.ckpt domain_attr.ckpt; do
        src="$(find "$tmp" -name "$f" | head -n1)"
        if [ -n "$src" ]; then mv "$src" "$GWT_PRETRAINED_DIR/$f"; else echo "ATTENTION : $f introuvable dans l'archive"; fi
    done
    rm -rf "$tmp"
else
    echo "Modules pré-entraînés déjà présents : $GWT_PRETRAINED_DIR"
fi

# 2c. Latents visuels pré-calculés (lus par le dataloader : saved_latents/<split>/vae_v_shimmer.npy)
if [ "$EVAL_ONLY" = 0 ] && [ ! -f "$GWT_DATASET_DIR/saved_latents/train/vae_v_shimmer.npy" ]; then
    # Équivalent de `ssd extract v`, avec la compatibilité CPU du dépôt (VAE sauvegardé sur GPU)
    python - "$GWT_PRETRAINED_DIR/domain_v.ckpt" "$REPO_ROOT/config" "$GWT_DATASET_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "SSD_eval")
import device_compat  # noqa: F401
from shimmer_ssd.cli.extract import save_v_latents
checkpoint, config_dir, dataset_dir = map(Path, sys.argv[1:4])
save_v_latents(checkpoint, config_dir, dataset_dir, "vae_v_shimmer.npy")
PY
fi

# 2d. Jeu de test structuré du mémoire (153 600 images ; ~1 à 2 h sur CPU)
if [ "$QUICK" = 1 ]; then
    python SSD_eval/SSD_generate_evalset.py --quick
elif [ ! -f "$GWT_EVALSET_DIR/attributes.csv" ]; then
    python SSD_eval/SSD_generate_evalset.py --output-dir "$GWT_EVALSET_DIR"
fi
echo "Données prêtes."
