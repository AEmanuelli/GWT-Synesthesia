#!/usr/bin/env bash
# Étape 2 — Télécharge le Simple Shapes Dataset, les modules de domaine pré-entraînés,
# pré-calcule les latents du VAE visuel et génère le jeu de test structuré.
# Usage : bash scripts/download_data.sh [--quick-evalset]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$REPO_ROOT"

# 2a. Simple Shapes Dataset (Devillers et al., Zenodo 8112838)
if [ ! -d "$GWT_DATASET_DIR" ]; then
    mkdir -p "$(dirname "$GWT_DATASET_DIR")"
    shapesd download --path "$(dirname "$GWT_DATASET_DIR")" --name "$(basename "$GWT_DATASET_DIR")"
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
if [ ! -f "$GWT_DATASET_DIR/saved_latents/train/vae_v_shimmer.npy" ]; then
    ssd extract v "$GWT_PRETRAINED_DIR/domain_v.ckpt" \
        --dataset_path "$GWT_DATASET_DIR" --latent_name vae_v_shimmer.npy --config_path "$REPO_ROOT/config"
fi

# 2d. Jeu de test structuré du mémoire (153 600 images ; ~1 à 2 h sur CPU)
if [ "${1:-}" = "--quick-evalset" ]; then
    python SSD_eval/SSD_generate_evalset.py --quick
elif [ ! -f "$GWT_EVALSET_DIR/attributes.csv" ]; then
    python SSD_eval/SSD_generate_evalset.py --output-dir "$GWT_EVALSET_DIR"
fi
echo "Données prêtes."
