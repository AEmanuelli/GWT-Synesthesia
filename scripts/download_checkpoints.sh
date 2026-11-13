#!/usr/bin/env bash
# Télécharge les Global Workspaces entraînés pendant le stage (ceux du mémoire)
# et les place dans $GWT_RUNS_DIR sous leurs noms d'origine.
# Usage : bash scripts/download_checkpoints.sh
#   puis  bash scripts/evaluate_all.sh --use-original-names
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# TODO(auteur) : remplacer par l'URL de l'archive publiée (Zenodo ou GitHub Release)
GWT_CKPT_URL="${GWT_CKPT_URL:-}"
[ -n "$GWT_CKPT_URL" ] || { echo "Définissez GWT_CKPT_URL (URL de gwt_synesthesia_checkpoints.tar.gz)"; exit 1; }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -L --fail -o "$tmp/ckpt.tar.gz" "$GWT_CKPT_URL"
tar -xzf "$tmp/ckpt.tar.gz" -C "$tmp"
(cd "$tmp" && sha256sum -c SHA256SUMS --quiet) && echo "Sommes de contrôle OK"
mkdir -p "$GWT_RUNS_DIR"
cp -r "$tmp/runs/." "$GWT_RUNS_DIR/"
echo "Checkpoints installés dans $GWT_RUNS_DIR"
