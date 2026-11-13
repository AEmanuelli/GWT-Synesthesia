#!/usr/bin/env bash
# Télécharge les Global Workspaces entraînés pendant le stage (ceux du mémoire)
# depuis Zenodo (record 24208192) et les place dans $GWT_RUNS_DIR sous leurs noms d'origine.
# Usage : bash scripts/download_checkpoints.sh
#   puis  bash scripts/evaluate_all.sh --use-original-names
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ZENODO_RECORD="${ZENODO_RECORD:-24208192}"
if [ -z "${GWT_CKPT_URL:-}" ]; then
    # Retrouve l'URL de l'archive .tar.gz via l'API Zenodo (indépendant du nom du fichier)
    GWT_CKPT_URL="$(curl -sSL --fail "https://zenodo.org/api/records/$ZENODO_RECORD" | python3 -c '
import json, sys
files = json.load(sys.stdin)["files"]
tars = [f for f in files if f["key"].endswith(".tar.gz")]
if len(tars) != 1:
    keys = [f["key"] for f in files]
    sys.exit("Archive .tar.gz introuvable ou multiple dans le record : " + str(keys))
print(tars[0]["links"]["self"])
')"
fi
echo "Téléchargement : $GWT_CKPT_URL"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
curl -L --fail -o "$tmp/ckpt.tar.gz" "$GWT_CKPT_URL"
tar -xzf "$tmp/ckpt.tar.gz" -C "$tmp"
(cd "$tmp" && sha256sum -c SHA256SUMS --quiet) && echo "Sommes de contrôle OK"
mkdir -p "$GWT_RUNS_DIR"
cp -r "$tmp/runs/." "$GWT_RUNS_DIR/"
echo "$(find "$GWT_RUNS_DIR" -name '*.ckpt' | wc -l) checkpoints installés dans $GWT_RUNS_DIR"
