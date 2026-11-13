#!/usr/bin/env bash
# Pipeline complet : données -> entraînement -> évaluation.
set -euo pipefail
DIR="$(dirname "$0")"
bash "$DIR/download_data.sh"
bash "$DIR/train_all.sh"
bash "$DIR/evaluate_all.sh"
