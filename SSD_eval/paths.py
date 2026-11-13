"""Chemins centralisés du projet.

Tous les scripts lisent leurs chemins ici. Chaque chemin peut être surchargé
par une variable d'environnement (voir `.env.example`), sinon il est relatif
à la racine du dépôt. Plus aucun chemin absolu propre à une machine.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default)).expanduser().resolve()


# Fichiers de configuration (main.yaml + un fichier par expérience)
CONFIG_DIR = _path("GWT_CONFIG_DIR", REPO_ROOT / "config")

# Données
DATA_DIR = _path("GWT_DATA_DIR", REPO_ROOT / "data")
# Simple Shapes Dataset d'entraînement (500 000 paires), cf. mémoire §2.1.1
DATASET_DIR = _path("GWT_DATASET_DIR", DATA_DIR / "simple_shapes_dataset")
# Jeu de test structuré (153 600 images), cf. mémoire §2.1.2
EVALSET_DIR = _path("GWT_EVALSET_DIR", DATA_DIR / "evaluation_set")

# Modules de domaine pré-entraînés (VAE visuel gelé) : domain_v.ckpt, domain_attr.ckpt
PRETRAINED_DIR = _path("GWT_PRETRAINED_DIR", REPO_ROOT / "checkpoints" / "pretrained")
DOMAIN_V_CKPT = PRETRAINED_DIR / "domain_v.ckpt"
DOMAIN_ATTR_CKPT = PRETRAINED_DIR / "domain_attr.ckpt"

# Global Workspaces entraînés (un sous-dossier par run)
RUNS_DIR = _path("GWT_RUNS_DIR", REPO_ROOT / "checkpoints" / "runs")

# Sorties des évaluations (distributions, tests KS, figures)
RESULTS_DIR = _path("GWT_RESULTS_DIR", REPO_ROOT / "results")

# Manifeste des runs utilisés dans le mémoire
THESIS_RUNS_FILE = REPO_ROOT / "experiments" / "thesis_runs.yaml"


def gw_checkpoint(run_dir_name: str, ckpt_name: str = "last.ckpt") -> Path:
    """Chemin du checkpoint d'un Global Workspace entraîné."""
    return RUNS_DIR / run_dir_name / "checkpoints" / ckpt_name
