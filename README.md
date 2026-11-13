# GWT-Synesthesia — Synesthésie artificielle dans un modèle Global Workspace

Code du mémoire de M2 d'Alexis Emanuelli :
**« Synesthésie artificielle dans un modèle inspiré de la Global Workspace Theory »**
([PDF](docs/EMANUELLI_Memoire_M2.pdf)).

On entraîne un Global Workspace (GW) bimodal (image ↔ vecteur d'attributs) sur le
[Simple Shapes Dataset](https://github.com/ruflab/simple-shapes-dataset), en retirant
la couleur de la modalité attributs (« vide sémantique »). On teste ensuite si le
modèle développe des associations attribut → couleur **régulières**, **arbitraires**
(spécifiques à chaque instance) et **additionnelles** — notre définition opérationnelle
de la synesthésie artificielle.

---

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide)
2. [Installation](#2-installation)
3. [Données](#3-données)
4. [Entraînement](#4-entraînement)
5. [Évaluation](#5-évaluation)
6. [Correspondance mémoire ↔ code](#6-correspondance-mémoire--code)
7. [Organisation du dépôt](#7-organisation-du-dépôt)
8. [Écarts connus et points d'attention](#8-écarts-connus-et-points-dattention)

---

## 1. Démarrage rapide

```bash
git clone https://github.com/aemanuelli/gwt-synesthesia.git
```

```bash
cd gwt-synesthesia
```

```bash
bash scripts/setup_env.sh          # Python 3.11 + dépendances figées (.venv/)
```

```bash
source .venv/bin/activate
```

```bash
bash scripts/reproduce_all.sh      # données -> 20 entraînements -> évaluation
```

Le pipeline complet est long (20 runs de 300 000 pas, ~1 GPU de 16 Go). Pour vérifier
que tout fonctionne en quelques minutes :

```bash
bash scripts/download_data.sh --quick-evalset
```

```bash
MAX_STEPS=2000 bash scripts/train_all.sh base_params
```

```bash
python SSD_eval/SSD_eval_arbitrary.py --experiment base_params --dataset-csv data/evaluation_set_quick/attributes.csv
```

## 2. Installation

Environnement de référence (celui des runs d'origine) : **Python 3.11, PyTorch 2.5.1,
Lightning 2.5.1, CUDA, 1 GPU Quadro RTX 5000 (16 Go)**. Toutes les versions sont figées
dans [`requirements.txt`](requirements.txt), y compris les bibliothèques du laboratoire
(`shimmer`, `simple-shapes-dataset`, `cfg-tools`), figées sur leurs commits de mai 2025.

Avec `venv` (recommandé) : `bash scripts/setup_env.sh`. Avec conda :

```bash
conda env create -f environment.yml
```

```bash
conda activate gwt-synesthesia
```

```bash
pip install --no-deps "git+https://github.com/aemanuelli/shimmer-ssd.git@gwt-synesthesia"
```

> **Dépendance `shimmer-ssd` modifiée.** Ce projet utilise une version de
> [`shimmer-ssd`](https://github.com/ruflab/shimmer-ssd) modifiée pendant le stage
> (variante de domaine `attr_legacy_no_color`, module attributs paramétré par
> `alpha`/`temperature`, décodeur attributs dédié, correction LogSoftmax → exp → log,
> option `exclude_colors` du callback d'images). Ces modifications ne sont **pas** dans
> la version publique de `ruflab/shimmer-ssd`. Le script `setup_env.sh` vérifie que la
> bonne version est installée. L'URL du fork est réglable avec la variable
> `SHIMMER_SSD_FORK` (voir [`.env.example`](.env.example)).

Les chemins sont tous relatifs à la racine du dépôt par défaut (`data/`,
`checkpoints/`, `results/`) et se règlent par variables d'environnement
(`GWT_DATA_DIR`, `GWT_RUNS_DIR`, …) : voir [`SSD_eval/paths.py`](SSD_eval/paths.py) et
[`.env.example`](.env.example). Aucun chemin absolu n'est codé en dur.

Le logging [Weights & Biases](https://wandb.ai) est optionnel : sans `WANDB_API_KEY`,
les runs sont enregistrés hors-ligne dans le dossier du run.

## 3. Données

`bash scripts/download_data.sh` effectue les quatre étapes suivantes :

| Étape | Contenu | Destination |
|---|---|---|
| Simple Shapes Dataset (500 000 paires, [Zenodo 8112838](https://zenodo.org/records/8112838)) | jeu d'entraînement (mémoire §2.1.1) | `data/simple_shapes_dataset/` |
| Modules de domaine pré-entraînés ([Zenodo 14747474](https://zenodo.org/records/14747474)) | VAE visuel gelé `domain_v.ckpt` (latent 12-d) | `checkpoints/pretrained/` |
| Latents visuels pré-calculés | `ssd extract v` → `saved_latents/<split>/vae_v_shimmer.npy` | dans le dataset |
| Jeu de test structuré (mémoire §2.1.2) | 3 formes × 8 tailles × 16 positions × 16 orientations × 25 teintes = **153 600 images** | `data/evaluation_set/` |

Le jeu de test peut aussi être généré seul :

```bash
python SSD_eval/SSD_generate_evalset.py --output-dir data/evaluation_set
```

## 4. Entraînement

Une configuration = un fichier dans [`config/`](config/). Les coefficients
(α traduction, β contrastive, γ demi-cycle, δ cycle complet) sont ceux du mémoire (§2.3.3) :

| # | Configuration | Fichier | α | β | γ | δ | autre |
|---|---|---|---|---|---|---|---|
| 1 | Référence | `base_params.yaml` | 1 | 0.1 | 1 | 1 | |
| 2 | Emphase Traduction | `high_translation.yaml` | 10 | 0.1 | 1 | 1 | |
| 3 | Emphase Contrastive | `high_contrastive.yaml` | 1 | 1 | 1 | 1 | |
| 4 | Emphase Cycle-Complet | `high_cycles.yaml` | 1 | 0.1 | 1 | 10 | |
| 5 | Emphase Attributs Continus | `high_alpha.yaml` | 1 | 0.1 | 1 | 1 | `--attr-alpha 5` |

Hyperparamètres communs : GW de dimension 12, encodeurs/décodeurs MLP (256 unités
cachées), AdamW (weight decay 1e-6), OneCycleLR (`max_lr` 1.5e-4), batch 2056,
300 000 pas, initialisation Kaiming, gradient clipping 1.0.

```bash
# Un run « vide sémantique » (couleur retirée des attributs)
python SSD_eval/SSD_Train.py --experiment high_cycles --seed 0
```

```bash
# Le Modèle Complet de contrôle (couleur présente dans les attributs)
python SSD_eval/SSD_Train.py --experiment high_cycles --seed 0 --with-colors
```

```bash
# Les 5 configurations x 3 graines + 5 contrôles
bash scripts/train_all.sh
```

Les checkpoints sont écrits dans `checkpoints/runs/<experiment>_seed<k>/checkpoints/`
(`last.ckpt` + meilleur `epoch=*.ckpt` selon `val/loss`).

## 5. Évaluation

Le script principal applique les critères du mémoire (§2.3) à chaque configuration
listée dans [`experiments/thesis_runs.yaml`](experiments/thesis_runs.yaml) :

1. **Extraction des couleurs** reconstruites (seuillage d'Otsu, moyenne autour du
   barycentre) pour les chemins *traduction* (attributs → image), *demi-cycle*
   (image → GW → image) et *cycle complet* (image → GW → attributs → GW → image) ;
2. **Régularité** : tests KS entre catégories d'un même attribut (Hue + R, G, B,
   15 bins, p < 0.05) ;
3. **Arbitraire** : tests KS entre instances (graines) pour une même catégorie.

Les Modèles Complets (contrôles) sont évalués séparément (régularité seule) par
`scripts/evaluate_all.sh`.

```bash
bash scripts/evaluate_all.sh
```

```bash
python SSD_eval/SSD_eval_arbitrary.py --experiment high_cycles
```

```bash
python SSD_eval/SSD_eval_regularity.py --checkpoint checkpoints/runs/high_cycles_seed0/checkpoints/last.ckpt
```

Les résultats (distributions, p-values, figures) sont écrits dans `results/`.
Les notebooks de [`notebooks/`](notebooks/) servent à l'inspection visuelle des
reconstructions (critère d'Additionnalité, évalué qualitativement dans le mémoire).

### Utiliser les poids du stage plutôt que ré-entraîner

Si les poids d'origine sont publiés (voir §8), décompressez-les dans `checkpoints/runs/`
sous leurs noms d'origine (colonne `original_run_name` du manifeste), puis :

```bash
bash scripts/evaluate_all.sh --use-original-names
```

## 6. Correspondance mémoire ↔ code

| Mémoire | Code |
|---|---|
| §2.1.1 Jeu d'entraînement (SSD) | `scripts/download_data.sh` |
| §2.1.2 Jeu de test structuré | `SSD_eval/SSD_generate_evalset.py` |
| §2.2.2 Vide sémantique (couleur retirée des attributs) | `custom_collate_factory` + `attr_legacy_no_color` dans `SSD_eval/SSD_Train.py` |
| §2.2.3 Paramétrage technique | `config/*.yaml`, `SSD_eval/SSD_Train.py` |
| §2.3.1 Extraction des couleurs | `SSD_eval/SSD_utils.py`, `HueShapeAnalyzer` (`SSD_eval/SSD_H_evaluation_functions.py`) |
| §2.3.2 Critère de Régularité | `SSD_eval/SSD_eval_regularity.py`, `HueShapeAnalyzer.analyze_dataset` |
| §2.3.2 Critère d'Arbitraire | `SSD_eval/SSD_eval_arbitrary.py` |
| §2.3.3 Les 5 configurations × 3 instances | `config/`, `experiments/thesis_runs.yaml`, `scripts/train_all.sh` |
| Figures du chapitre 3 | `SSD_eval/SSD_visualize_functions.py`, `archive/scripts/Figures_*.py` |

## 7. Organisation du dépôt

```
config/                  main.yaml + 1 fichier par configuration de pondération
experiments/
  thesis_runs.yaml       runs utilisés dans le mémoire (config, graine, nom d'origine)
  all_training_runs.csv  hyperparamètres de tous les runs du stage (extraits des logs W&B)
SSD_eval/
  paths.py               chemins centralisés (surchargeables par variables d'environnement)
  SSD_Train.py           entraînement d'un GW
  SSD_generate_evalset.py
  SSD_eval_regularity.py critère de Régularité (1 modèle)
  SSD_eval_arbitrary.py  évaluation principale : Régularité + Arbitraire (N graines)
  SSD_H_evaluation_functions.py, SSD_utils.py, SSD_visualize_functions.py
scripts/                 setup_env, download_data, train_all, evaluate_all, reproduce_all
notebooks/               exploration / inspection visuelle
archive/                 scripts exploratoires non nécessaires à la réplication
docs/                    mémoire (PDF)
```

## 8. Écarts connus et points d'attention

- **Poids entraînés.** Les fichiers `.ckpt` présents dans l'historique git (dossier
  `checkpoints/` des commits précédents) sont **tronqués** (archives zip incomplètes,
  toutes coupées à 2,5–2,75 Mo) et ne peuvent pas être chargés. Ils ont été retirés de
  la branche. Les poids complets doivent être republiés (ex. GitHub Release ou Zenodo)
  pour permettre une évaluation sans ré-entraînement.
- **Fork `shimmer-ssd`.** Voir §2 : sans cette version modifiée, l'entraînement et
  l'évaluation ne peuvent pas s'exécuter.
- **`shimmer` possiblement modifié aussi.** Le code d'origine importait
  `FixedSharedSelection` depuis `shimmer.modules`, qui n'existe dans aucune version
  publique de `shimmer` (import inutilisé, retiré). L'environnement du stage utilisait
  donc probablement un `shimmer` modifié lui aussi (la correction « LogSoftmax → exp →
  log » peut s'y trouver). Le cas échéant, le publier et le désigner avec `SHIMMER_FORK`.
- **Taux d'apprentissage.** Le mémoire (§2.2.3) indique un OneCycleLR de 2.6e-5 à 2e-6 ;
  le code effectivement exécuté utilise `OneCycleLR(max_lr=1.5e-4, div_factor=0.38,
  final_div_factor=5, pct_start=0.03)`. C'est le code qui fait foi.
- **« 300 000 epochs »** dans le mémoire correspond à 300 000 *pas* d'optimisation
  (`max_steps`).
- **Configuration 3.** Le mémoire indique β = 1 ; les runs utilisés portaient le nom
  « High_contrastive (10) ». `config/high_contrastive.yaml` suit le mémoire (β = 1).
- **Graines de la configuration 5.** Deux des trois instances d'origine ont été lancées
  avec `seed=1` (voir `experiments/all_training_runs.csv`). La réplication utilise
  les graines 0, 1, 2.
- **Positions du jeu de test.** Le mémoire décrit 16 positions (4 × 4) ; le générateur
  utilise x, y ∈ {8, 13, 18, 24} px, cohérent avec la configuration de binning
  (`binning_config_6144`, plage 7–25 px).
- **Palette du jeu de test.** L'analyseur reconstruisait les couleurs d'entrée avec une
  palette fixe de 100 teintes, alors que le jeu de test du mémoire en compte 25 : pour
  les Modèles Complets, la couleur injectée dans le vecteur d'attributs ne correspondait
  pas à celle de l'image. La palette est désormais déduite du CSV (sans effet sur les
  modèles « vide sémantique », dont les attributs n'ont pas de couleur).
- **Déterminisme.** Les graines fixent l'initialisation et l'échantillonnage des données,
  mais les opérations CUDA ne sont pas strictement déterministes : on attend une
  réplication des *conclusions* (régularité / arbitraire), pas des p-values identiques
  au bit près.

## Références

- Devillers, B., Maytié, L., VanRullen, R. (2024). *Semi-supervised multimodal
  representation learning through a global workspace.*
- VanRullen, R., Kanai, R. (2021). *Deep learning and the Global Workspace Theory.*
- Bibliothèques : [shimmer](https://github.com/ruflab/shimmer),
  [shimmer-ssd](https://github.com/ruflab/shimmer-ssd),
  [simple-shapes-dataset](https://github.com/ruflab/simple-shapes-dataset).
