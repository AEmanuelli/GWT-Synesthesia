# GWT-Synesthesia — Synesthésie artificielle dans un modèle Global Workspace

Code du mémoire de M2 d'Alexis Emanuelli :
**« Synesthésie artificielle dans un modèle inspiré de la Global Workspace Theory »**
([PDF](docs/EMANUELLI_Memoire_M2.pdf)).
Poids des modèles entraînés : Zenodo, [doi:10.5281/zenodo.24208192](https://doi.org/10.5281/zenodo.24208192).

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
pip install --no-deps "git+https://github.com/AEmanuelli/shimmer-ssd.git@21f8271f163ec72950e147ebe1e192d467ef7f2a"
```

> **Bibliothèques modifiées.** Ce projet utilise les forks de l'auteur des trois
> bibliothèques du laboratoire, figés sur leur état pendant les entraînements (mai 2025) :
>
> | Fork | Commit | Modifications par rapport à ruflab |
> |---|---|---|
> | [`AEmanuelli/shimmer-ssd`](https://github.com/AEmanuelli/shimmer-ssd) | `21f8271` (11/04/2025) | variante de domaine `attr_legacy_no_color`, module attributs paramétré par `alpha`/`temperature`, décodeur attributs dédié, option `exclude_colors` du callback d'images |
> | [`AEmanuelli/shimmer`](https://github.com/AEmanuelli/shimmer) | `fdb8e2f` (28/02/2025) | décodeur attributs `GWDecoder_legacy` (`gw_module.py`) |
> | [`AEmanuelli/simple-shapes-dataset`](https://github.com/AEmanuelli/simple-shapes-dataset) | `34d45cd` (27/03/2025) | option `use_color_attributes` du data module ; option de téléchargement d'une version allégée |
>
> `setup_env.sh` installe les trois et vérifie que la bonne version de `shimmer-ssd` est
> présente.

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

Hyperparamètres communs (mémoire §2.2.3, vérifiés dans les checkpoints d'origine) :
GW de dimension 12, encodeurs/décodeurs MLP (256 unités cachées), AdamW (weight decay
1e-6), batch 2056, 300 000 pas, initialisation Kaiming, gradient clipping 1.0.
Taux d'apprentissage : voir §8.

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

Les 20 Global Workspaces évalués dans le mémoire (5 configurations × 3 graines + 5
contrôles) sont publiés sur [Zenodo (record 24208192)](https://zenodo.org/records/24208192).
[`scripts/download_checkpoints.sh`](scripts/download_checkpoints.sh) retrouve l'archive
via l'API Zenodo, vérifie les sommes SHA-256 et l'installe dans `checkpoints/runs/` :

```bash
bash scripts/download_checkpoints.sh
```

```bash
bash scripts/evaluate_all.sh --use-original-names
```

<details><summary>Pour l'auteur : retrouver et publier les checkpoints</summary>

Sur la machine contenant les checkpoints d'origine, retrouver les 20 fichiers par leur
contenu (dossier d'origine, epoch, pas, coefficients, couleur), quels que soient leurs
noms ou emplacements actuels. Le croisement W&B (optionnel) vérifie graine, alpha,
couleur et meilleure val/loss :

```bash
python scripts/identify_checkpoints.py ~/Desktop /mnt/HD2 --wandb ENTITY/PROJECT
```

Puis empaqueter les fichiers identifiés :

```bash
python scripts/package_checkpoints.py --located located_checkpoints.yaml --out gwt_synesthesia_checkpoints.tar.gz
```

Le script vérifie chaque fichier (archive torch complète), suit les liens `last.ckpt`
et ajoute un fichier `SHA256SUMS`. Publier l'archive sur Zenodo (DOI citable) ou en
GitHub Release, puis renseigner l'URL dans `scripts/download_checkpoints.sh`.
</details>

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
  `checkpoints/` des commits précédents) sont **tronqués** (archives zip incomplètes)
  et ne peuvent pas être chargés. Ils ont été retirés ; utiliser l'archive publiée
  (voir §5).
- **Commits « Final » des forks (24/06/2025) non utilisés.** Postérieurs aux
  entraînements, ils changent la loss : `shimmer` pondère ×3 la 2ᵉ loss de cycle et
  `shimmer-ssd` pondère ×2 la rotation dans la loss des attributs. Les logs W&B montrent
  qu'aucune des deux n'était active : sur les 1 235 points de validation de
  « (Seed 0) High_cycles (10) v5 », `val/cycles` est la moyenne simple des deux cycles
  (écart max 3·10⁻⁸) et la loss de cycle des attributs vaut `loss_attr + loss_cat`
  (écart max 7·10⁻⁹). Les forks sont donc figés sur les commits précédents. De plus,
  les deux commits « Final » sont incompatibles entre eux (`downlad_file` renommé).
- **Taux d'apprentissage (erratum du mémoire).** Le mémoire (§2.2.3) indique un
  OneCycleLR de 2.6e-5 à 2e-6. Les 20 checkpoints évalués contiennent l'état de leur
  scheduler : `base_lrs = 3.95e-4`, `max_lr = 1.5e-4`, dernier LR ≈ 7.9e-5 en fin de
  run, soit `OneCycleLR(max_lr=1.5e-4, div_factor=0.38, final_div_factor=5,
  pct_start=0.03)`, ce que fait le code. Pour reproduire les modèles du mémoire, on
  garde donc ces valeurs.
- **Architecture des MLP.** Les encodeurs ont 4 couches linéaires
  (12→256→256→256→12), soit 3 couches cachées de 256 (le mémoire dit « quatre couches
  cachées »).
- **« 300 000 epochs »** dans le mémoire correspond à 300 000 *pas* d'optimisation
  (`max_steps`).
- **Noms de runs trompeurs.** Les coefficients enregistrés dans les checkpoints font foi :
  les runs « High_contrastive (10) » ont bien β = 1 (conforme au mémoire) ; à l'inverse,
  « (Seed 0) (bis) High_translations (10) » a δ = 10 et « High translation (10) -
  corrected » a α = 1. Le manifeste `experiments/thesis_runs.yaml` stocke pour chaque
  run l'empreinte attendue (`expect`), vérifiable avec `scripts/identify_checkpoints.py`.
- **Contrôles (Modèles Complets).** Les contrôles des configurations 3 et 4 ont été
  entraînés sur un planning de 150 000 pas (et non 300 000). Le dossier
  « (Colored) High_contrastive (1) v5 » contient deux lancements : le checkpoint du
  mémoire est `epoch=605.ckpt` (le `last.ckpt` est une relance avortée à l'epoch 46).
  Plusieurs runs « vide sémantique » se sont arrêtés avant 300 000 pas
  (ex. base seed 1 : 206 550 pas).
- **Vérification par le tableau B.1.** `python experiments/check_table_B1.py
  export_wandb.csv` recalcule, à partir d'un export CSV de W&B, les losses de validation
  moyennes du tableau B.1 pour les runs du manifeste. Quatre contrôles sur cinq
  correspondent au millième près sur les 5 losses (le 5ᵉ, H_T, à ±0,002) ; les moyennes
  « vide sémantique » sont à 1–2 % près (l'export W&B donne la dernière valeur loggée,
  le mémoire a pu en lire une autre). Pour la configuration 1, le tableau correspond
  exactement à la moyenne des seules graines 1 et 2 : il a vraisemblablement été calculé
  avant l'ajout de la 3ᵉ instance.
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

## Citer

> Emanuelli, A. (2025). *Synesthésie artificielle dans un modèle inspiré de la Global
> Workspace Theory* [Mémoire de M2]. Code : https://github.com/AEmanuelli/GWT-Synesthesia.
> Poids des modèles : Zenodo, https://doi.org/10.5281/zenodo.24208192.

## Références

- Devillers, B., Maytié, L., VanRullen, R. (2024). *Semi-supervised multimodal
  representation learning through a global workspace.*
- VanRullen, R., Kanai, R. (2021). *Deep learning and the Global Workspace Theory.*
- Bibliothèques : [shimmer](https://github.com/ruflab/shimmer),
  [shimmer-ssd](https://github.com/ruflab/shimmer-ssd),
  [simple-shapes-dataset](https://github.com/ruflab/simple-shapes-dataset).
