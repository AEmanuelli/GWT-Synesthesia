# GWT-Synesthesia — Synesthésie artificielle dans un modèle Global Workspace

Code du mémoire de M2 d'Alexis Emanuelli,
**« Synesthésie artificielle dans un modèle inspiré de la Global Workspace Theory »**
([PDF](docs/EMANUELLI_Memoire_M2.pdf)).
Modèles entraînés : [doi:10.5281/zenodo.24208192](https://doi.org/10.5281/zenodo.24208192).

Un Global Workspace bimodal (image ↔ vecteur d'attributs) est entraîné sur le
[Simple Shapes Dataset](https://github.com/ruflab/simple-shapes-dataset) sans la couleur
dans les attributs (« vide sémantique »). On teste ensuite si le modèle associe malgré
tout des couleurs aux attributs, de façon **régulière** et **propre à chaque instance**.

> **État du dépôt.** La procédure de reproduction est en cours de validation : l'auteur
> doit encore vérifier de bout en bout l'évaluation des modèles publiés et la
> correspondance des tableaux obtenus avec ceux du mémoire. Les retours sont bienvenus.

## 1. Installation

Python 3.11 requis.

```bash
git clone -b reproducibility https://github.com/AEmanuelli/GWT-Synesthesia.git
```

```bash
cd GWT-Synesthesia
```

```bash
bash scripts/setup_env.sh
```

Puis, dans chaque nouveau terminal :

```bash
source .venv/bin/activate
```

Sans `python3.11` installé : `conda create -y -n py311 python=3.11`, puis
`PYTHON="$(conda run -n py311 which python)" bash scripts/setup_env.sh`.

## 2. Reproduire les résultats avec les modèles publiés

Recommandé : pas d'entraînement, CPU possible (GPU plus rapide).

Télécharger le VAE pré-entraîné et générer le jeu de test du mémoire (153 600 images, 1 à 2 h) :

```bash
bash scripts/download_data.sh --eval-only
```

Télécharger les 20 modèles du mémoire (5 configurations × 3 graines + 5 contrôles) :

```bash
bash scripts/download_checkpoints.sh
```

Lancer l'évaluation (résultats dans `results/`) :

```bash
bash scripts/evaluate_all.sh --use-original-names
```

Produire les tableaux 3.1 et B.2 :

```bash
python experiments/make_tables.py results/arbitrary
```

Pour vérifier l'installation en quelques minutes avant de lancer le tout, utiliser un mini-jeu de test :
`bash scripts/download_data.sh --eval-only --quick-evalset`, puis
`python SSD_eval/SSD_eval_arbitrary.py --experiment high_cycles --use-original-names --dataset-csv data/evaluation_set_quick/attributes.csv --output-dir results/quick_test`.

### Annexe E (bruit sur les attributs de forme)

L'annexe E refait le test sur le chemin de traduction en ajoutant un faible bruit
(0,01) aux attributs de forme. Ce bruit est désactivé dans l'évaluation principale.

```bash
bash scripts/evaluate_annex_E.sh --use-original-names
```

```bash
python experiments/make_tables.py results/arbitrary_shape_noise_0.01
```

## 3. Réentraîner les modèles

GPU nécessaire : 20 entraînements de 300 000 pas.

```bash
bash scripts/download_data.sh
```

```bash
bash scripts/train_all.sh
```

```bash
bash scripts/evaluate_all.sh
```

Les cinq configurations du mémoire (§2.3.3) sont dans [`config/`](config/). Coefficients des losses : α traduction, β contrastive, γ demi-cycle, δ cycle complet.

| # | Configuration | Fichier | α | β | γ | δ |
|---|---|---|---|---|---|---|
| 1 | Référence | `base_params.yaml` | 1 | 0,1 | 1 | 1 |
| 2 | Emphase Traduction | `high_translation.yaml` | 10 | 0,1 | 1 | 1 |
| 3 | Emphase Contrastive | `high_contrastive.yaml` | 1 | 1 | 1 | 1 |
| 4 | Emphase Cycle-Complet | `high_cycles.yaml` | 1 | 0,1 | 1 | 10 |
| 5 | Emphase Attributs Continus | `high_alpha.yaml` | 1 | 0,1 | 1 | 1 |

La configuration 5 donne en plus un poids ×5 aux attributs continus (taille, position, orientation) ; `train_all.sh` le gère.

Pour entraîner un seul modèle : `python SSD_eval/SSD_Train.py --experiment high_cycles --seed 0`.
Ajouter `--with-colors` pour le modèle de contrôle.

## 4. Où trouver quoi

| Mémoire | Code |
|---|---|
| Jeu de test (§2.1.2) | `SSD_eval/SSD_generate_evalset.py` |
| Entraînement (§2.2) | `SSD_eval/SSD_Train.py`, `config/` |
| Extraction des couleurs, régularité (§2.3) | `SSD_eval/SSD_H_evaluation_functions.py`, `SSD_eval/SSD_utils.py` |
| Arbitraire (§2.3) | `SSD_eval/SSD_eval_arbitrary.py` |
| Tableaux 3.1, B.2, E.1 | `experiments/make_tables.py` |
| Modèles du mémoire (configuration, graine, run d'origine) | `experiments/thesis_runs.yaml` |

## 5. Différences avec le mémoire

Relevées dans les checkpoints et les logs d'entraînement des modèles publiés :

- **Durée d'entraînement.** Les modèles n'ont pas tous été entraînés aussi longtemps.
  Prévu : 300 000 pas (le mémoire écrit « epochs »). Certains entraînements se sont
  arrêtés plus tôt, et deux contrôles ont été lancés sur un planning de 150 000 pas :

  | Configuration | Modèle | Pas effectués |
  |---|---|---|
  | 1. Référence | graine 1 | 206 550 / 300 000 (69 %) |
  | 1. Référence | contrôle | 211 896 / 300 000 (71 %) |
  | 2. Emphase Traduction | graine 1 | 216 027 / 300 000 (72 %) |
  | 2. Emphase Traduction | graine 2 | 230 607 / 300 000 (77 %) |
  | 2. Emphase Traduction | graine 0 | 283 824 / 300 000 (95 %) |
  | 3. Emphase Contrastive | contrôle | 147 258 / 150 000 |
  | 4. Emphase Cycle-Complet | contrôle | 149 931 / 150 000 |

  Les autres modèles sont allés au moins jusqu'à 98 % des 300 000 pas.
- **Taux d'apprentissage.** Le mémoire (§2.2.3) indique 2,6e-5 → 2e-6. Le planning
  utilisé est `OneCycleLR(max_lr=1.5e-4, div_factor=0.38, final_div_factor=5)` : le taux
  part de 3,95e-4 et finit vers 7,9e-5.
- **Architecture.** Les encodeurs et décodeurs ont 3 couches cachées de 256 unités
  (12 → 256 → 256 → 256 → 12), et non 4.
- **Graines de la configuration 5.** Deux des trois instances ont été entraînées avec la
  même graine (1). Un réentraînement utilise les graines 0, 1 et 2.
- **Tableau B.1.** Pour la configuration 1 (vide sémantique), les valeurs correspondent à
  la moyenne des graines 1 et 2 seulement.

## 6. Bon à savoir

- **Versions.** `setup_env.sh` installe les versions exactes utilisées en mai 2025, dont
  les forks de l'auteur de [`shimmer`](https://github.com/AEmanuelli/shimmer),
  [`shimmer-ssd`](https://github.com/AEmanuelli/shimmer-ssd) et
  [`simple-shapes-dataset`](https://github.com/AEmanuelli/simple-shapes-dataset).
- **Correspondance avec les runs d'origine.** [`experiments/thesis_runs.yaml`](experiments/thesis_runs.yaml)
  indique, pour chaque modèle publié, sa configuration, sa graine et son nom de run d'origine.
- **Chemins.** Tout est rangé dans le dépôt (`data/`, `checkpoints/`, `results/`). Un jeu
  de test existant peut être réutilisé avec `export GWT_EVALSET_DIR=/chemin/vers/evaluation_set`
  (autres variables : [`.env.example`](.env.example)).
- **Reproductibilité.** Certaines opérations GPU ne sont pas déterministes : un
  réentraînement reproduit les conclusions, pas les p-values au bit près.

## Citer

> Emanuelli, A. (2025). *Synesthésie artificielle dans un modèle inspiré de la Global
> Workspace Theory* [Mémoire de M2]. https://github.com/AEmanuelli/GWT-Synesthesia —
> modèles : https://doi.org/10.5281/zenodo.24208192.

## Références

- Devillers, B., Maytié, L., VanRullen, R. (2024). *Semi-supervised multimodal
  representation learning through a global workspace.*
- VanRullen, R., Kanai, R. (2021). *Deep learning and the Global Workspace Theory.*
