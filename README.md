# GWT-Synesthesia — Synesthésie artificielle dans un modèle Global Workspace

Code du mémoire de M2 d'Alexis Emanuelli,
**« Définition et émergence d'une synesthésie artificielle dans un modèle d'apprentissage
profond inspiré de la théorie du Global Workspace »** ([PDF](docs/EMANUELLI_Memoire_M2.pdf)).
Modèles entraînés : [doi:10.5281/zenodo.24208192](https://doi.org/10.5281/zenodo.24208192).

Un Global Workspace bimodal (image ↔ vecteur d'attributs) est entraîné sur le
[Simple Shapes Dataset](https://github.com/ruflab/simple-shapes-dataset) sans la couleur
dans les attributs (« vide sémantique »). On teste ensuite si le modèle associe malgré
tout des couleurs aux attributs, de façon **régulière** et **propre à chaque instance**.

> **État du dépôt.** La procédure a été exécutée de bout en bout sur une machine vierge
> (installation, téléchargements, génération du jeu de test, évaluation des modèles
> publiés, tableaux, démarrage d'un entraînement). Ce qui est reproduit à l'identique et
> ce qui ne l'est pas est détaillé en [§5](#5-ce-qui-est-reproduit-et-les-écarts-avec-le-mémoire).

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

Si des fichiers du projet sont déjà sur la machine (modèles, jeu de test…),
`bash scripts/setup_local.sh` installe l'environnement puis les retrouve et les range
au bon endroit, ce qui évite de les télécharger ou de les régénérer.

## 2. Reproduire les résultats avec les modèles publiés

Recommandé : pas d'entraînement. Un GPU n'est pas nécessaire, mais sur CPU il faut compter
environ 20 minutes par modèle (4 cœurs) et 8 Go de mémoire vive.

Télécharger le VAE pré-entraîné et générer le jeu de test du mémoire (153 600 images, 1 à 2 h) :

```bash
bash scripts/download_data.sh --eval-only
```

Télécharger les 20 modèles du mémoire (5 configurations × 3 graines + 5 contrôles) :

```bash
bash scripts/download_checkpoints.sh
```

Lancer l'évaluation (résultats dans `results/`, 20 modèles) :

```bash
bash scripts/evaluate_all.sh --use-original-names
```

Produire les tableaux 3.1 et B.2, et le bilan des modèles de contrôle :

```bash
python experiments/make_tables.py results/arbitrary
```

Pour vérifier l'installation en quelques minutes avant de lancer le tout, utiliser un mini-jeu de test :
`bash scripts/download_data.sh --eval-only --quick-evalset`, puis
`python SSD_eval/SSD_eval_arbitrary.py --experiment high_cycles --use-original-names --dataset-csv data/evaluation_set_quick/attributes.csv --output-dir results/quick_test`.

L'évaluation reprend là où elle s'est arrêtée : un modèle déjà évalué
(`analysis_results.pkl` présent) n'est pas recalculé.

### Annexe E (bruit sur les attributs de forme)

L'annexe E refait le test sur le chemin de traduction après avoir perturbé le vecteur de
forme. Le même vecteur de bruit est ajouté au one-hot de la forme de toutes les images et
de toutes les instances : trois valeurs gaussiennes (graine 42) multipliées par 0,01, soit
(+0,0034 ; +0,0013 ; +0,0023) ; le vecteur est ensuite ramené à une somme de 1. Les
chemins demi-cycle et cycle complet ne sont pas modifiés. Ce bruit est désactivé dans
l'évaluation principale.

```bash
bash scripts/evaluate_annex_E.sh --use-original-names
```

```bash
python experiments/make_tables.py results/arbitrary_shape_noise_0.01
```

### Saturation et luminosité (prolongement de l'annexe E)

Compte rendu des résultats, avec les figures principales : [`docs/compte_rendu_saturation.md`](docs/compte_rendu_saturation.md).

Options ajoutées pour tester si la saturation est portée par les composantes résiduelles des
attributs ; désactivées par défaut, elles ne changent pas les résultats du chapitre 3.

- `--hls` (`SSD_eval_arbitrary.py`, `SSD_eval_regularity.py`) : extrait aussi la saturation S et la luminosité L (HLS, même masque et même barycentre que la teinte, échelle 0-255 comme R, G, B), les compare entre catégories et écrit dans `results/saturation/`.
- `bash scripts/evaluate_saturation.sh --use-original-names` : évaluation `--hls` des 15 modèles et des contrôles, avec le bruit de l'annexe E puis sans bruit.
- `--translation-only` (mêmes scripts) : ne calcule que le chemin de traduction (un passage dans le décodeur du VAE par image au lieu de quatre) ; résultats dans un dossier `traduction_seule/`, jamais dans ceux du chapitre 3.
- `--low-memory` (mêmes scripts) : une fois les couleurs extraites, garde les images reconstruites en uint8 (elles ne servent plus qu'aux planches d'exemples) et pas les masques ; mêmes résultats, environ 4 Go de mémoire vive par modèle au lieu de 9.
- `python experiments/make_tables.py results/saturation/arbitrary --channel S` : tableaux B.2 et 3.1 sur S (ou L) au lieu de la teinte ; `--test raw` remplace le test KS sur histogrammes par le test KS sur valeurs brutes.
- `python experiments/saturation_analysis.py` : niveaux de S et L par chemin, tests entre catégories et entre instances (test du mémoire, KS brut, δ de Cliff), figures ; `--from-rgb` donne un aperçu à partir des résultats du chapitre 3.
- `python experiments/saturation_analysis.py --figures-only` : refait seulement les figures depuis le cache (`cache_distributions.pkl`), sans refaire les tests.
- `python experiments/saturation_reference.py` : témoin de mesure, S et L extraites des images d'entrée et du VAE seul (sans GW).
- `python experiments/shape_noise_sweep.py` : balayage du bruit sur le one-hot de forme (36 directions du plan tangent au simplexe × 9 amplitudes de 0,001 à 0,1), forme re-décodée depuis l'image, témoins (attributs continus, Modèles Complets).
- `python experiments/sweep_statistics.py` : statistiques du balayage (intervalles de confiance, tests appariés corrigés de Holm, témoin de spécificité, masse résiduelle sur de vraies images, dépendance de la teinte à la direction) ; à lancer après le balayage et `residual_regression.py`. Résultat : `balayage/statistiques.md` (annexe A du compte rendu).
- `python experiments/synesthesia_figure.py` : associations forme → couleur les plus nettes de chaque modèle (traduction avec le bruit de l'annexe E), redessinées avec de vraies traductions (`galerie.png`, `palettes.png`) ; à lancer après l'évaluation `--translation-only`.
- `python experiments/synesthesia_figure.py --noise-robustness <csv>` : marque ◆ dans la galerie les associations stables selon la graine et l'amplitude du bruit (par défaut `results/saturation/robustesse/bruit_conditions.csv`, produit par `saturation_robustness.py` ; ignoré s'il manque).
- `python experiments/residual_regression.py` : lecture des résidus par régression. La couleur d'une image d'entrée est-elle lisible dans les attributs que le GW en décode (image → VAE → GW → attributs) ? 19 200 images stratifiées (teinte × forme × taille) ; ridge, k plus proches voisins et petit perceptron multicouche sur le résidu des probabilités de forme (log-rapports des formes non cibles à la forme cible), sur les écarts des attributs continus à leurs vraies valeurs, ou sur les deux, avec ou sans les vraies valeurs des attributs ; R² en validation croisée, ligne de base à couleurs mélangées (`results/saturation/regression/`).
- `python experiments/saturation_robustness.py` puis `multiple_testing.py`, `regression_sensitivity.py` (après `residual_regression.py --seed 1 --regressions ridge kNN --out results/saturation/robustesse/regression/graine1_par_case32`, etc.) et `robustness_statistics.py` : robustesse (étape 4) — extraction par le pixel le plus intérieur du masque, 6 graines × 3 amplitudes du bruit de l'annexe E, corrections de Holm et Benjamini-Hochberg, sensibilité de la régression au tirage et au nombre d'images ; `results/saturation/robustesse/` (annexe B du compte rendu).

Durée sur un GeForce 940MX : environ 35 min par modèle pour l'évaluation complète (le GPU est
le facteur limitant : quatre passages dans le VAE par image), environ 10 min avec `--translation-only`.
`residual_regression.py` demande environ 1 h 10 pour les 19 modèles, dont 50 min pour le perceptron
(`--device cpu` sur une partie des configurations permet de le répartir entre GPU et CPU).
Sur disque dur, une copie du jeu de test en mémoire évite en plus des lectures lentes
(`cp -r data/evaluation_set /dev/shm/`, puis `export GWT_EVALSET_DIR=/dev/shm/evaluation_set`).

### Tableau B.1 (losses de validation)

Nécessite le dataset d'entraînement (`bash scripts/download_data.sh`, 7 Go) :

```bash
python experiments/compute_table_B1.py --use-original-names
```

### Figures

Pour un modèle évalué, dans `results/arbitrary/<configuration>/` (ou
`results/arbitrary_shape_noise_0.01/` pour l'annexe E) :

| Figure du mémoire | Fichier |
|---|---|
| 3.1, E.1a (teinte d'une catégorie, deux instances) | `cross_model_comparison_<configuration>_N3/translated/shape/<forme>/seed0_vs_seed2_overlay.png` |
| 3.2, E.1d (exemples de reconstructions) | `results_<configuration>_sans_couleurs_seed<i>/diverse_examples_plate.png` |
| C.2a-b, E.1c (p-valeurs entre catégories) | `results_…_seed<i>/<attribut>/translated/translated_<attribut>_heatmap_KS_pvalue.png` (canaux R, G, B : `…_heatmap_KS_stat_R.png`) |
| C.2c-e, E.1b (distributions de deux catégories) | `results_…_seed<i>/<attribut>/translated/<attribut>_<cat1>_vs_<cat2>/H_comparison.png`, et `R_`, `G_`, `B_` (paires régulières seulement) |

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
Pour vérifier que l'entraînement démarre, sans attendre (possible sans GPU) :
`python SSD_eval/SSD_Train.py --experiment base_params --seed 0 --max-steps 20 --run-name essai`.

## 4. Où trouver quoi

| Mémoire | Code |
|---|---|
| Jeu de test (§2.1.2) | `SSD_eval/SSD_generate_evalset.py` |
| Entraînement (§2.2) | `SSD_eval/SSD_Train.py`, `config/` |
| Extraction des couleurs, régularité (§2.3) | `SSD_eval/SSD_H_evaluation_functions.py`, `SSD_eval/SSD_utils.py` |
| Arbitraire (§2.3) | `SSD_eval/SSD_eval_arbitrary.py` |
| Bruit de l'annexe E | `_process_translation_path_batch` dans `SSD_eval/SSD_utils.py` (option `--shape-noise`) |
| Tableaux 3.1, B.2, E.1 | `experiments/make_tables.py` |
| Tableau B.1 | `experiments/compute_table_B1.py` |
| Sortie « probabilités » du décodeur d'attributs | `SSD_eval/category_probs.py` |
| Modèles du mémoire (configuration, graine, run d'origine) | `experiments/thesis_runs.yaml` |
| Saturation, luminosité, balayage du bruit sur la forme, figure de synesthésie, lecture des résidus par régression | `experiments/saturation_analysis.py`, `experiments/shape_noise_sweep.py`, `experiments/sweep_statistics.py`, `experiments/synesthesia_figure.py`, `experiments/residual_regression.py` |

## 5. Ce qui est reproduit, et les écarts avec le mémoire

Vérification faite avec les modèles publiés et le jeu de test régénéré. Les chemins
Traduction et Demi-Cycle ont été évalués sur deux machines : résultats identiques.

### Reproduit à l'identique

- **Tableau B.2** (régularité) : en entier, les trois chemins des cinq configurations.
- **Modèles Complets** : aucune paire régulière sur aucun des trois chemins, pour les
  quatre contrôles disponibles.
- **Tableau 3.1** (arbitraire) : toutes les cases sauf deux (Forme en Référence et en
  Emphase Attributs Continus, voir plus bas), dont la ligne Emphase Cycle-Complet
  (2/2, 0/3, 0/2, 14/16) et le 8/16 d'Emphase Traduction en Cycle-Complet.
- **Tableau E.1**, Traduction Bruitée : Forme 3/3 et Orientation 16/16 dans les trois
  configurations concernées, numérateurs nuls ailleurs, aucune régularité pour Emphase
  Traduction et Emphase Contrastive.
- **Figures 3.1, E.1a, E.1b** : histogrammes de teinte identiques classe par classe.
- **Tableau B.1** : les cinq losses des contrôles sont identiques au millième pour quatre
  configurations ; pour les modèles « vide sémantique », l'écart maximal est de 0,015.
  Les 18 métriques de validation enregistrées pendant l'entraînement de la graine 0 de la
  configuration Référence sont retrouvées à cinq décimales.

### Écarts dans les résultats

- **Forme, tableau 3.1.** Le mémoire donne 2/2 (Référence) et 1/2 (Emphase Attributs
  Continus). Le script d'origine donne 0/3 et 0/2 : la première instance n'y présente
  aucune paire significative pour la Forme, donc l'attribut n'est pas comparé. Sans ce
  filtre on obtient 2/3 et 1/2, avec les formes citées dans le texte du mémoire.
- **Taille, Pos X et Pos Y, tableau E.1.** Les numérateurs sont retrouvés, pas les
  dénominateurs (Référence : Taille 0/3 et Pos X 0/2 au lieu de 0/2 et 0/4 ; Emphase
  Cycle-Complet : Taille 0/2 au lieu de « - » ; Emphase Attributs Continus : Pos X et
  Pos Y 0/4 au lieu de 0/2). Cause non identifiée.
- **Lecture des zéros.** Dans les tableaux 3.1 et E.1, un score nul pour Taille, Pos X ou
  Pos Y signifie que l'attribut n'a pas été comparé entre instances (filtre sur la
  première instance), et non que ses couleurs sont identiques d'une instance à l'autre :
  testées directement, toutes ces catégories diffèrent entre au moins deux instances
  (`make_tables.py --no-first-instance-filter`).
- **Légendes.** Les figures 3.1 et E.1b (« configuration de référence », « Réf. ») sont
  obtenues avec les modèles Emphase Cycle-Complet, pas avec la configuration Référence.
- **Contrôle de la configuration Emphase Traduction.** Le fichier publié sur Zenodo n'est
  pas le modèle du tableau B.1. Le dossier d'origine contient deux entraînements : celui
  du mémoire (150 000 pas, `epoch=599.ckpt`) et un autre (`last.ckpt`, 300 000 pas) lancé
  avec un coefficient de traduction entier, que `shimmer` ignore sans avertir. Ce second
  modèle, celui de l'archive, n'a donc jamais été entraîné à traduire (loss de traduction
  de 10,44 en validation). `experiments/thesis_runs.yaml` désigne désormais
  `epoch=599.ckpt` ; tant que l'archive Zenodo n'est pas mise à jour, ce contrôle est
  signalé absent et ignoré par l'évaluation.
- **Jeu de test.** Il n'a pas été publié : il est régénéré d'après le §2.1.2, avec les
  positions 8, 13, 18 et 24 px sur chaque axe. Les histogrammes des figures 3.1 et E.1
  étant retrouvés à l'identique, il correspond à celui du mémoire.

### Écarts entre le texte du mémoire et le code

- **Décodeur d'attributs.** Les modèles ont été entraînés avec une version locale de
  `shimmer` (runs « corrected v5 (Logsoftmax --> exp --> log) ») dont le décodeur
  GW → attributs renvoie des probabilités de forme. Cette modification n'est pas sur le
  fork public. `SSD_eval/category_probs.py` la rétablit, à l'évaluation comme à
  l'entraînement ; sans elle, le chemin Cycle-Complet est inutilisable (erreur de
  reconstruction 30 à 900 fois plus grande selon le modèle).
- **Loss totale.** Le mémoire (§2.2.3) divise la somme pondérée par S = α + β + γ + δ.
  Le code en prend la moyenne sur les quatre termes : (αL_tr + βL_cont + γL_dcy + δL_cy) / 4.
  C'est cette valeur qui figure dans le tableau B.1.
- **Test de Kolmogorov-Smirnov.** Il n'est pas appliqué aux valeurs brutes. Chaque
  distribution est d'abord réduite à un histogramme de 15 classes sur l'étendue commune
  des deux échantillons, puis le test compare ces deux histogrammes, ramenés à un effectif
  qui ne dépend que de l'étendue des valeurs (une quarantaine de points pour une teinte
  couvrant tout le cercle) et non du nombre d'images. Les p-valeurs en dépendent directement.
- **Durée d'entraînement.** Les modèles n'ont pas tous été entraînés aussi longtemps.
  Prévu : 300 000 pas (le mémoire écrit « epochs »). Certains entraînements se sont
  arrêtés plus tôt, et trois contrôles ont été lancés sur un planning de 150 000 pas :

  | Configuration | Modèle | Pas effectués |
  |---|---|---|
  | 1. Référence | graine 1 | 206 550 / 300 000 (69 %) |
  | 1. Référence | contrôle | 211 896 / 300 000 (71 %) |
  | 2. Emphase Traduction | graine 1 | 216 027 / 300 000 (72 %) |
  | 2. Emphase Traduction | graine 2 | 230 607 / 300 000 (77 %) |
  | 2. Emphase Traduction | graine 0 | 283 824 / 300 000 (95 %) |
  | 2. Emphase Traduction | contrôle | 145 800 / 150 000 |
  | 3. Emphase Contrastive | contrôle | 147 258 / 150 000 |
  | 4. Emphase Cycle-Complet | contrôle | 149 931 / 150 000 |

  Les autres modèles sont allés au moins jusqu'à 98 % des 300 000 pas.
- **Taux d'apprentissage.** Le mémoire (§2.2.3) indique 2,6e-5 → 2e-6. Le planning
  utilisé est `OneCycleLR(max_lr=1.5e-4, div_factor=0.38, final_div_factor=5)` : le taux
  part de 3,95e-4, descend à 1,5e-4 en 9 000 pas et finit vers 7,9e-5.
- **Architecture.** Le mémoire annonce quatre couches cachées. Les encodeurs et le
  décodeur visuel en ont trois de 256 unités (entrée 12, ou 8 pour les attributs sans
  couleur et 11 avec ; sortie 12). Le décodeur d'attributs en a deux, suivies d'une sortie
  pour la forme (3 valeurs) et d'une sortie `tanh` pour les attributs continus.
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
- **Reproductibilité.** Un réentraînement ne redonne pas les mêmes poids : l'ordre des
  lots n'est pas fixé par la graine et certaines opérations GPU ne sont pas déterministes.
  Il doit reproduire les conclusions, pas les p-valeurs. L'évaluation des modèles publiés,
  elle, est déterministe.
- **Notebooks.** `notebooks/` charge les modèles sans `category_probs.py` : la traduction
  attributs → image y est correcte, pas un passage par les attributs décodés.

## Citer

> Emanuelli, A. (2025). *Définition et émergence d'une synesthésie artificielle dans un
> modèle d'apprentissage profond inspiré de la théorie du Global Workspace* [Mémoire de M2].
> https://github.com/AEmanuelli/GWT-Synesthesia — modèles : https://doi.org/10.5281/zenodo.24208192.

## Références

- Devillers, B., Maytié, L., VanRullen, R. (2024). *Semi-supervised multimodal
  representation learning through a global workspace.*
- VanRullen, R., Kanai, R. (2021). *Deep learning and the Global Workspace Theory.*
