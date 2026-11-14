# GWT-Synesthesia

Le code de mon mémoire de M2 : **« Synesthésie artificielle dans un modèle inspiré de
la Global Workspace Theory »** ([le PDF](docs/EMANUELLI_Memoire_M2.pdf)).

L'idée : j'entraîne un Global Workspace qui relie des images de formes simples
([Simple Shapes Dataset](https://github.com/ruflab/simple-shapes-dataset)) à leur
description (forme, taille, position, orientation)… sauf que la description ne
contient pas la couleur. Est-ce que le modèle finit quand même par associer des
couleurs aux formes, et chacun à sa façon ? Un peu comme un synesthète.

Les modèles entraînés sont sur Zenodo : [doi:10.5281/zenodo.24208192](https://doi.org/10.5281/zenodo.24208192).

> 🚧 Je n'ai pas encore tout testé de bout en bout de mon côté (l'évaluation des modèles
> publiés et la comparaison avec les tableaux du mémoire). Si quelque chose coince,
> ouvre une issue !

## Installer

Il te faut Python 3.11.

```bash
git clone -b reproducibility https://github.com/AEmanuelli/GWT-Synesthesia.git
```

```bash
cd GWT-Synesthesia
```

```bash
bash scripts/setup_env.sh
```

Et à chaque nouveau terminal :

```bash
source .venv/bin/activate
```

Pas de `python3.11` sous la main ? Avec conda : `conda create -y -n py311 python=3.11`,
puis `PYTHON="$(conda run -n py311 which python)" bash scripts/setup_env.sh`.

## Retrouver les résultats du mémoire (sans rien entraîner)

C'est le plus simple. Ça marche sur CPU, c'est juste plus rapide avec un GPU.

D'abord les données : le VAE pré-entraîné et le jeu de test (153 600 images, compte 1 à 2 h) :

```bash
bash scripts/download_data.sh --eval-only
```

Ensuite les 20 modèles du mémoire (5 configurations × 3 graines + 5 contrôles) :

```bash
bash scripts/download_checkpoints.sh
```

On lance l'évaluation (tout atterrit dans `results/`) :

```bash
bash scripts/evaluate_all.sh --use-original-names
```

Et on sort les tableaux 3.1 et B.2 :

```bash
python experiments/make_tables.py results/arbitrary
```

Envie de vérifier que tout tourne avant d'y passer deux heures ? Il y a un mini-jeu
de test : `bash scripts/download_data.sh --eval-only --quick-evalset`, puis
`python SSD_eval/SSD_eval_arbitrary.py --experiment high_cycles --use-original-names --dataset-csv data/evaluation_set_quick/attributes.csv --output-dir results/quick_test`.

### L'annexe E (le petit bruit sur la forme)

Dans l'annexe E, je rajoute un tout petit bruit (0,01) sur l'encodage de la forme
pour voir ce qui se cache dedans. Il n'est pas activé dans l'évaluation principale.

```bash
bash scripts/evaluate_annex_E.sh --use-original-names
```

```bash
python experiments/make_tables.py results/arbitrary_shape_noise_0.01
```

## Tout réentraîner

Là il faut un vrai GPU : 20 entraînements de 300 000 pas.

```bash
bash scripts/download_data.sh
```

```bash
bash scripts/train_all.sh
```

```bash
bash scripts/evaluate_all.sh
```

Les 5 configurations sont dans [`config/`](config/). Ce qui change, c'est le poids
de chaque loss : α traduction, β contrastive, γ demi-cycle, δ cycle complet.

| # | Configuration | Fichier | α | β | γ | δ |
|---|---|---|---|---|---|---|
| 1 | Référence | `base_params.yaml` | 1 | 0,1 | 1 | 1 |
| 2 | Emphase Traduction | `high_translation.yaml` | 10 | 0,1 | 1 | 1 |
| 3 | Emphase Contrastive | `high_contrastive.yaml` | 1 | 1 | 1 | 1 |
| 4 | Emphase Cycle-Complet | `high_cycles.yaml` | 1 | 0,1 | 1 | 10 |
| 5 | Emphase Attributs Continus | `high_alpha.yaml` | 1 | 0,1 | 1 | 1 |

La 5 donne en plus un poids ×5 à la taille, la position et l'orientation
(`train_all.sh` s'en occupe).

Pour un seul modèle : `python SSD_eval/SSD_Train.py --experiment high_cycles --seed 0`
(ajoute `--with-colors` pour le modèle de contrôle, celui qui voit la couleur).

## Où est quoi

| Dans le mémoire | Dans le code |
|---|---|
| Le jeu de test (§2.1.2) | `SSD_eval/SSD_generate_evalset.py` |
| L'entraînement (§2.2) | `SSD_eval/SSD_Train.py`, `config/` |
| Extraction des couleurs et régularité (§2.3) | `SSD_eval/SSD_H_evaluation_functions.py`, `SSD_eval/SSD_utils.py` |
| L'arbitraire (§2.3) | `SSD_eval/SSD_eval_arbitrary.py` |
| Les tableaux 3.1, B.2, E.1 | `experiments/make_tables.py` |
| Quel modèle publié correspond à quel run | `experiments/thesis_runs.yaml` |

## Ce qui diffère un peu du mémoire

En fouillant les checkpoints et les logs, j'ai trouvé quelques écarts :

- **Tous les modèles n'ont pas tourné aussi longtemps.** Normalement c'est 300 000 pas
  (le mémoire dit « epochs », c'est bien des pas). Certains se sont arrêtés avant, et
  deux contrôles ont été lancés pour 150 000 pas :

  | Configuration | Modèle | Pas faits |
  |---|---|---|
  | 1. Référence | graine 1 | 206 550 / 300 000 (69 %) |
  | 1. Référence | contrôle | 211 896 / 300 000 (71 %) |
  | 2. Emphase Traduction | graine 1 | 216 027 / 300 000 (72 %) |
  | 2. Emphase Traduction | graine 2 | 230 607 / 300 000 (77 %) |
  | 2. Emphase Traduction | graine 0 | 283 824 / 300 000 (95 %) |
  | 3. Emphase Contrastive | contrôle | 147 258 / 150 000 |
  | 4. Emphase Cycle-Complet | contrôle | 149 931 / 150 000 |

  Tous les autres sont allés à 98 % ou plus.
- **Le taux d'apprentissage.** Le mémoire annonce 2,6e-5 → 2e-6. En vrai, c'est
  `OneCycleLR(max_lr=1.5e-4, div_factor=0.38, final_div_factor=5)`, qui part de
  3,95e-4 et finit vers 7,9e-5.
- **L'architecture.** Les MLP ont 3 couches cachées de 256, pas 4.
- **Les graines de la configuration 5.** Deux des trois modèles ont la même graine (1).
  Si tu réentraînes, ce sera 0, 1 et 2.
- **Le tableau B.1.** Pour la configuration 1 (sans couleur), c'est la moyenne des
  graines 1 et 2 seulement.

## À savoir aussi

- `setup_env.sh` installe exactement les versions de mai 2025, y compris mes forks de
  [`shimmer`](https://github.com/AEmanuelli/shimmer),
  [`shimmer-ssd`](https://github.com/AEmanuelli/shimmer-ssd) et
  [`simple-shapes-dataset`](https://github.com/AEmanuelli/simple-shapes-dataset).
- Tout se range dans le dépôt (`data/`, `checkpoints/`, `results/`). Si tu as déjà le
  jeu de test quelque part : `export GWT_EVALSET_DIR=/chemin/vers/evaluation_set`
  (les autres réglages sont dans [`.env.example`](.env.example)).
- Le GPU n'est pas parfaitement déterministe : en réentraînant, tu retrouveras les
  mêmes conclusions, pas forcément les mêmes p-values à la virgule près.

## Citer

> Emanuelli, A. (2025). *Synesthésie artificielle dans un modèle inspiré de la Global
> Workspace Theory* [Mémoire de M2]. https://github.com/AEmanuelli/GWT-Synesthesia —
> modèles : https://doi.org/10.5281/zenodo.24208192.

## Références

- Devillers, B., Maytié, L., VanRullen, R. (2024). *Semi-supervised multimodal
  representation learning through a global workspace.*
- VanRullen, R., Kanai, R. (2021). *Deep learning and the Global Workspace Theory.*
