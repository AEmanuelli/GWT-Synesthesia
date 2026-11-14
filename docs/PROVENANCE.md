# Provenance des modèles et vérifications

Ce document complète le [README](../README.md). Il explique d'où viennent les
20 modèles publiés sur Zenodo et comment leur correspondance avec le mémoire a été
vérifiée. Il n'est pas nécessaire pour reproduire les résultats.

## Les 20 modèles du mémoire

[`experiments/thesis_runs.yaml`](../experiments/thesis_runs.yaml) associe chaque
configuration et chaque graine au run d'origine du stage (nom de dossier et checkpoint),
ainsi qu'à une empreinte du checkpoint (`expect`) : epoch, pas, coefficients des
losses, présence de la couleur dans les attributs.

- Les noms de runs ne reflètent pas toujours les coefficients. Par exemple, les runs
  « High_contrastive (10) » ont β = 1, comme indiqué dans le mémoire. Ce sont les
  coefficients enregistrés dans les checkpoints qui font foi.
- Les contrôles des configurations 3 et 4 suivent un planning de 150 000 pas. Pour
  celui de la configuration 3, le checkpoint retenu est `epoch=605.ckpt` (le dossier
  contient aussi une relance arrêtée à l'epoch 46).
- Quelques runs se sont arrêtés avant 300 000 pas (par exemple la configuration 1,
  graine 1 : 206 550 pas).
- Pour la configuration 5, deux des trois instances d'origine utilisent la graine 1 ;
  un réentraînement utilise les graines 0, 1 et 2.
- [`experiments/all_training_runs.csv`](../experiments/all_training_runs.csv) liste
  les hyperparamètres de tous les runs du stage, relevés dans les logs W&B.

## Vérifications avec les logs W&B

**Tableau B.1.** `python experiments/check_table_B1.py export_wandb.csv` recalcule les
losses de validation à partir d'un export CSV des runs W&B. Elles correspondent au
tableau au millième près pour quatre contrôles sur cinq (le cinquième à ±0,002), et à
1–2 % près pour les moyennes « vide sémantique » (l'export donne la dernière valeur
enregistrée). Pour la configuration 1, les valeurs du tableau sont celles de la
moyenne des graines 1 et 2.

**Versions des bibliothèques.** Les forks sont figés sur leur état de mai 2025 :
`shimmer` `fdb8e2f`, `shimmer-ssd` `21f8271`, `simple-shapes-dataset` `34d45cd`.
Leurs commits du 24/06/2025 ajoutent des pondérations de loss (cycle, rotation). Sur
les 1 235 points de validation de « (Seed 0) High_cycles (10) v5 », la loss de cycle
est la moyenne simple des deux cycles et la loss des attributs vaut
`loss_attr + loss_cat`, à la précision machine près : ces pondérations n'étaient pas
actives pendant les entraînements.

**Taux d'apprentissage.** L'état du scheduler enregistré dans chaque checkpoint
correspond à `OneCycleLR(max_lr=1.5e-4, div_factor=0.38, final_div_factor=5,
pct_start=0.03)`.

**Architecture.** Les encodeurs et décodeurs ont 4 couches linéaires
(12 → 256 → 256 → 256 → 12), soit 3 couches cachées de 256 unités.

## Jeu de test

Les 16 positions sont x, y ∈ {8, 13, 18, 24} pixels ; les 25 teintes sont espacées de
14,4° (HLS, S ≈ 0,90, L ≈ 0,70). La palette est lue dans le CSV du jeu de test.

## Préparer une nouvelle archive de checkpoints

Sur la machine contenant les checkpoints, les retrouver par leur contenu, quels que
soient leurs noms ou emplacements ; le croisement avec W&B (optionnel) vérifie aussi
graine, alpha et meilleure loss de validation :

```bash
python scripts/identify_checkpoints.py ~/Bureau --wandb ENTITE/PROJET
```

```bash
python scripts/package_checkpoints.py --located located_checkpoints.yaml
```

L'archive produite contient un fichier `SHA256SUMS`. Une fois publiée, renseigner son
record Zenodo dans `scripts/download_checkpoints.sh` (`ZENODO_RECORD`).

Les fichiers `.ckpt` présents dans les anciens commits du dépôt sont incomplets ; les
modèles de référence sont ceux de l'archive Zenodo.
