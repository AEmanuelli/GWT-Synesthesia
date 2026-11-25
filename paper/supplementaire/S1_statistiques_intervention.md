> Supplément S1 de l'article (`paper/article.md`), § « Une infime masse résiduelle colore la traduction ». Dans ces tableaux, « annexe E » désigne la traduction bruitée (vecteur de bruit fixe, graine 42, amplitude 0,01). Fichier généré par `python experiments/sweep_statistics.py` (copie de `results/saturation/balayage/statistiques.md`, données du balayage `experiments/shape_noise_sweep.py`).

# Balayage du bruit sur la forme : statistiques

Généré par `experiments/sweep_statistics.py` à partir de `balayage/donnees/`. Unité statistique : l'exemple (48 par instance), avec une valeur par exemple (médiane sur les 7 directions admissibles, 240° à 300°). Intervalles : bootstrap des exemples (5 000 tirages, 95 %) ou Wilson pour une proportion. Corrections de Holm sur l'ensemble des tests de chaque tableau. Les instances d'une même configuration ne sont que 3 : la généralisation à la configuration repose sur ce nombre.

Masse déplacée : une perturbation de norme ε dans une direction admissible retire de 0,71 ε (240°, 300°) à 0,82 ε (270°) à la forme cible. ε = 0,001 correspond donc à 0,07–0,08 % de masse, ε = 0,01 à 0,7–0,8 %.

## A. La masse résiduelle colore-t-elle la traduction ?

S par exemple (médiane sur les directions admissibles). « colorés » : exemples avec S ≥ 0,3. p : Wilcoxon apparié unilatéral (S(ε) > S(0)), corrigé (Holm, 57 tests). ρ : Spearman entre ε et S par exemple (10 niveaux, 0 compris) ; « + / − » : exemples à ρ positif / négatif, test des signes corrigé.

| Modèle | S(0) | S(ε=0,001) [IC 95 %] | colorés | p | S(ε=0,01) [IC 95 %] | colorés | p | ρ médian (+ / −) | p (signes) | ε* |
|---|---|---|---|---|---|---|---|---|---|---|
| Référence — graine 0 | 0,05 | 0,07 [0,05–0,09] | 0/48 | < 10⁻⁴ | 0,10 [0,06–0,30] | 17/48 | < 10⁻⁴ | 0,99 (45 / 3) | < 10⁻⁴ | – |
| Référence — graine 1 | 0,05 | 0,23 [0,19–0,27] | 14/48 | < 10⁻⁴ | 0,71 [0,66–0,76] | 48/48 | < 10⁻⁴ | 1,00 (48 / 0) | < 10⁻⁴ | 0,00178 |
| Référence — graine 2 | 0,06 | 0,45 [0,42–0,49] | 46/48 | < 10⁻⁴ | 0,81 [0,78–0,84] | 48/48 | < 10⁻⁴ | 0,99 (48 / 0) | < 10⁻⁴ | 0,001 |
| Référence — contrôle | 0,81 | 0,81 [0,78–0,82] | 48/48 | 0,304 | 0,80 [0,78–0,82] | 48/48 | 0,349 | 0,98 (28 / 20) | 1,000 | 0,001 |
| Emphase Traduction — graine 0 | 0,04 | 0,04 [0,03–0,06] | 0/48 | 0,042 | 0,04 [0,04–0,06] | 0/48 | < 10⁻⁴ | 1,00 (44 / 4) | < 10⁻⁴ | – |
| Emphase Traduction — graine 1 | 0,04 | 0,04 [0,03–0,05] | 0/48 | 1,000 | 0,04 [0,03–0,05] | 0/48 | 1,000 | 0,44 (35 / 13) | 0,022 | – |
| Emphase Traduction — graine 2 | 0,05 | 0,05 [0,04–0,05] | 0/48 | 0,922 | 0,06 [0,05–0,07] | 0/48 | 0,0032 | 0,93 (42 / 6) | < 10⁻⁴ | – |
| Emphase Contrastive — graine 0 | 0,05 | 0,05 [0,04–0,07] | 0/48 | 1,000 | 0,05 [0,04–0,07] | 0/48 | 1,000 | -0,45 (20 / 28) | 1,000 | – |
| Emphase Contrastive — graine 1 | 0,04 | 0,04 [0,03–0,07] | 1/48 | 1,000 | 0,05 [0,04–0,06] | 3/48 | 1,000 | 0,56 (33 / 15) | 0,113 | – |
| Emphase Contrastive — graine 2 | 0,06 | 0,06 [0,05–0,07] | 0/48 | 1,000 | 0,05 [0,04–0,07] | 0/48 | 1,000 | 0,61 (30 / 18) | 0,780 | – |
| Emphase Contrastive — contrôle | 0,80 | 0,80 [0,77–0,82] | 48/48 | < 10⁻⁴ | 0,80 [0,77–0,82] | 48/48 | < 10⁻⁴ | 1,00 (34 / 14) | 0,052 | 0,001 |
| Emphase Cycle-Complet — graine 0 | 0,03 | 0,24 [0,17–0,40] | 22/48 | < 10⁻⁴ | 0,81 [0,76–0,88] | 48/48 | < 10⁻⁴ | 0,96 (48 / 0) | < 10⁻⁴ | 0,00316 |
| Emphase Cycle-Complet — graine 1 | 0,03 | 0,04 [0,03–0,06] | 0/48 | 0,0024 | 0,85 [0,79–0,90] | 46/48 | < 10⁻⁴ | 0,93 (48 / 0) | < 10⁻⁴ | 0,00562 |
| Emphase Cycle-Complet — graine 2 | 0,03 | 0,04 [0,04–0,06] | 0/48 | 0,014 | 0,59 [0,46–0,68] | 39/48 | < 10⁻⁴ | 0,97 (48 / 0) | < 10⁻⁴ | 0,00562 |
| Emphase Cycle-Complet — contrôle | 0,83 | 0,83 [0,79–0,85] | 48/48 | 0,013 | 0,83 [0,81–0,85] | 48/48 | 0,786 | 0,19 (25 / 23) | 1,000 | 0,001 |
| Emphase Attributs Continus — graine 0 | 0,05 | 0,09 [0,07–0,13] | 0/48 | 0,0002 | 0,50 [0,37–0,60] | 34/48 | < 10⁻⁴ | 0,91 (48 / 0) | < 10⁻⁴ | 0,01 |
| Emphase Attributs Continus — graine 1 | 0,05 | 0,38 [0,31–0,41] | 34/48 | < 10⁻⁴ | 0,75 [0,71–0,78] | 48/48 | < 10⁻⁴ | 1,00 (47 / 1) | < 10⁻⁴ | 0,001 |
| Emphase Attributs Continus — graine 2 | 0,05 | 0,30 [0,27–0,33] | 24/48 | < 10⁻⁴ | 0,93 [0,89–0,95] | 48/48 | < 10⁻⁴ | 0,99 (48 / 0) | < 10⁻⁴ | 0,00178 |
| Emphase Attributs Continus — contrôle | 0,81 | 0,81 [0,79–0,82] | 48/48 | < 10⁻⁴ | 0,81 [0,79–0,82] | 48/48 | 0,0027 | 1,00 (34 / 14) | 0,052 | 0,001 |

ε* : plus petite amplitude où la médiane des exemples atteint S ≥ 0,3.

## B. Spécificité : la même amplitude sur les attributs continus

ΔC : déplacement de la couleur dans le plan de chrominance (unités RGB), médiane par exemple ; forme : directions admissibles ; continus : 36 directions aléatoires de même norme ε. Rapport : médiane des rapports par exemple [IC 95 %]. p : Wilcoxon apparié unilatéral, corrigé (Holm, 38 tests).

| Modèle | ΔC forme / continus (ε=0,001) | rapport [IC] | p | ΔC forme / continus (ε=0,01) | rapport [IC] | p |
|---|---|---|---|---|---|---|
| Référence — graine 0 | 1,9 / 0,56 | 2,2 [1,5–12,0] | 0,0002 | 10,7 / 5,70 | 1,3 [0,5–11,9] | 0,031 |
| Référence — graine 1 | 54,5 / 0,46 | 118,1 [92,8–134,4] | < 10⁻⁴ | 125,7 / 4,70 | 26,7 [24,4–28,6] | < 10⁻⁴ |
| Référence — graine 2 | 85,5 / 0,56 | 165,6 [134,3–176,4] | < 10⁻⁴ | 156,6 / 5,71 | 27,1 [23,8–30,0] | < 10⁻⁴ |
| Référence — contrôle | 0,0 / 0,04 | 0,8 [0,6–1,0] | 1,000 | 0,3 / 0,43 | 0,8 [0,6–1,1] | 1,000 |
| Emphase Traduction — graine 0 | 0,2 / 0,42 | 0,6 [0,5–0,7] | 1,000 | 2,5 / 4,39 | 0,6 [0,5–0,7] | 1,000 |
| Emphase Traduction — graine 1 | 0,3 / 0,39 | 0,7 [0,6–0,9] | 1,000 | 2,7 / 3,73 | 0,7 [0,5–0,8] | 1,000 |
| Emphase Traduction — graine 2 | 0,8 / 0,36 | 2,1 [1,6–2,4] | < 10⁻⁴ | 7,0 / 3,70 | 1,9 [1,6–2,2] | < 10⁻⁴ |
| Emphase Contrastive — graine 0 | 0,1 / 0,14 | 0,4 [0,3–0,6] | 1,000 | 0,7 / 1,42 | 0,5 [0,3–0,6] | 1,000 |
| Emphase Contrastive — graine 1 | 0,2 / 0,14 | 1,1 [0,8–1,8] | 0,227 | 1,7 / 1,46 | 1,1 [1,0–1,7] | 0,385 |
| Emphase Contrastive — graine 2 | 0,2 / 0,17 | 0,9 [0,8–1,2] | 1,000 | 1,6 / 1,48 | 1,0 [0,7–1,3] | 1,000 |
| Emphase Contrastive — contrôle | 0,0 / 0,05 | 0,7 [0,5–0,8] | 1,000 | 0,3 / 0,49 | 0,6 [0,5–0,8] | 1,000 |
| Emphase Cycle-Complet — graine 0 | 15,9 / 0,12 | 132,8 [118,4–152,2] | < 10⁻⁴ | 96,7 / 1,22 | 67,5 [62,9–79,9] | < 10⁻⁴ |
| Emphase Cycle-Complet — graine 1 | 3,2 / 0,03 | 84,2 [64,0–118,4] | < 10⁻⁴ | 94,5 / 0,33 | 271,2 [213,7–365,2] | < 10⁻⁴ |
| Emphase Cycle-Complet — graine 2 | 5,7 / 0,06 | 85,4 [67,3–102,5] | < 10⁻⁴ | 88,0 / 0,65 | 118,2 [69,5–166,0] | < 10⁻⁴ |
| Emphase Cycle-Complet — contrôle | 4,4 / 0,09 | 55,8 [39,7–65,4] | < 10⁻⁴ | 32,3 / 0,82 | 36,5 [26,8–58,4] | < 10⁻⁴ |
| Emphase Attributs Continus — graine 0 | 6,6 / 0,64 | 10,0 [7,8–16,2] | < 10⁻⁴ | 67,7 / 6,77 | 10,7 [6,8–12,4] | < 10⁻⁴ |
| Emphase Attributs Continus — graine 1 | 76,9 / 0,78 | 107,8 [82,8–119,5] | < 10⁻⁴ | 143,0 / 7,47 | 17,7 [16,1–20,4] | < 10⁻⁴ |
| Emphase Attributs Continus — graine 2 | 65,9 / 0,62 | 95,1 [87,7–112,9] | < 10⁻⁴ | 128,7 / 6,24 | 20,6 [18,2–27,9] | < 10⁻⁴ |
| Emphase Attributs Continus — contrôle | 0,0 / 0,04 | 0,5 [0,4–0,6] | 1,000 | 0,2 / 0,42 | 0,5 [0,4–0,6] | 1,000 |

## C. La plage balayée est-elle réaliste ?

Masse résiduelle 1 − p(forme vraie) produite par le décodeur d'attributs sur 19 200 vraies images (image → VAE → GW → attributs ; données de `residual_regression.py`). « dans la plage » : part des images dont la masse résiduelle est entre 0,07 % et 0,8 % (ε de 0,001 à 0,01).

| Modèle | 5 % | 25 % | médiane | 75 % | 95 % | dans la plage | ≥ 0,07 % |
|---|---|---|---|---|---|---|---|
| Référence — graine 0 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 1,1e-05 | 1,9e-03 | 15 % | 15 % |
| Référence — graine 1 | 3,3e-06 | 2,3e-05 | 1,6e-04 | 7,3e-04 | 2,4e-03 | 25 % | 26 % |
| Référence — graine 2 | 9,5e-07 | 8,0e-06 | 1,0e-04 | 7,9e-04 | 3,2e-03 | 26 % | 27 % |
| Référence — contrôle | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 3,6e-06 | 0 % | 0 % |
| Emphase Traduction — graine 0 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0 % | 0 % |
| Emphase Traduction — graine 1 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 1,2e-07 | 0 % | 0 % |
| Emphase Traduction — graine 2 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 1,2e-07 | 0 % | 0 % |
| Emphase Contrastive — graine 0 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 1,2e-07 | 0 % | 0 % |
| Emphase Contrastive — graine 1 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 4,5e-06 | 1 % | 1 % |
| Emphase Contrastive — graine 2 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 3,6e-06 | 1 % | 1 % |
| Emphase Contrastive — contrôle | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 2,0e-06 | 0 % | 1 % |
| Emphase Cycle-Complet — graine 0 | 6,3e-04 | 1,9e-03 | 4,4e-03 | 6,6e-03 | 1,1e-02 | 79 % | 93 % |
| Emphase Cycle-Complet — graine 1 | 2,6e-03 | 4,8e-03 | 7,1e-03 | 1,0e-02 | 1,6e-02 | 59 % | 100 % |
| Emphase Cycle-Complet — graine 2 | 3,9e-03 | 6,7e-03 | 9,5e-03 | 1,3e-02 | 1,8e-02 | 37 % | 100 % |
| Emphase Cycle-Complet — contrôle | 1,2e-03 | 2,5e-03 | 4,1e-03 | 6,5e-03 | 1,1e-02 | 84 % | 99 % |
| Emphase Attributs Continus — graine 0 | 1,2e-07 | 6,1e-06 | 1,2e-04 | 1,1e-03 | 4,6e-03 | 28 % | 31 % |
| Emphase Attributs Continus — graine 1 | 1,1e-06 | 6,6e-06 | 8,0e-05 | 7,4e-04 | 3,7e-03 | 25 % | 26 % |
| Emphase Attributs Continus — graine 2 | 0,0e+00 | 2,6e-06 | 1,7e-04 | 7,2e-04 | 2,3e-03 | 25 % | 25 % |
| Emphase Attributs Continus — contrôle | 0,0e+00 | 0,0e+00 | 0,0e+00 | 0,0e+00 | 7,2e-07 | 0 % | 0 % |

## D. La teinte dépend-elle de la répartition du résidu ? (ε = 0,01)

Exemples colorés aux deux extrêmes (S ≥ 0,3 à 240° et à 300° : tout le résidu sur l'une ou l'autre forme non cible). |Δh| : écart de teinte entre ces deux extrêmes. Plancher de mesure : 99e centile de |Δh| chez les 4 contrôles, qui ont une couleur dans leurs attributs (14,9° ; maximum 16,7°). p : test binomial (part des écarts au-dessus du plancher > 1 %), corrigé (Holm, 19 tests). Monotonie : part des exemples dont la teinte (dépliée) varie dans un seul sens de 240° à 300°.

| Modèle | n | |Δh| médian [IC 95 %] | au-dessus du plancher [IC] | p | pas médian | plus grand pas (médiane) | monotonie |
|---|---|---|---|---|---|---|---|
| Référence — graine 0 | 0 | – | – | – | – | – | – |
| Référence — graine 1 | 11 | 17,9° [9,6–32,4] | 6/11 [28–79 %] | < 10⁻⁴ | 2,3° | 7,1° | 82 % |
| Référence — graine 2 | 2 | 28,2° [24,3–32,2] | 2/2 [34–100 %] | 0,0013 | 4,8° | 9,4° | 100 % |
| Référence — contrôle | 48 | 0,1° [0,0–0,1] | 0/48 [0–7 %] | 1,000 | 0,0° | 0,0° | 75 % |
| Emphase Traduction — graine 0 | 0 | – | – | – | – | – | – |
| Emphase Traduction — graine 1 | 0 | – | – | – | – | – | – |
| Emphase Traduction — graine 2 | 0 | – | – | – | – | – | – |
| Emphase Contrastive — graine 0 | 0 | – | – | – | – | – | – |
| Emphase Contrastive — graine 1 | 1 | 0,4° [0,4–0,4] | 0/1 [0–79 %] | 1,000 | 0,1° | 0,1° | 100 % |
| Emphase Contrastive — graine 2 | 0 | – | – | – | – | – | – |
| Emphase Contrastive — contrôle | 48 | 0,1° [0,0–0,1] | 0/48 [0–7 %] | 1,000 | 0,0° | 0,0° | 69 % |
| Emphase Cycle-Complet — graine 0 | 43 | 155,6° [138,6–164,8] | 43/43 [92–100 %] | < 10⁻⁴ | 14,5° | 73,6° | 70 % |
| Emphase Cycle-Complet — graine 1 | 16 | 77,8° [64,6–114,6] | 16/16 [81–100 %] | < 10⁻⁴ | 13,7° | 37,6° | 25 % |
| Emphase Cycle-Complet — graine 2 | 22 | 65,3° [52,4–84,5] | 22/22 [85–100 %] | < 10⁻⁴ | 8,5° | 25,8° | 77 % |
| Emphase Cycle-Complet — contrôle | 48 | 4,2° [2,8–6,2] | 2/48 [1–14 %] | 0,918 | 0,8° | 2,2° | 21 % |
| Emphase Attributs Continus — graine 0 | 8 | 139,9° [63,1–150,1] | 8/8 [68–100 %] | < 10⁻⁴ | 12,6° | 54,2° | 75 % |
| Emphase Attributs Continus — graine 1 | 16 | 16,2° [7,5–23,1] | 8/16 [28–72 %] | < 10⁻⁴ | 0,5° | 10,1° | 31 % |
| Emphase Attributs Continus — graine 2 | 3 | 13,5° [5,9–25,2] | 1/3 [6–79 %] | 0,356 | 1,2° | 17,1° | 0 % |
| Emphase Attributs Continus — contrôle | 48 | 0,0° [0,0–0,0] | 0/48 [0–7 %] | 1,000 | 0,0° | 0,0° | 65 % |

Sensibilité au seuil : mêmes mesures avec S ≥ 0,15 aux deux extrêmes (même plancher).

| Modèle | n | |Δh| médian [IC 95 %] | au-dessus du plancher | p |
|---|---|---|---|---|
| Référence — graine 0 | 1 | 23,0° [23,0–23,0] | 1/1 | 0,110 |
| Référence — graine 1 | 21 | 24,0° [10,0–33,4] | 13/21 | < 10⁻⁴ |
| Référence — graine 2 | 18 | 27,0° [19,8–41,1] | 14/18 | < 10⁻⁴ |
| Référence — contrôle | 48 | 0,1° [0,0–0,1] | 0/48 | 1,000 |
| Emphase Traduction — graine 0 | 0 | – | – | – |
| Emphase Traduction — graine 1 | 1 | 0,0° [0,0–0,0] | 0/1 | 1,000 |
| Emphase Traduction — graine 2 | 1 | 24,7° [24,7–24,7] | 1/1 | 0,110 |
| Emphase Contrastive — graine 0 | 3 | 0,0° [0,0–0,2] | 0/3 | 1,000 |
| Emphase Contrastive — graine 1 | 4 | 4,9° [0,3–12,9] | 0/4 | 1,000 |
| Emphase Contrastive — graine 2 | 2 | 1,6° [0,6–2,5] | 0/2 | 1,000 |
| Emphase Contrastive — contrôle | 48 | 0,1° [0,0–0,1] | 0/48 | 1,000 |
| Emphase Cycle-Complet — graine 0 | 47 | 151,3° [135,7–163,7] | 47/47 | < 10⁻⁴ |
| Emphase Cycle-Complet — graine 1 | 24 | 67,8° [58,6–90,6] | 24/24 | < 10⁻⁴ |
| Emphase Cycle-Complet — graine 2 | 36 | 68,3° [59,7–102,8] | 34/36 | < 10⁻⁴ |
| Emphase Cycle-Complet — contrôle | 48 | 4,2° [2,8–6,2] | 2/48 | 0,751 |
| Emphase Attributs Continus — graine 0 | 19 | 126,6° [92,2–138,3] | 19/19 | < 10⁻⁴ |
| Emphase Attributs Continus — graine 1 | 16 | 16,2° [7,5–23,1] | 8/16 | < 10⁻⁴ |
| Emphase Attributs Continus — graine 2 | 7 | 13,5° [5,9–29,6] | 3/7 | 0,0004 |
| Emphase Attributs Continus — contrôle | 48 | 0,0° [0,0–0,0] | 0/48 | 1,000 |
