> Supplément S2 de l'article (`paper/article.md`). Dans ces tableaux, « mémoire » désigne les réglages et le test des travaux antérieurs [1] (KS sur histogrammes à 15 classes ; bruit de graine 42, amplitude 0,01) et « annexe E » la traduction bruitée. Fichier généré par `python experiments/robustness_statistics.py` (copie de `results/saturation/robustesse/annexe_B.md`), complété en B.1 par les tableaux de `results/saturation/robustesse/extraction_bruit/synthese.md` (`experiments/saturation_robustness.py`). Données : `saturation_robustness.py` (volets 1 et 2), `multiple_testing.py` (volet 3), `residual_regression.py --regressions ridge kNN` puis `regression_sensitivity.py` (volet 4).

# Robustesse des résultats sur la saturation : statistiques

Généré par `experiments/robustness_statistics.py`. 19 modèles. Intervalles : bootstrap (2000 tirages, 95 %) ou Wilson ; corrections de Holm par tableau. Hors de portée sur ce GPU : autres instances ou réentraînements (3 instances par configuration seulement).

## B.1 Autre extraction : pixel le plus intérieur du masque

Unité : l'image traduite (n = 4 608 par modèle et condition, appariées entre extractions). ΔS = S(intérieur) − S(barycentre), médiane [IC 95 %, bootstrap] ; p : Wilcoxon apparié bilatéral, Holm sur les 38 tests. δ(forme) : |δ de Cliff| maximal entre deux formes (S), pour chaque extraction ; « contrôles » : effet de mesure de référence.

| Modèle | condition | S barycentre | S intérieur | ΔS [IC 95 %] | p (Holm) | δ(forme) barycentre | δ(forme) intérieur |
|---|---|---|---|---|---|---|---|
| Référence — graine 0 | sans bruit | 0,04 | 0,05 | 0,000 [0,000 ; 0,001] | < 10⁻⁴ | 0,07 | 0,04 |
| Référence — graine 0 | annexe E | 0,08 | 0,08 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,96 | 0,96 |
| Référence — graine 1 | sans bruit | 0,04 | 0,05 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,18 | 0,11 |
| Référence — graine 1 | annexe E | 0,58 | 0,59 | 0,000 [0,000 ; 0,000] | 0,299 | 0,82 | 0,82 |
| Référence — graine 2 | sans bruit | 0,05 | 0,05 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,21 | 0,12 |
| Référence — graine 2 | annexe E | 0,57 | 0,58 | 0,000 [0,000 ; 0,000] | < 10⁻⁴ | 0,82 | 0,82 |
| Référence — contrôle | sans bruit | 0,80 | 0,81 | 0,005 [0,004 ; 0,005] | < 10⁻⁴ | 0,77 | 0,64 |
| Référence — contrôle | annexe E | 0,80 | 0,81 | 0,005 [0,004 ; 0,005] | < 10⁻⁴ | 0,77 | 0,64 |
| Emphase Traduction — graine 0 | sans bruit | 0,04 | 0,04 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,11 | 0,04 |
| Emphase Traduction — graine 0 | annexe E | 0,04 | 0,04 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,08 | 0,01 |
| Emphase Traduction — graine 1 | sans bruit | 0,04 | 0,04 | 0,000 [0,000 ; 0,001] | < 10⁻⁴ | 0,14 | 0,08 |
| Emphase Traduction — graine 1 | annexe E | 0,04 | 0,04 | 0,000 [0,000 ; 0,001] | < 10⁻⁴ | 0,18 | 0,13 |
| Emphase Traduction — graine 2 | sans bruit | 0,05 | 0,05 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,11 | 0,07 |
| Emphase Traduction — graine 2 | annexe E | 0,05 | 0,05 | 0,000 [0,000 ; 0,001] | < 10⁻⁴ | 0,14 | 0,19 |
| Emphase Contrastive — graine 0 | sans bruit | 0,04 | 0,04 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,05 | 0,03 |
| Emphase Contrastive — graine 0 | annexe E | 0,04 | 0,04 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,06 | 0,02 |
| Emphase Contrastive — graine 1 | sans bruit | 0,05 | 0,05 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,14 | 0,07 |
| Emphase Contrastive — graine 1 | annexe E | 0,05 | 0,05 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,14 | 0,07 |
| Emphase Contrastive — graine 2 | sans bruit | 0,05 | 0,05 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,09 | 0,11 |
| Emphase Contrastive — graine 2 | annexe E | 0,05 | 0,05 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,09 | 0,10 |
| Emphase Contrastive — contrôle | sans bruit | 0,79 | 0,80 | 0,005 [0,004 ; 0,005] | < 10⁻⁴ | 0,75 | 0,62 |
| Emphase Contrastive — contrôle | annexe E | 0,79 | 0,80 | 0,005 [0,004 ; 0,005] | < 10⁻⁴ | 0,75 | 0,62 |
| Emphase Cycle-Complet — graine 0 | sans bruit | 0,04 | 0,04 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,22 | 0,14 |
| Emphase Cycle-Complet — graine 0 | annexe E | 0,53 | 0,58 | 0,003 [0,003 ; 0,004] | < 10⁻⁴ | 0,77 | 0,84 |
| Emphase Cycle-Complet — graine 1 | sans bruit | 0,04 | 0,04 | 0,000 [0,000 ; 0,001] | < 10⁻⁴ | 0,36 | 0,34 |
| Emphase Cycle-Complet — graine 1 | annexe E | 0,40 | 0,43 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,79 | 0,81 |
| Emphase Cycle-Complet — graine 2 | sans bruit | 0,04 | 0,04 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,32 | 0,34 |
| Emphase Cycle-Complet — graine 2 | annexe E | 0,29 | 0,30 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,71 | 0,77 |
| Emphase Cycle-Complet — contrôle | sans bruit | 0,81 | 0,82 | 0,004 [0,003 ; 0,004] | < 10⁻⁴ | 0,81 | 0,64 |
| Emphase Cycle-Complet — contrôle | annexe E | 0,82 | 0,83 | 0,003 [0,003 ; 0,004] | < 10⁻⁴ | 0,73 | 0,58 |
| Emphase Attributs Continus — graine 0 | sans bruit | 0,05 | 0,05 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,32 | 0,27 |
| Emphase Attributs Continus — graine 0 | annexe E | 0,15 | 0,15 | 0,000 [0,000 ; 0,000] | 0,028 | 0,87 | 0,87 |
| Emphase Attributs Continus — graine 1 | sans bruit | 0,05 | 0,05 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,19 | 0,17 |
| Emphase Attributs Continus — graine 1 | annexe E | 0,62 | 0,63 | 0,001 [0,001 ; 0,001] | < 10⁻⁴ | 0,90 | 0,90 |
| Emphase Attributs Continus — graine 2 | sans bruit | 0,05 | 0,05 | 0,001 [0,000 ; 0,001] | < 10⁻⁴ | 0,11 | 0,04 |
| Emphase Attributs Continus — graine 2 | annexe E | 0,71 | 0,71 | 0,000 [0,000 ; 0,000] | 0,299 | 0,55 | 0,55 |
| Emphase Attributs Continus — contrôle | sans bruit | 0,80 | 0,81 | 0,005 [0,005 ; 0,006] | < 10⁻⁴ | 0,77 | 0,64 |
| Emphase Attributs Continus — contrôle | annexe E | 0,80 | 0,81 | 0,005 [0,005 ; 0,006] | < 10⁻⁴ | 0,76 | 0,64 |

### Associations forme → couleur (bruit de l'annexe E, graine 42, 0,01)

Constance = part des images de la forme dont la couleur est colorée (chroma ≥ 40) et à ±30° de la teinte dominante [IC de Wilson] ; n = 1 536 images par forme. Δteinte : écart des teintes dominantes entre extractions [IC 95 %, bootstrap apparié des images]. Association : constance ≥ 60 %, et « sûre » si la borne basse de l'IC l'est aussi.

| Modèle | forme | constance barycentre | constance intérieur | Δteinte dominante |
|---|---|---|---|---|
| Référence — graine 0 | losange | 72% [69%–74%] | 72% [70%–74%] | 0° [0 ; 0] |
| Référence — graine 1 | losange | 69% [67%–71%] | 69% [67%–71%] | 0° [0 ; 0] |
| Référence — graine 1 | œuf | 88% [86%–89%] | 88% [86%–90%] | 0° [0 ; 0] |
| Référence — graine 1 | triangle | 86% [84%–88%] | 86% [84%–88%] | 0° [0 ; 0] |
| Référence — graine 2 | losange | 78% [75%–80%] | 78% [76%–80%] | 0° [0 ; 0] |
| Référence — graine 2 | œuf | 67% [64%–69%] | 67% [64%–69%] | 0° [0 ; 0] |
| Référence — graine 2 | triangle | 60% [58%–63%] | 61% [58%–63%] | 0° [0 ; 0] |
| Emphase Cycle-Complet — graine 0 | œuf | 92% [91%–94%] | 92% [91%–94%] | 0° [0 ; 0] |
| Emphase Cycle-Complet — graine 0 | triangle | 81% [79%–83%] | 83% [81%–85%] | 0° [0 ; 0] |
| Emphase Cycle-Complet — graine 2 | triangle | 73% [71%–75%] | 74% [72%–76%] | 0° [0 ; 0] |
| Emphase Attributs Continus — graine 1 | losange | 66% [63%–68%] | 65% [63%–68%] | 0° [0 ; 0] |
| Emphase Attributs Continus — graine 1 | œuf | 88% [86%–89%] | 88% [86%–89%] | 0° [0 ; 0] |
| Emphase Attributs Continus — graine 1 | triangle | 85% [83%–87%] | 86% [84%–87%] | 0° [0 ; 0] |
| Emphase Attributs Continus — graine 2 | losange | 80% [77%–81%] | 80% [77%–81%] | 0° [0 ; 0] |
| Emphase Attributs Continus — graine 2 | œuf | 94% [93%–95%] | 94% [93%–95%] | 0° [0 ; 0] |
| Emphase Attributs Continus — graine 2 | triangle | 81% [79%–83%] | 81% [79%–83%] | 0° [0 ; 0] |

Associations sûres : 15 avec le barycentre, 15 avec le pixel intérieur, 15 avec les deux.

### Régularité et arbitraire de S selon l'extraction

Unité : la paire de catégories (forme : 3 paires par instance ; orientation : 120) ou la paire d'instances pour une même forme (9 par configuration). « a/b » : paires qui diffèrent au test du mémoire / au critère d'effet (KS brut p < 0,05 et |δ de Cliff| ≥ 0,147), sans correction (voir B.3 pour l'effet des corrections). n = 1 536 images par forme, ≈ 288 par orientation.


|  | sans bruit, barycentre | sans bruit, intérieur | annexe E, barycentre | annexe E, intérieur |
|---|---|---|---|---|
| Référence | F 1/2 sur 9 ; O 142/118 sur 360 | F 0/0 sur 9 ; O 146/115 sur 360 | F 7/8 sur 9 ; O 41/112 sur 360 | F 8/8 sur 9 ; O 45/114 sur 360 |
| Référence, contrôle | F 3/3 sur 3 ; O 1/7 sur 120 | F 3/3 sur 3 ; O 0/0 sur 120 | F 3/3 sur 3 ; O 1/6 sur 120 | F 3/3 sur 3 ; O 0/0 sur 120 |
| Emphase Traduction | F 0/0 sur 9 ; O 95/35 sur 360 | F 0/0 sur 9 ; O 95/37 sur 360 | F 1/1 sur 9 ; O 86/37 sur 360 | F 0/1 sur 9 ; O 97/39 sur 360 |
| Emphase Contrastive | F 0/0 sur 9 ; O 80/82 sur 360 | F 0/0 sur 9 ; O 76/88 sur 360 | F 0/0 sur 9 ; O 77/84 sur 360 | F 0/0 sur 9 ; O 73/86 sur 360 |
| Emphase Contrastive, contrôle | F 2/3 sur 3 ; O 0/1 sur 120 | F 2/3 sur 3 ; O 0/4 sur 120 | F 2/3 sur 3 ; O 0/1 sur 120 | F 2/3 sur 3 ; O 0/4 sur 120 |
| Emphase Cycle-Complet | F 5/6 sur 9 ; O 134/78 sur 360 | F 4/4 sur 9 ; O 125/74 sur 360 | F 8/9 sur 9 ; O 156/203 sur 360 | F 8/9 sur 9 ; O 165/224 sur 360 |
| Emphase Cycle-Complet, contrôle | F 2/3 sur 3 ; O 0/0 sur 120 | F 2/3 sur 3 ; O 0/5 sur 120 | F 2/3 sur 3 ; O 0/1 sur 120 | F 2/3 sur 3 ; O 0/15 sur 120 |
| Emphase Attributs Continus | F 3/4 sur 9 ; O 184/152 sur 360 | F 2/3 sur 9 ; O 184/156 sur 360 | F 7/7 sur 9 ; O 63/141 sur 360 | F 7/7 sur 9 ; O 69/146 sur 360 |
| Emphase Attributs Continus, contrôle | F 3/3 sur 3 ; O 2/6 sur 120 | F 3/3 sur 3 ; O 0/2 sur 120 | F 3/3 sur 3 ; O 0/6 sur 120 | F 3/3 sur 3 ; O 0/2 sur 120 |

### Arbitraire de S : paires d'instances qui diffèrent pour une même forme (mémoire / effet), |δ| maximal

|  | sans bruit, barycentre | sans bruit, intérieur | annexe E, barycentre | annexe E, intérieur |
|---|---|---|---|---|
| Référence | 1/2 sur 9 (|δ| ≤ 0,23) | 1/1 sur 9 (|δ| ≤ 0,24) | 8/9 sur 9 (|δ| ≤ 1,00) | 8/9 sur 9 (|δ| ≤ 1,00) |
| Emphase Traduction | 4/5 sur 9 (|δ| ≤ 0,27) | 3/5 sur 9 (|δ| ≤ 0,27) | 4/4 sur 9 (|δ| ≤ 0,35) | 5/4 sur 9 (|δ| ≤ 0,36) |
| Emphase Contrastive | 1/4 sur 9 (|δ| ≤ 0,22) | 0/4 sur 9 (|δ| ≤ 0,22) | 1/4 sur 9 (|δ| ≤ 0,22) | 0/4 sur 9 (|δ| ≤ 0,22) |
| Emphase Cycle-Complet | 2/4 sur 9 (|δ| ≤ 0,24) | 3/4 sur 9 (|δ| ≤ 0,24) | 7/7 sur 9 (|δ| ≤ 0,81) | 7/7 sur 9 (|δ| ≤ 0,82) |
| Emphase Attributs Continus | 1/3 sur 9 (|δ| ≤ 0,20) | 2/3 sur 9 (|δ| ≤ 0,19) | 8/8 sur 9 (|δ| ≤ 1,00) | 8/8 sur 9 (|δ| ≤ 1,00) |

## B.2 Graines et amplitudes du bruit de l'annexe E

Unité : l'image traduite (1 536 par forme, mêmes attributs dans toutes les conditions). Le bruit est un seul vecteur pour toutes les images (comme dans l'évaluation) : la graine ne tire donc pas un bruit par image mais **une direction du résidu** ; 6 graines = 6 directions.

| graine | vecteur (× 0,01 : losange ; œuf ; triangle) | forme qui reçoit le plus de résidu |
|---|---|---|
| 42 | 0,0034 ; 0,0013 ; 0,0023 | losange → triangle, œuf → losange, triangle → losange |
| 0 | 0,0154 ; −0,0029 ; −0,0218 | losange → œuf, œuf → losange, triangle → losange |
| 1 | 0,0066 ; 0,0027 ; 0,0006 | losange → œuf, œuf → losange, triangle → losange |
| 2 | 0,0039 ; −0,0022 ; −0,0032 | losange → œuf, œuf → losange, triangle → losange |
| 3 | 0,0080 ; 0,0017 ; 0,0009 | losange → œuf, œuf → losange, triangle → losange |
| 4 | −0,0161 ; 0,0023 ; 0,0224 | losange → triangle, œuf → triangle, triangle → œuf |

### Saturation médiane (images), moyenne des 3 instances

| Configuration | sans bruit | amplitude 0,003 : min–max sur 6 graines | amplitude 0,010 : min–max sur 6 graines | amplitude 0,030 : min–max sur 6 graines |
|---|---|---|---|---|
| Référence | 0,05 | 0,15–0,38 (g42 : 0,31) | 0,24–0,48 (g42 : 0,41) | 0,30–0,54 (g42 : 0,50) |
| Emphase Traduction | 0,04 | 0,04–0,04 (g42 : 0,04) | 0,04–0,05 (g42 : 0,04) | 0,05–0,06 (g42 : 0,05) |
| Emphase Contrastive | 0,05 | 0,05–0,05 (g42 : 0,05) | 0,05–0,05 (g42 : 0,05) | 0,05–0,06 (g42 : 0,05) |
| Emphase Cycle-Complet | 0,04 | 0,10–0,43 (g42 : 0,10) | 0,40–0,58 (g42 : 0,41) | 0,56–0,79 (g42 : 0,79) |
| Emphase Attributs Continus | 0,05 | 0,15–0,37 (g42 : 0,25) | 0,29–0,64 (g42 : 0,50) | 0,41–0,76 (g42 : 0,70) |

### Associations sûres (borne basse de Wilson de la constance ≥ 60 %), 15 modèles × 3 formes

| amplitude | graine 42 | graine 0 | graine 1 | graine 2 | graine 3 | graine 4 |
|---|---|---|---|---|---|---|
| 0,003 | 11 | 9 | 12 | 8 | 14 | 5 |
| 0,010 | 15 | 11 | 15 | 9 | 16 | 5 |
| 0,030 | 15 | 11 | 15 | 11 | 15 | 5 |

### Associations du mémoire (graine 42, 0,01) face aux autres graines et amplitudes

Pour chaque association sûre à la graine 42 : dans combien d'autres conditions reste-t-elle sûre **avec la même couleur** (IC 95 % bootstrap apparié de l'écart de teinte dominante inclus dans ±30°) ; entre parenthèses, conditions où le résidu principal part vers la même forme qu'à la graine 42.

| Modèle | forme | constance g42 [IC] | même couleur, autres graines (même receveuse) | même couleur, amplitudes 0,003 et 0,03 | écart de teinte des graines sûres |
|---|---|---|---|---|---|
| Référence — graine 0 | losange | 72% [69%–74%] | 3/5 (1/1) | 1/2 | g1 1°, g3 1°, g4 0° |
| Référence — graine 1 | losange | 69% [67%–71%] | 3/5 (1/1) | 2/2 | g1 9°, g3 3°, g4 10° |
| Référence — graine 1 | œuf | 88% [86%–89%] | 4/5 (4/4) | 2/2 | g0 0°, g1 0°, g2 −1°, g3 0° |
| Référence — graine 1 | triangle | 86% [84%–88%] | 4/5 (4/4) | 2/2 | g0 0°, g1 0°, g2 0°, g3 0° |
| Référence — graine 2 | losange | 78% [75%–80%] | 3/5 (1/1) | 2/2 | g1 −8°, g3 −3°, g4 −8° |
| Référence — graine 2 | œuf | 67% [64%–69%] | 4/5 (4/4) | 2/2 | g0 4°, g1 2°, g2 1°, g3 2° |
| Emphase Cycle-Complet — graine 0 | œuf | 92% [91%–94%] | 0/5 (0/4) | 1/2 | g0 −67°, g1 −51°, g2 −101°, g3 −35° |
| Emphase Cycle-Complet — graine 0 | triangle | 81% [79%–83%] | 1/5 (1/4) | 0/2 | g0 −43°, g1 29°, g2 −55°, g3 5° |
| Emphase Cycle-Complet — graine 2 | triangle | 73% [71%–75%] | 4/5 (4/4) | 1/2 | g0 −13°, g1 −2°, g2 −5°, g3 −5° |
| Emphase Attributs Continus — graine 1 | losange | 66% [63%–68%] | 1/5 (0/1) | 1/2 | g3 −2° |
| Emphase Attributs Continus — graine 1 | œuf | 88% [86%–89%] | 4/5 (4/4) | 2/2 | g0 1°, g1 1°, g2 0°, g3 1° |
| Emphase Attributs Continus — graine 1 | triangle | 85% [83%–87%] | 4/5 (4/4) | 2/2 | g0 −1°, g1 0°, g2 0°, g3 0° |
| Emphase Attributs Continus — graine 2 | losange | 80% [77%–81%] | 3/5 (1/1) | 1/2 | g1 −3°, g3 −1°, g4 −4° |
| Emphase Attributs Continus — graine 2 | œuf | 94% [93%–95%] | 4/5 (4/4) | 2/2 | g0 0°, g1 1°, g2 1°, g3 1° |
| Emphase Attributs Continus — graine 2 | triangle | 81% [79%–83%] | 3/5 (2/4) | 2/2 | g1 −2°, g3 −1°, g4 −3° |

15 associations sûres à la graine 42 ; 0 gardent la même couleur avec les 5 autres graines.

### La couleur dépend-elle de la direction du résidu ?

Pour chaque modèle × forme, paires de graines (amplitude 0,01) où la forme est associée (sûre) dans les deux : |écart de teinte dominante| selon que le résidu principal part vers la même forme ou non. Test : Mann-Whitney unilatéral (même receveuse < autre receveuse) ; unité : la paire de graines (non indépendantes : indication seulement).

Même receveuse : médiane 1° (n = 104 paires) ; autre receveuse : 2° (n = 20) ; p = 0,145.

## B.3 Tests : test du mémoire, KS, critère d'effet, corrections

Unité : la paire de catégories (régularité, au sein d'une instance) ou la paire (instances, catégorie) (arbitraire). Famille de correction : toutes les paires d'un même tableau, chemin et canal. Tableaux complets : `results/saturation/robustesse/tests/tests_*.md`. Ici : chemin de traduction (évaluation exacte), sans bruit et avec le bruit de l'annexe E ; % de paires qui diffèrent.

**Régularité, Traduction, canal H**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Emphase Attributs Continus | 9 | 11 | 0 | 0 | 100 | 100 | 11 | 11 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 0 | 0 |
| Forme | Emphase Cycle-Complet | 9 | 11 | 0 | 0 | 100 | 100 | 44 | 44 |
| Forme | Emphase Traduction | 9 | 0 | 0 | 0 | 100 | 100 | 0 | 0 |
| Forme | Modèles Complets (contrôles) | 12 | 0 | 0 | 0 | 83 | 58 | 0 | 0 |
| Forme | Référence | 9 | 22 | 0 | 22 | 100 | 100 | 33 | 33 |
| Orientation | Emphase Attributs Continus | 360 | 14 | 0 | 2 | 100 | 100 | 39 | 39 |
| Orientation | Emphase Contrastive | 360 | 0 | 0 | 0 | 100 | 100 | 14 | 14 |
| Orientation | Emphase Cycle-Complet | 360 | 34 | 17 | 21 | 100 | 100 | 48 | 48 |
| Orientation | Emphase Traduction | 360 | 0 | 0 | 0 | 100 | 100 | 4 | 4 |
| Orientation | Modèles Complets (contrôles) | 480 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| Orientation | Référence | 360 | 16 | 0 | 4 | 100 | 100 | 42 | 42 |

**Régularité, Traduction, canal S**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Emphase Attributs Continus | 9 | 22 | 0 | 22 | 100 | 100 | 44 | 44 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 0 | 0 |
| Forme | Emphase Cycle-Complet | 9 | 56 | 11 | 44 | 100 | 100 | 67 | 67 |
| Forme | Emphase Traduction | 9 | 0 | 0 | 0 | 100 | 100 | 11 | 11 |
| Forme | Modèles Complets (contrôles) | 12 | 75 | 67 | 67 | 100 | 100 | 100 | 100 |
| Forme | Référence | 9 | 11 | 0 | 0 | 100 | 100 | 22 | 22 |
| Orientation | Emphase Attributs Continus | 360 | 39 | 8 | 22 | 100 | 100 | 38 | 38 |
| Orientation | Emphase Contrastive | 360 | 12 | 1 | 6 | 100 | 100 | 18 | 18 |
| Orientation | Emphase Cycle-Complet | 360 | 21 | 1 | 8 | 100 | 100 | 13 | 13 |
| Orientation | Emphase Traduction | 360 | 10 | 0 | 2 | 100 | 100 | 6 | 6 |
| Orientation | Modèles Complets (contrôles) | 480 | 0 | 0 | 0 | 95 | 88 | 1 | 1 |
| Orientation | Référence | 360 | 29 | 7 | 18 | 100 | 100 | 28 | 28 |

**Régularité, Traduction bruitée (annexe E), canal H**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Emphase Attributs Continus | 9 | 67 | 33 | 56 | 100 | 100 | 67 | 67 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 0 | 0 |
| Forme | Emphase Cycle-Complet | 9 | 33 | 33 | 33 | 100 | 100 | 56 | 56 |
| Forme | Emphase Traduction | 9 | 0 | 0 | 0 | 100 | 100 | 11 | 11 |
| Forme | Modèles Complets (contrôles) | 12 | 0 | 0 | 0 | 83 | 67 | 0 | 0 |
| Forme | Référence | 9 | 67 | 44 | 67 | 100 | 100 | 67 | 67 |
| Orientation | Emphase Attributs Continus | 360 | 34 | 17 | 28 | 100 | 100 | 51 | 51 |
| Orientation | Emphase Contrastive | 360 | 0 | 0 | 0 | 100 | 100 | 13 | 13 |
| Orientation | Emphase Cycle-Complet | 360 | 48 | 25 | 36 | 100 | 100 | 53 | 53 |
| Orientation | Emphase Traduction | 360 | 0 | 0 | 0 | 100 | 100 | 9 | 9 |
| Orientation | Modèles Complets (contrôles) | 480 | 0 | 0 | 0 | 12 | 1 | 0 | 0 |
| Orientation | Référence | 360 | 32 | 13 | 20 | 100 | 100 | 42 | 42 |

**Régularité, Traduction bruitée (annexe E), canal S**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Emphase Attributs Continus | 9 | 78 | 56 | 78 | 100 | 100 | 78 | 78 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 0 | 0 |
| Forme | Emphase Cycle-Complet | 9 | 67 | 56 | 67 | 100 | 100 | 100 | 100 |
| Forme | Emphase Traduction | 9 | 11 | 0 | 0 | 100 | 100 | 22 | 22 |
| Forme | Modèles Complets (contrôles) | 12 | 75 | 67 | 67 | 100 | 100 | 100 | 100 |
| Forme | Référence | 9 | 78 | 67 | 67 | 100 | 100 | 89 | 89 |
| Orientation | Emphase Attributs Continus | 360 | 12 | 0 | 3 | 100 | 100 | 38 | 38 |
| Orientation | Emphase Contrastive | 360 | 12 | 1 | 6 | 100 | 100 | 18 | 18 |
| Orientation | Emphase Cycle-Complet | 360 | 40 | 7 | 24 | 100 | 100 | 59 | 59 |
| Orientation | Emphase Traduction | 360 | 12 | 0 | 3 | 100 | 100 | 8 | 8 |
| Orientation | Modèles Complets (contrôles) | 480 | 0 | 0 | 0 | 95 | 86 | 1 | 1 |
| Orientation | Référence | 360 | 10 | 0 | 4 | 100 | 100 | 27 | 27 |

**Arbitraire, Traduction, canal H**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Référence | 9 | 33 | 0 | 11 | 100 | 100 | 89 | 89 |
| Forme | Emphase Attributs Continus | 9 | 11 | 0 | 0 | 100 | 100 | 11 | 11 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 11 | 11 |
| Forme | Emphase Cycle-Complet | 9 | 44 | 11 | 22 | 100 | 100 | 78 | 78 |
| Forme | Emphase Traduction | 9 | 0 | 0 | 0 | 100 | 100 | 11 | 11 |
| Orientation | Référence | 48 | 25 | 6 | 19 | 100 | 100 | 40 | 40 |
| Orientation | Emphase Attributs Continus | 48 | 2 | 0 | 2 | 100 | 100 | 21 | 21 |
| Orientation | Emphase Contrastive | 48 | 0 | 0 | 0 | 100 | 100 | 17 | 17 |
| Orientation | Emphase Cycle-Complet | 48 | 50 | 35 | 40 | 100 | 100 | 65 | 65 |
| Orientation | Emphase Traduction | 48 | 0 | 0 | 0 | 100 | 100 | 23 | 23 |

**Arbitraire, Traduction, canal S**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Référence | 9 | 11 | 0 | 11 | 100 | 100 | 22 | 22 |
| Forme | Emphase Attributs Continus | 9 | 11 | 0 | 0 | 100 | 100 | 22 | 22 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 44 | 44 |
| Forme | Emphase Cycle-Complet | 9 | 11 | 0 | 11 | 100 | 100 | 33 | 33 |
| Forme | Emphase Traduction | 9 | 44 | 0 | 0 | 100 | 100 | 56 | 56 |
| Orientation | Référence | 48 | 31 | 10 | 19 | 100 | 100 | 33 | 33 |
| Orientation | Emphase Attributs Continus | 48 | 21 | 2 | 10 | 100 | 100 | 15 | 15 |
| Orientation | Emphase Contrastive | 48 | 31 | 6 | 12 | 100 | 100 | 40 | 40 |
| Orientation | Emphase Cycle-Complet | 48 | 31 | 2 | 10 | 100 | 100 | 17 | 17 |
| Orientation | Emphase Traduction | 48 | 44 | 6 | 31 | 100 | 100 | 44 | 44 |

**Arbitraire, Traduction bruitée (annexe E), canal H**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Référence | 9 | 89 | 33 | 89 | 100 | 100 | 89 | 89 |
| Forme | Emphase Attributs Continus | 9 | 89 | 78 | 89 | 100 | 100 | 78 | 78 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 0 | 0 |
| Forme | Emphase Cycle-Complet | 9 | 100 | 78 | 100 | 100 | 100 | 100 | 100 |
| Forme | Emphase Traduction | 9 | 0 | 0 | 0 | 100 | 100 | 11 | 11 |
| Orientation | Référence | 48 | 79 | 23 | 77 | 100 | 100 | 42 | 42 |
| Orientation | Emphase Attributs Continus | 48 | 90 | 48 | 83 | 100 | 100 | 100 | 100 |
| Orientation | Emphase Contrastive | 48 | 0 | 0 | 0 | 100 | 100 | 17 | 17 |
| Orientation | Emphase Cycle-Complet | 48 | 94 | 69 | 94 | 100 | 100 | 85 | 85 |
| Orientation | Emphase Traduction | 48 | 0 | 0 | 0 | 100 | 100 | 19 | 19 |

**Arbitraire, Traduction bruitée (annexe E), canal S**

| attribut | groupe | n paires | mémoire | mémoire + Holm | mémoire + BH | KS | KS + Holm | effet | effet + Holm |
|---|---|---|---|---|---|---|---|---|---|
| Forme | Référence | 9 | 89 | 56 | 89 | 100 | 100 | 100 | 100 |
| Forme | Emphase Attributs Continus | 9 | 89 | 78 | 89 | 100 | 100 | 89 | 89 |
| Forme | Emphase Contrastive | 9 | 0 | 0 | 0 | 100 | 100 | 44 | 44 |
| Forme | Emphase Cycle-Complet | 9 | 78 | 22 | 78 | 100 | 100 | 67 | 67 |
| Forme | Emphase Traduction | 9 | 33 | 22 | 33 | 100 | 100 | 44 | 44 |
| Orientation | Référence | 48 | 69 | 67 | 67 | 100 | 100 | 73 | 73 |
| Orientation | Emphase Attributs Continus | 48 | 81 | 69 | 79 | 100 | 100 | 92 | 92 |
| Orientation | Emphase Contrastive | 48 | 29 | 6 | 25 | 100 | 100 | 44 | 44 |
| Orientation | Emphase Cycle-Complet | 48 | 73 | 29 | 71 | 100 | 100 | 75 | 75 |
| Orientation | Emphase Traduction | 48 | 62 | 21 | 58 | 100 | 100 | 58 | 58 |

## B.4 Régression : sensibilité au sous-ensemble d'images

Unité : le modèle. R² de la chrominance (validation croisée à 5 plis, meilleure de ridge et kNN), référence « graine 0, 32 par case (mémoire) ». Écart : R² variante − R² référence, médiane [min ; max] sur les 19 modèles. ρ : Spearman du classement des 15 modèles « vide sémantique », p corrigé (Holm, sur tous les ρ du tableau).

| variables | variante | n modèles | écart de R² médian [min ; max] | ρ | p (Holm) | R² max des contrôles |
|---|---|---|---|---|---|---|
| résidu | graine 0, 16 par case | 19 | 0,00 [−0,02 ; 0,00] | 1,00 | < 10⁻⁴ | 0,00 |
| résidu | graine 0, 64 par case | 19 | 0,00 [0,00 ; 0,02] | 1,00 | < 10⁻⁴ | 0,03 |
| résidu | graine 1, 32 par case | 19 | 0,00 [−0,01 ; 0,01] | 0,99 | < 10⁻⁴ | 0,02 |
| résidu | graine 2, 32 par case | 19 | 0,00 [−0,01 ; 0,01] | 1,00 | < 10⁻⁴ | 0,01 |
| continus + géométrie | graine 0, 16 par case | 19 | −0,08 [−0,09 ; −0,02] | 1,00 | < 10⁻⁴ | 0,05 |
| continus + géométrie | graine 0, 64 par case | 19 | 0,11 [0,07 ; 0,14] | 0,99 | < 10⁻⁴ | 0,24 |
| continus + géométrie | graine 1, 32 par case | 19 | 0,00 [−0,01 ; 0,01] | 0,99 | < 10⁻⁴ | 0,14 |
| continus + géométrie | graine 2, 32 par case | 19 | 0,00 [−0,01 ; 0,01] | 1,00 | < 10⁻⁴ | 0,13 |
| tout | graine 0, 16 par case | 19 | −0,07 [−0,10 ; −0,03] | 0,99 | < 10⁻⁴ | 0,06 |
| tout | graine 0, 64 par case | 19 | 0,09 [0,03 ; 0,13] | 1,00 | < 10⁻⁴ | 0,28 |
| tout | graine 1, 32 par case | 19 | 0,00 [−0,01 ; 0,01] | 1,00 | < 10⁻⁴ | 0,15 |
| tout | graine 2, 32 par case | 19 | 0,00 [−0,01 ; 0,00] | 1,00 | < 10⁻⁴ | 0,15 |
| latent GW | graine 0, 16 par case | 19 | −0,07 [−0,25 ; 0,00] | 0,94 | < 10⁻⁴ | 0,98 |
| latent GW | graine 0, 64 par case | 19 | 0,17 [0,00 ; 0,39] | 0,84 | 0,0001 | 0,98 |
| latent GW | graine 1, 32 par case | 19 | 0,00 [−0,01 ; 0,01] | 0,98 | < 10⁻⁴ | 0,98 |
| latent GW | graine 2, 32 par case | 19 | 0,00 [−0,01 ; 0,00] | 0,99 | < 10⁻⁴ | 0,98 |
