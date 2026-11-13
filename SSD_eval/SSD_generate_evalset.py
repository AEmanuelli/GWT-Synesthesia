"""Génère le jeu de test structuré du mémoire (§2.1.2).

Par défaut : 3 formes x 8 tailles (7..14 px) x 16 positions (4x4) x 16 orientations
(pas de pi/8) x 25 teintes (HLS, S~0.90, L~0.70) = 153 600 images 32x32.

    python SSD_eval/SSD_generate_evalset.py                 # jeu complet -> data/evaluation_set
    python SSD_eval/SSD_generate_evalset.py --quick         # mini-jeu de test rapide (~1 min)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 # Importation nécessaire si generate_image l'utilise implicitement via matplotlib/patches
import matplotlib.path as mpath
from matplotlib import patches
import csv
import argparse
import math
from pathlib import Path
from PIL import Image # Importation nécessaire si generate_image l'utilise implicitement
from typing import cast
# Supposons que ces fonctions proviennent de modules externes comme indiqué dans le script original
# Si elles sont définies localement ou dans un fichier utils spécifique, ajustez l'importation
import matplotlib
matplotlib.use("Agg")
from simple_shapes_dataset.cli import generate_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from SSD_utils import generate_fixed_colors, generate_fixed_colors_original


# ====================
# FONCTION DE GENERATION DU DATASET
# ====================

def generate_dataset(output_dir: str, imsize: int = 32, min_scale: int = 7, max_scale: int = 14,
                     n_sizes: int = 4, n_colors: int = 100, seed: int = 0,
                     rotations_array: np.ndarray = None, positions_array: np.ndarray = None,
                     shapes: list[int] = [0, 1, 2], gen_col = None) -> None:
    """
    Génère un jeu de données d'images de formes simples avec leurs attributs.

    Args:
        output_dir: Répertoire où sauvegarder les images et le fichier CSV.
        imsize: Taille des images générées (imsize x imsize).
        min_scale: Taille minimale de la forme.
        max_scale: Taille maximale de la forme.
        n_sizes: Nombre de tailles différentes à générer entre min_scale et max_scale.
        n_colors: Nombre de couleurs différentes à générer.
        seed: Graine pour la génération aléatoire des couleurs (reproductibilité), optionnel mais utile si volonté d'introduire une génération de couleur aléatoire.
        Paramètre inactif pour le moment.
        rotations_array: Tableau NumPy des rotations à appliquer (en radians).
                         Si None, utilise [0, pi/2, pi, 3pi/2].
        positions_array: Tableau NumPy des positions [x, y] du centre des formes.
                         Si None, utilise les 4 coins avec une marge.
        shapes: Liste des identifiants des formes à générer (ex: [0, 1, 2] pour cercle, carré, triangle).
    """
    print("--- Démarrage de la génération du dataset ---")
    np.random.seed(seed)

    # Définition des rotations par défaut si non fournies
    if rotations_array is None:
        rotations_array = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        print(f"Utilisation des rotations par défaut: {rotations_array}")

    # Définition des positions par défaut si non fournies
    if positions_array is None:
        margin = max_scale // 2 + 1 # Ajout d'une marge pour éviter les coupures
        positions_array = np.array([
            [margin, margin],
            [margin, imsize - margin],
            [imsize - margin, margin],
            [imsize - margin, imsize - margin]
        ])
        print(f"Utilisation des positions par défaut:\n{positions_array}")

    # Calcul des tailles et génération des couleurs
    sizes = np.linspace(min_scale, max_scale, n_sizes, dtype=int)
    print(f"Génération avec {len(sizes)} tailles: {sizes}")

    if gen_col == "original":
        output_dir = output_dir + "_original"
        rgb_colors, _ = generate_fixed_colors_original(n_colors, seed=seed) # Utilisation de la seed ici aussi
        print(f"Génération avec {n_colors} couleurs uniques.")

    elif gen_col == "single_gray":
        rgb_colors = np.tile(np.array([[127, 127, 127]]), (n_colors, 1))
    else : 
        rgb_colors, _ = generate_fixed_colors(n_colors)
        print(f"Génération avec {n_colors} couleurs uniques.")

    

    # Création du répertoire de sortie et du fichier CSV
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_file = out_dir / "attributes.csv"
    print(f"Le dataset sera sauvegardé dans: {out_dir.resolve()}")

    total_expected = len(shapes) * n_colors * len(sizes) * len(rotations_array) * len(positions_array)
    print(f"Nombre total d'images attendues: {total_expected}")
    count = 0

    # Boucle de génération
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "color_index", "size", "rotation", "location"])
        for color_idx in range(n_colors):
            for cls in shapes:
                for size in sizes:
                    for rotation in rotations_array:
                        for pos in positions_array:
                            color = rgb_colors[color_idx]
                            dpi = 1
                            
                            fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
                            ax = cast(plt.Axes, ax)
                            
                            generate_image(ax, cls, pos, size, rotation, color, imsize)
                            ax.set_facecolor("black")
                            plt.tight_layout(pad=0)
                            
                            filename = f"image_{count:05d}.png"
                            filepath = out_dir / filename
                            plt.savefig(filepath, dpi=dpi, format="png")
                            plt.close(fig)
                            writer.writerow([filename, cls, color_idx, size, rotation, pos.tolist()])
                            count += 1
                            if count % 1000 == 0:
                                print(f"... {count}/{total_expected} images générées.")

    print(f"\n--- Génération terminée ---")
    print(f"Dataset créé avec {count} images (attendu: {total_expected}).")
    print(f"Fichier d'attributs sauvegardé : {csv_file.resolve()}")



# ====================
# MAIN : ARGUMENTS & EXECUTION
# ====================
THESIS_POSITIONS = [8, 13, 18, 24]  # 4 valeurs en x et 4 en y -> 16 positions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(paths.EVALSET_DIR))
    parser.add_argument("--n-colors", type=int, default=25, help="Nombre de teintes (25 dans le mémoire)")
    parser.add_argument("--n-sizes", type=int, default=8)
    parser.add_argument("--min-scale", type=int, default=7)
    parser.add_argument("--max-scale", type=int, default=14)
    parser.add_argument("--n-rotations", type=int, default=16)
    parser.add_argument("--quick", action="store_true",
                        help="Mini-jeu (2 teintes, 2 tailles, 4 rotations, 4 positions) pour tester le pipeline")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    positions = THESIS_POSITIONS
    if args.quick:
        args.n_colors, args.n_sizes, args.n_rotations = 2, 2, 4
        positions = [8, 24]
        if args.output_dir == str(paths.EVALSET_DIR):
            args.output_dir = str(paths.DATA_DIR / "evaluation_set_quick")
    rotation = np.array([i * 2 * np.pi / args.n_rotations for i in range(args.n_rotations)])
    positions_array = np.array([[x, y] for x in positions for y in positions])
    generate_dataset(
        rotations_array=rotation,
        positions_array=positions_array,
        output_dir=args.output_dir,
        imsize=32,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        n_sizes=args.n_sizes,
        n_colors=args.n_colors,
        seed=0,
        gen_col="normal",
    )
