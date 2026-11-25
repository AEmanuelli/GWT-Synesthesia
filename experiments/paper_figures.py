"""Figures de l'article (branche publication), limitées aux réglages retenus.

Refait, à partir des données en cache (aucun modèle n'est relancé) :
  - fig1_associations_galerie.png et figS7_palettes.png (synesthesia_figure.py) ;
  - fig2_saturation_dose_reponse.png, fig5_deplacement_chrominance_<réglage>.png et
    figS6_teintes_par_instance.png (shape_noise_sweep.py) ;
  - fig3_couleur_lisible_regression.png (residual_regression.py) ;
  - figS1 à figS4, saturation par chemin et par forme (saturation_analysis.py).
Les figures de la greffe et de la carte du code sont produites par colour_code_transplant.py et
colour_code_map.py.

    python experiments/paper_figures.py                                   # Référence et Emphase Cycle-Complet
    python experiments/paper_figures.py --configs high_cycles --out /tmp/figures
"""
import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402

SLUG = {"base_params": "reference", "high_translation": "emphase_traduction",
        "high_contrastive": "emphase_contrastive", "high_cycles": "emphase_cycle_complet",
        "high_alpha": "emphase_attributs_continus"}


def sweep_figures(configs, out, tmp):
    from shape_noise_sweep import figures, load_sweeps
    sweeps = load_sweeps(paths.RESULTS_DIR / "saturation" / "balayage")
    keys = [k for k in sweeps if k[0] in configs]
    keys.sort(key=lambda k: (configs.index(k[0]), k[1] == "control", k[1]))
    figures(sweeps, keys, 0.01, tmp)
    shutil.copy(tmp / "saturation_vs_epsilon.png", out / "fig2_saturation_dose_reponse.png")
    shutil.copy(tmp / "teintes_par_instance.png", out / "figS6_teintes_par_instance.png")
    for c in configs:
        name = "fig5" if c == "high_cycles" else "figS5"
        shutil.copy(tmp / f"chrominance_{c}.png", out / f"{name}_deplacement_chrominance_{SLUG[c]}.png")


def regression_figure(configs, out):
    import pandas as pd
    from residual_regression import figure, read_transmission
    df = pd.read_csv(paths.RESULTS_DIR / "saturation" / "regression" / "r2.csv")
    figure(df[df["config"].isin(configs)].copy(), read_transmission(), out / "fig3_couleur_lisible_regression.png")


def association_figures(configs, out):
    from synesthesia_figure import gallery_figure, noise_robust, palettes_figure, read_csv
    d = paths.RESULTS_DIR / "saturation" / "synesthesie"
    stats = [r for r in read_csv(d / "associations.csv") if r["config"] in configs]
    z = np.load(d / "bandes.npz")
    strips = {tuple(k.split("|")): z[k] for k in z.files}
    z = np.load(d / "exemples.npz")
    images = {tuple(k.split("|")): z[k] for k in z.files}
    stable = noise_robust(paths.RESULTS_DIR / "saturation" / "robustesse" / "bruit_conditions.csv")
    gallery_figure(stats, strips, images, out / "fig1_associations_galerie.png", stable)
    palettes_figure(stats, strips, images, out / "figS7_palettes.png")


def saturation_figures(configs, out, tmp):
    from saturation_analysis import figures
    for cache, prefix in (("traduction_seule/analysis", "traduction"), ("apercu_chapitre3", "cycle")):
        models = pickle.load(open(paths.RESULTS_DIR / "saturation" / cache / "cache_distributions.pkl", "rb"))
        models = {k: v for k, v in models.items() if k[0] in configs}
        sub = tmp / prefix
        figures(models, sub)
        del models
        shutil.copy(sub / "S_par_chemin.png",
                    out / ("figS1_S_par_chemin_traduction.png" if prefix == "traduction"
                           else "figS2_S_par_chemin_trois_chemins.png"))
        for c in configs:
            n = "figS3" if prefix == "traduction" else "figS4"
            shutil.copy(sub / f"S_par_forme_{c}.png", out / f"{n}_S_par_forme_{prefix}_{SLUG[c]}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["base_params", "high_cycles"])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "figures")
    ap.add_argument("--only", nargs="+", choices=("balayage", "regression", "associations", "saturation"),
                    default=["balayage", "regression", "associations", "saturation"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = paths.RESULTS_DIR / "publication" / "figures_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if "balayage" in args.only:
        sweep_figures(args.configs, args.out, tmp)
    if "regression" in args.only:
        regression_figure(args.configs, args.out)
    if "associations" in args.only:
        association_figures(args.configs, args.out)
    if "saturation" in args.only:
        saturation_figures(args.configs, args.out, tmp)
    print(f"Figures dans {args.out}")


if __name__ == "__main__":
    main()
