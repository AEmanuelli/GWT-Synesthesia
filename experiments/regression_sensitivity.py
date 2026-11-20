"""Robustesse de la lecture des résidus par régression (étape 4, volet 4).

Compare les R² de residual_regression.py obtenus avec le sous-ensemble d'images du mémoire
(--seed 0 --per-cell 32) à ceux d'autres tirages (--seed 1, 2) et d'autres tailles (--per-cell 16, 64).
Les variantes sont calculées avec ridge et kNN seulement (le perceptron prendrait ~1 h par variante
sur le 940MX) : la comparaison porte donc sur la meilleure de ces deux régressions, dans la référence
comme dans les variantes.

    python experiments/residual_regression.py --seed 1 --per-cell 32 --regressions ridge kNN \\
        --out results/saturation/robustesse/regression/graine1_par_case32
    python experiments/regression_sensitivity.py

Sorties : results/saturation/robustesse/regression/sensibilite.md et .csv.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
from shape_noise_sweep import CONFIG_LABELS, md_table  # noqa: E402

GROUPS = {"residu": "résidu", "continus": "continus", "continus_contexte": "continus + géométrie",
          "les_deux_contexte": "tout", "latent_gw": "latent GW"}
REGS = ("ridge", "kNN")


def read(dir_):
    rows = []
    for f in sorted((dir_ / "donnees").glob("*_r2_palette.csv")):
        m = re.match(r"(.+)_(seed\d|control)_r2_palette", f.stem)
        d = pd.read_csv(f)
        d = d[(d.etiquettes == "vraies") & d.regression.isin(REGS) & d.groupe.isin(GROUPS)]
        best = d.groupby("groupe").r2_chrominance.max()
        for g, v in best.items():
            rows.append(dict(config=m.group(1), instance=m.group(2), groupe=g, r2=v))
    return pd.DataFrame(rows)


def fr(x):
    return f"{x:.2f}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = paths.RESULTS_DIR / "saturation"
    parser.add_argument("--reference", type=Path, default=root / "regression")
    parser.add_argument("--variants", type=Path, default=root / "robustesse" / "regression")
    args = parser.parse_args()
    tables = {"graine 0, 32 par case (mémoire)": read(args.reference)}
    for d in sorted(args.variants.glob("graine*_par_case*")):
        m = re.match(r"graine(\d+)_par_case(\d+)", d.name)
        t = read(d)
        if not t.empty:
            tables[f"graine {m.group(1)}, {m.group(2)} par case"] = t
    df = pd.concat([t.assign(variante=k) for k, t in tables.items()])
    df.to_csv(args.variants / "sensibilite.csv", index=False)
    ref_name = next(iter(tables))
    L = ["# Régression : sensibilité au sous-ensemble d'images\n",
         "Meilleure de ridge et kNN (R² de la chrominance en validation croisée), moyenne des 3 graines "
         "par configuration ; contrôles : moyenne des 4 Modèles Complets. Écart = max |R² variante − R² "
         "mémoire| sur les modèles.\n"]
    variants = list(tables)
    for g, gl in GROUPS.items():
        L.append(f"## {gl}\n")
        rows = []
        for exp in CONFIG_LABELS:
            for ctrl in (False, True):
                cells = []
                for v in variants:
                    x = df[(df.variante == v) & (df.groupe == g) & (df.config == exp) & ((df.instance == "control") == ctrl)]
                    cells.append(fr(x.r2.mean()) if len(x) else "–")
                if any(c != "–" for c in cells):
                    rows.append([CONFIG_LABELS[exp] + (", contrôle" if ctrl else "")] + cells)
        L.append(md_table(["", *variants], rows) + "\n")
        ref = df[(df.variante == ref_name) & (df.groupe == g)].set_index(["config", "instance"]).r2
        notes = []
        for v in variants[1:]:
            x = df[(df.variante == v) & (df.groupe == g)].set_index(["config", "instance"]).r2
            common = ref.index.intersection(x.index)
            sem = [k for k in common if k[1] != "control"]
            rho = spearmanr(ref[sem], x[sem]).statistic if len(sem) > 2 else np.nan
            notes.append(f"{v} : écart max {fr((x[common] - ref[common]).abs().max())}, "
                         f"médian {fr((x[common] - ref[common]).abs().median())} ; ρ de Spearman du "
                         f"classement des 15 modèles {fr(rho)}")
        L += [f"- {n}" for n in notes] + [""]
    (args.variants / "sensibilite.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
