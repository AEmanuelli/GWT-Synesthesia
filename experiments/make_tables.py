"""Génère les tableaux B.2, 3.1 et E.1 du mémoire à partir des résultats d'évaluation.

    python experiments/make_tables.py results/arbitrary                    # tableaux B.2 et 3.1
    python experiments/make_tables.py results/arbitrary_shape_noise_0.01   # "Traduction Bruitée" (E.1)

Critères (mémoire §2.3.2), appliqués aux analysis_results.pkl de chaque instance :
  - Régularité : une paire de catégories d'un attribut est "régulière" dans une instance si
    le test KS est significatif (p < alpha) sur la teinte ET sur au moins un canal R, G ou B.
    Une catégorie "contribue" si elle appartient à au moins une paire régulière.
  - Arbitraire : une catégorie est "spécifique" si sa distribution de teinte diffère
    significativement (KS, p < alpha) entre au moins une paire d'instances.

Score d'arbitraire = Nb catégories spécifiques / Nb catégories ayant contribué à la régularité
(dans au moins une instance). Un tiret signale l'absence de régularité.
Le numérateur peut se lire de deux façons (option --numerator) :
  - all (défaut) : catégories spécifiques parmi toutes celles de l'attribut. C'est la lecture
    qui redonne les scores Forme et Orientation du mémoire ; le numérateur peut alors dépasser
    le dénominateur (ex. Taille) ;
  - contributing : catégories spécifiques parmi celles qui ont contribué à la régularité.
Filtre du script d'origine (actif par défaut, --no-first-instance-filter pour le retirer) :
un attribut n'est comparé entre instances que si la première instance présente au moins une
paire significative sur la teinte, sur l'un des trois chemins. Sinon le numérateur vaut 0 :
ce zéro signifie « non comparé », pas « identique entre instances ».
Avec --denominator sum, numérateur et dénominateur sont sommés sur les instances
(catégories contribuant dans chaque instance).

Si les Modèles Complets ont été évalués (results/regularity/<config>_control), le nombre de
paires régulières de chacun est affiché à la suite (0 attendu partout).
"""

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
from SSD_utils import comparison_metrics  # noqa: E402  (même test que l'évaluation)

ATTRIBUTES = [("shape", "Forme"), ("size", "Taille"), ("position_x", "Pos X"),
              ("position_y", "Pos Y"), ("rotation", "Orientation")]
PATHS = [("translated", "Traduction"), ("half_cycle", "Demi-Cycle"), ("full_cycle", "Cycle Complet")]
CONFIG_LABELS = {"base_params": "Référence", "high_translation": "Emphase Traduction",
                 "high_contrastive": "Emphase Contrastive", "high_cycles": "Emphase Cycle-Complet",
                 "high_alpha": "Emphase Attributs Continus"}


def is_regular_pair(metrics, alpha):
    p_h = metrics.get("ks_pval")
    if p_h is None or not np.isfinite(p_h) or p_h >= alpha:
        return False
    return any(metrics.get(f"ks_pval_{c}", 1.0) < alpha for c in "rgb")


def contributing_categories(results, attr, path, alpha):
    pairs = (results.get(attr) or {}).get(f"bin_comparison_metrics_{path}") or {}
    cats = set()
    for (b1, b2), m in pairs.items():
        if isinstance(m, dict) and is_regular_pair(m, alpha):
            cats.update((b1, b2))
    return cats


def hue_values(results, attr, path, cat):
    vals = (((results.get(attr) or {}).get(f"{path}_distributions") or {}).get(cat) or {}).get("H") or []
    return np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="ex. results/arbitrary")
    parser.add_argument("--numerator", choices=["all", "contributing"], default="all")
    parser.add_argument("--denominator", choices=["union", "sum"], default="union")
    parser.add_argument("--no-first-instance-filter", action="store_true",
                        help="Comparer aussi les attributs non significatifs dans la première instance")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--details", action="store_true", help="Lister les catégories par instance")
    args = parser.parse_args()

    manifest = yaml.safe_load(open(REPO_ROOT / "experiments" / "thesis_runs.yaml"))["experiments"]
    header = f"{'Configuration':36s}{'Chemin':15s}" + "".join(f"{lab:>13s}" for _, lab in ATTRIBUTES)
    table_b2, table_arb = [header], [header]

    for exp_name, exp in manifest.items():
        runs = {}
        for run in exp["runs"]:
            f = args.results_dir / exp_name / f"results_{exp_name}_sans_couleurs_{run['run_id']}" / "analysis_results.pkl"
            if f.exists():
                runs[run["run_id"]] = pickle.load(open(f, "rb"))
        if not runs:
            continue
        label = f"{CONFIG_LABELS.get(exp_name, exp_name)} ({len(runs)} inst.)"
        for path, path_label in PATHS:
            row_b2, row_arb = [], []
            for attr, attr_label in ATTRIBUTES:
                per_inst = {rid: contributing_categories(r, attr, path, args.alpha) for rid, r in runs.items()}
                n_regular = sum(1 for c in per_inst.values() if c)
                row_b2.append(f"{100 * n_regular / len(runs):.0f}")
                union = set().union(*per_inst.values())
                if not union:
                    row_arb.append("-")
                    continue
                categories = set()
                for r in runs.values():
                    categories.update((r.get(attr) or {}).get(f"{path}_distributions") or {})
                candidates = categories if args.numerator == "all" and args.denominator == "union" else union
                first = next(iter(runs.values())).get(attr) or {}
                compared = args.no_first_instance_filter or any(
                    isinstance(m, dict) and m.get("ks_pval") is not None and m["ks_pval"] < args.alpha
                    for p, _ in PATHS for m in (first.get(f"bin_comparison_metrics_{p}") or {}).values())
                specific = set()
                for cat in (candidates if compared else ()):
                    for r1, r2 in itertools.combinations(runs, 2):
                        h1, h2 = (hue_values(runs[r], attr, path, cat) for r in (r1, r2))
                        if len(h1) > 1 and len(h2) > 1 and comparison_metrics(h1, h2, args.num_bins)[4] < args.alpha:
                            specific.add(cat)
                            break
                if args.denominator == "union":
                    row_arb.append(f"{len(specific)}/{len(union)}")
                else:
                    num = sum(len(c & specific) for c in per_inst.values())
                    den = sum(len(c) for c in per_inst.values())
                    row_arb.append(f"{num}/{den}")
                if args.details:
                    print(f"[{exp_name} | {path} | {attr}] " + " ; ".join(
                        f"{rid}: {sorted(c)}" for rid, c in per_inst.items()) + f" -> spécifiques : {sorted(specific)}")
            table_b2.append(f"{label:36s}{path_label:15s}" + "".join(f"{v:>13s}" for v in row_b2))
            table_arb.append(f"{label:36s}{path_label:15s}" + "".join(f"{v:>13s}" for v in row_arb))

    print("\nTableau B.2 — % d'instances avec au moins une paire de catégories régulière")
    print("\n".join(table_b2))
    print(f"\nTableaux 3.1 / E.1 — scores d'arbitraire "
          f"(numérateur : {args.numerator}, dénominateur : {args.denominator})")
    print("\n".join(table_arb))

    # Modèles Complets (contrôles), évalués par scripts/evaluate_all.sh
    control_lines = []
    for exp_name in manifest:
        for f in sorted((args.results_dir.parent / "regularity" / f"{exp_name}_control").glob(
                "*/within_model_comparison_summary.json")):
            stats = json.load(open(f))["path_stats"]
            control_lines.append(f"{CONFIG_LABELS.get(exp_name, exp_name):36s}" + "".join(
                f"{stats.get(path, {}).get('significant', 0):>15d}" for path, _ in PATHS))
    if control_lines:
        print("\nModèles Complets (contrôles) — nombre de paires de catégories régulières")
        print(f"{'Configuration':36s}" + "".join(f"{lab:>15s}" for _, lab in PATHS))
        print("\n".join(control_lines))


if __name__ == "__main__":
    main()
