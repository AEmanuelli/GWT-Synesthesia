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

Saturation et luminosité (évaluation lancée avec --hls, résultats dans results/saturation/) :
    python experiments/make_tables.py results/saturation/arbitrary --channel S
Avec --channel S (ou L), une paire de catégories est régulière si le test KS est significatif
sur S (ou L) ; l'arbitraire et le filtre « première instance » portent sur ce même canal.
--test raw remplace le test du mémoire (KS sur des histogrammes de 15 classes) par le test KS
sur les valeurs brutes, pour la régularité comme pour l'arbitraire.
"""

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ks_2samp

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


def channel_values(results, attr, path, cat, channel="H"):
    vals = (((results.get(attr) or {}).get(f"{path}_distributions") or {}).get(cat) or {}).get(channel) or []
    return np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)


def hue_values(results, attr, path, cat):
    return channel_values(results, attr, path, cat, "H")


def pvalue(v1, v2, test, num_bins):
    """p-valeur du test KS entre deux échantillons : celui du mémoire (binned) ou sur valeurs brutes."""
    if len(v1) <= 1 or len(v2) <= 1:
        return np.nan
    if test == "binned":
        return comparison_metrics(v1, v2, num_bins)[4]
    return ks_2samp(v1, v2).pvalue


def category_pairs(results, attr, path):
    """Paires de catégories comparées : celles de l'évaluation, sinon toutes les paires."""
    pairs = (results.get(attr) or {}).get(f"bin_comparison_metrics_{path}") or {}
    if pairs:
        return list(pairs)
    cats = list((((results.get(attr) or {}).get(f"{path}_distributions")) or {}))
    return list(itertools.combinations(cats, 2))


def pair_pvalue(results, attr, path, pair, channel, test, num_bins):
    """p-valeur d'une paire de catégories sur un canal (métrique enregistrée si possible)."""
    if test == "binned":
        stored = ((results.get(attr) or {}).get(f"bin_comparison_metrics_{path}") or {}).get(pair)
        key = "ks_pval" if channel == "H" else f"ks_pval_{channel.lower()}"
        if isinstance(stored, dict) and stored.get(key) is not None:
            return stored[key]
    v1, v2 = (channel_values(results, attr, path, cat, channel) for cat in pair)
    return pvalue(v1, v2, test, num_bins)


def is_regular(results, attr, path, pair, alpha, channel, test, num_bins):
    """Critère de régularité d'une paire de catégories.

    H : teinte ET au moins un canal R, G ou B significatifs (mémoire §2.3.2).
    S ou L : le canal lui-même significatif.
    """
    if channel == "H" and test == "binned":
        m = ((results.get(attr) or {}).get(f"bin_comparison_metrics_{path}") or {}).get(pair)
        if isinstance(m, dict):
            return is_regular_pair(m, alpha)
    p = pair_pvalue(results, attr, path, pair, channel, test, num_bins)
    if not np.isfinite(p) or p >= alpha:
        return False
    if channel != "H":
        return True
    return any(pair_pvalue(results, attr, path, pair, c, test, num_bins) < alpha for c in "RGB")


def contributing_categories(results, attr, path, alpha, channel="H", test="binned", num_bins=15):
    cats = set()
    for pair in category_pairs(results, attr, path):
        if is_regular(results, attr, path, pair, alpha, channel, test, num_bins):
            cats.update(pair)
    return cats


def load_control_results(directory):
    """Distributions d'un Modèle Complet (binned_results.pkl) au format de analysis_results.pkl."""
    f = directory / "binned_results.pkl"
    if not f.exists():
        return None
    binned = pickle.load(open(f, "rb"))
    results = {}
    for path, _ in PATHS:
        for attr, dists in (binned.get(f"{path}_colors_by_attr") or {}).items():
            results.setdefault(attr, {})[f"{path}_distributions"] = dists
    return results


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
    parser.add_argument("--channel", choices=["H", "S", "L"], default="H",
                        help="Canal testé : teinte (mémoire), saturation ou luminosité (évaluation --hls)")
    parser.add_argument("--test", choices=["binned", "raw"], default="binned",
                        help="Test KS du mémoire (histogrammes de 15 classes) ou sur les valeurs brutes")
    args = parser.parse_args()
    criteria = dict(channel=args.channel, test=args.test, num_bins=args.num_bins)

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
                per_inst = {rid: contributing_categories(r, attr, path, args.alpha, **criteria)
                            for rid, r in runs.items()}
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
                if args.channel == "H" and args.test == "binned":  # filtre du script d'origine
                    compared = args.no_first_instance_filter or any(
                        isinstance(m, dict) and m.get("ks_pval") is not None and m["ks_pval"] < args.alpha
                        for p, _ in PATHS for m in (first.get(f"bin_comparison_metrics_{p}") or {}).values())
                else:  # même filtre, sur le canal et le test choisis
                    first_results = next(iter(runs.values()))
                    compared = args.no_first_instance_filter or any(
                        pair_pvalue(first_results, attr, p, pair, args.channel, args.test, args.num_bins) < args.alpha
                        for p, _ in PATHS for pair in category_pairs(first_results, attr, p))
                specific = set()
                for cat in (candidates if compared else ()):
                    for r1, r2 in itertools.combinations(runs, 2):
                        h1, h2 = (channel_values(runs[r], attr, path, cat, args.channel) for r in (r1, r2))
                        if len(h1) > 1 and len(h2) > 1 and pvalue(h1, h2, args.test, args.num_bins) < args.alpha:
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

    if args.channel != "H" or args.test != "binned":
        print(f"\nCanal {args.channel}, test KS {'du mémoire (histogrammes)' if args.test == 'binned' else 'sur valeurs brutes'}")
    print("\nTableau B.2 — % d'instances avec au moins une paire de catégories régulière")
    print("\n".join(table_b2))
    print(f"\nTableaux 3.1 / E.1 — scores d'arbitraire "
          f"(numérateur : {args.numerator}, dénominateur : {args.denominator})")
    print("\n".join(table_arb))

    # Modèles Complets (contrôles), évalués par scripts/evaluate_all.sh
    # (pour la traduction bruitée : <config>_control_shape_noise_<x> s'il existe)
    noise_suffix = args.results_dir.name[len("arbitrary"):] if args.results_dir.name.startswith("arbitrary") else ""
    control_lines = []
    for exp_name in manifest:
        control_dir = args.results_dir.parent / "regularity" / f"{exp_name}_control{noise_suffix}"
        if not control_dir.exists():
            control_dir = args.results_dir.parent / "regularity" / f"{exp_name}_control"
        for f in sorted(control_dir.glob("*/within_model_comparison_summary.json")):
            if args.channel == "H" and args.test == "binned":
                stats = json.load(open(f))["path_stats"]
                counts = [stats.get(path, {}).get("significant", 0) for path, _ in PATHS]
            else:
                results = load_control_results(f.parent)
                if results is None:
                    continue
                counts = [sum(is_regular(results, attr, path, pair, args.alpha, **criteria)
                              for attr, _ in ATTRIBUTES for pair in category_pairs(results, attr, path))
                          for path, _ in PATHS]
            control_lines.append(f"{CONFIG_LABELS.get(exp_name, exp_name):36s}" + "".join(
                f"{n:>15d}" for n in counts))
    if control_lines:
        print("\nModèles Complets (contrôles) — nombre de paires de catégories régulières")
        print(f"{'Configuration':36s}" + "".join(f"{lab:>15s}" for _, lab in PATHS))
        print("\n".join(control_lines))


if __name__ == "__main__":
    main()
