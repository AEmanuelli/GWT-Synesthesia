"""Robustesse des tests (étape 4 du protocole « saturation ») : comparaisons multiples.

Relit les tableaux de paires écrits par saturation_analysis.py (paires_categories.csv : catégories
d'un attribut au sein d'une instance -> régularité ; paires_instances.csv : une même catégorie entre
instances -> arbitraire) et compare, pour chaque chemin, canal (H, S, L) et attribut :
  - test du mémoire (histogrammes), KS brut, critère d'effet (KS brut et |delta de Cliff| >= 0,147) ;
  - les mêmes après correction pour comparaisons multiples, Holm (contrôle du risque global) et
    Benjamini-Hochberg (taux de fausses découvertes). Famille = toutes les paires d'un même tableau,
    chemin et canal (toutes configurations, instances et attributs confondus) : correction prudente.
Pour la régularité : part des instances ayant au moins une paire qui diffère et part des paires ;
pour l'arbitraire : part des paires (instances, catégorie) qui diffèrent.

    python experiments/multiple_testing.py
    python experiments/multiple_testing.py --pairs-dir results/saturation/apercu_chapitre3 --out ...

Sorties : <out>/tests_<source>.csv et tests_<source>.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
from saturation_analysis import CONFIG_LABELS, SMALL_EFFECT  # noqa: E402

ALPHA = 0.05
CRITERIA = ["memoire", "memoire_holm", "memoire_bh", "brut", "brut_holm", "brut_bh",
            "effet", "effet_holm", "effet_bh"]
CRITERIA_LABELS = {"memoire": "mémoire", "memoire_holm": "mémoire + Holm", "memoire_bh": "mémoire + BH",
                   "brut": "KS", "brut_holm": "KS + Holm", "brut_bh": "KS + BH",
                   "effet": "effet", "effet_holm": "effet + Holm", "effet_bh": "effet + BH"}
ATTR_LABELS = {"shape": "Forme", "size": "Taille", "position_x": "Pos X", "position_y": "Pos Y",
               "rotation": "Orientation"}
PATH_LABELS = {"translated": "Traduction", "translated_noisy": "Trad. Bruitée", "half_cycle": "Demi-Cycle",
               "full_cycle": "Cycle Complet"}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


def flag(df):
    """Ajoute une colonne booléenne par critère, corrections calculées par famille (chemin, canal)."""
    df = df.copy()
    for col, name in (("p_memoire", "memoire"), ("p_brut", "brut")):
        df[f"{name}_holm_p"] = df.groupby(["chemin", "canal"])[col].transform(holm)
        df[f"{name}_bh_p"] = df.groupby(["chemin", "canal"])[col].transform(bh)
    big = df["cliff_delta"].abs() >= SMALL_EFFECT
    df["memoire"] = df["p_memoire"] < ALPHA
    df["memoire_holm"] = df["memoire_holm_p"] < ALPHA
    df["memoire_bh"] = df["memoire_bh_p"] < ALPHA
    df["brut"] = df["p_brut"] < ALPHA
    df["brut_holm"] = df["brut_holm_p"] < ALPHA
    df["brut_bh"] = df["brut_bh_p"] < ALPHA
    df["effet"] = df["brut"] & big
    df["effet_holm"] = df["brut_holm"] & big
    df["effet_bh"] = df["brut_bh"] & big
    return df


def group_of(instance):
    return "contrôles" if instance == "control" else "vide sémantique"


def regularity(cat):
    cat = cat.assign(groupe=[f"{CONFIG_LABELS[c]}" if i != "control" else "Modèles Complets (contrôles)"
                             for c, i in zip(cat.config, cat.instance)])
    rows = []
    for (path, ch, attr, g), x in cat.groupby(["chemin", "canal", "attribut", "groupe"]):
        per_inst = x.groupby(["config", "instance"])[CRITERIA].any()
        rows.append(dict(tableau="régularité", chemin=path, canal=ch, attribut=attr, groupe=g,
                         n_instances=len(per_inst), n_paires=len(x),
                         **{f"inst_{c}": per_inst[c].mean() for c in CRITERIA},
                         **{f"paires_{c}": x[c].mean() for c in CRITERIA}))
    return pd.DataFrame(rows)


def arbitrariness(inst):
    rows = []
    for (path, ch, attr, conf), x in inst.groupby(["chemin", "canal", "attribut", "config"]):
        rows.append(dict(tableau="arbitraire", chemin=path, canal=ch, attribut=attr, groupe=CONFIG_LABELS[conf],
                         n_paires=len(x), **{f"paires_{c}": x[c].mean() for c in CRITERIA}))
    return pd.DataFrame(rows)


def pct(x):
    return f"{100 * x:.0f}"


def write(reg, arb, out_dir, source):
    order = list(CONFIG_LABELS.values()) + ["Modèles Complets (contrôles)"]
    shown = ["memoire", "memoire_holm", "memoire_bh", "brut", "brut_bh", "effet", "effet_bh"]
    L = [f"# Comparaisons multiples : {source}\n",
         "Familles de correction : toutes les paires d'un tableau pour un chemin et un canal "
         f"(α = {ALPHA}). « effet » = KS brut significatif et |δ de Cliff| ≥ {SMALL_EFFECT}. "
         "Valeurs en % ; régularité : % d'instances avec au moins une paire de catégories qui diffère "
         "(entre parenthèses : % de paires) ; arbitraire : % de paires (instances, catégorie) qui diffèrent.\n"]
    for path in reg.chemin.unique():
        for ch in ("H", "S", "L"):
            L.append(f"## {PATH_LABELS.get(path, path)}, canal {ch}\n")
            for attr in ATTR_LABELS:
                r = reg[(reg.chemin == path) & (reg.canal == ch) & (reg.attribut == attr)]
                a = arb[(arb.chemin == path) & (arb.canal == ch) & (arb.attribut == attr)]
                if r.empty:
                    continue
                L.append(f"**{ATTR_LABELS[attr]}**\n")
                L.append("| groupe | " + " | ".join(CRITERIA_LABELS[c] for c in shown) + " |")
                L.append("|---|" + "---|" * len(shown))
                for g in order:
                    x = r[r.groupe == g]
                    if not x.empty:
                        x = x.iloc[0]
                        L.append(f"| régularité, {g} | " + " | ".join(
                            f"{pct(x[f'inst_{c}'])} ({pct(x[f'paires_{c}'])})" for c in shown) + " |")
                for g in order:
                    x = a[a.groupe == g]
                    if not x.empty:
                        x = x.iloc[0]
                        L.append(f"| arbitraire, {g} | " + " | ".join(pct(x[f"paires_{c}"]) for c in shown) + " |")
                L.append("")
    (out_dir / f"tests_{source}.md").write_text("\n".join(L))
    pd.concat([reg, arb]).to_csv(out_dir / f"tests_{source}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", type=Path, action="append",
                        help="Dossier(s) contenant paires_categories.csv et paires_instances.csv")
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "saturation" / "robustesse" / "tests")
    args = parser.parse_args()
    dirs = args.pairs_dir or [paths.RESULTS_DIR / "saturation" / "traduction_seule" / "analysis",
                              paths.RESULTS_DIR / "saturation" / "apercu_chapitre3"]
    args.out.mkdir(parents=True, exist_ok=True)
    for d in dirs:
        source = d.name if d.name != "analysis" else d.parent.name
        cat = flag(pd.read_csv(d / "paires_categories.csv"))
        inst = flag(pd.read_csv(d / "paires_instances.csv"))
        write(regularity(cat), arbitrariness(inst), args.out, source)
        print(f"Écrit : {args.out / f'tests_{source}.md'}")


if __name__ == "__main__":
    main()
