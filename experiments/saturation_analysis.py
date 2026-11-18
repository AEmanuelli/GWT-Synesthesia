"""Saturation (S) et luminosité (L) des couleurs reconstruites : niveaux, régularité, arbitraire.

À partir des évaluations lancées avec --hls (scripts/evaluate_saturation.sh) :

    python experiments/saturation_analysis.py

Aperçu à partir des résultats du chapitre 3 (sans --hls) : S et L y sont recalculées à partir des
canaux R, G, B enregistrés, arrondis à l'entier inférieur (écart sur S de l'ordre de 0,01) :

    python experiments/saturation_analysis.py --results-dir results --from-rgb --out results/saturation/apercu

Chemins comparés : demi-cycle (témoin : la couleur de l'image d'entrée est disponible), traduction,
traduction bruitée (annexe E, si l'évaluation avec --shape-noise a été faite) et cycle complet.
S et L sont sur l'échelle 0-255 dans les résultats (comme R, G, B) et affichées sur 0-1 ici.

Trois critères pour dire que deux distributions de S diffèrent :
  - memoire : test KS du mémoire (histogrammes de 15 classes, voir comparison_metrics), p < alpha.
    Son effectif ne dépend que de l'étendue des valeurs : avec S sur 0-255, il est bien plus
    sensible que pour la teinte (0-360) ;
  - brut : test KS sur les valeurs brutes, p < alpha (avec 9 600 à 51 200 images par catégorie,
    le moindre écart est significatif) ;
  - effet : test brut significatif ET taille d'effet au moins « petite » (|delta de Cliff| >= 0,147).
Régularité : au moins une paire de catégories d'un attribut diffère dans l'instance.
Arbitraire : une catégorie est spécifique si sa distribution diffère entre au moins deux instances
(numérateur : toutes les catégories de l'attribut ; dénominateur : catégories qui ont contribué
à la régularité dans au moins une instance, comme experiments/make_tables.py ; sans le filtre
« première instance »).

Sorties dans --out : niveaux_S_L.csv, paires_categories.csv, paires_instances.csv,
synthese.md (tableaux) et figures/.
"""

import argparse
import colorsys
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ks_2samp, mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
from SSD_utils import comparison_metrics  # noqa: E402  (test du mémoire)

ATTRIBUTES = [("shape", "Forme"), ("size", "Taille"), ("position_x", "Pos X"),
              ("position_y", "Pos Y"), ("rotation", "Orientation")]
PATHS = [("half_cycle", "Demi-Cycle"), ("translated", "Traduction"),
         ("translated_noisy", "Trad. Bruitée"), ("full_cycle", "Cycle Complet")]
CONFIG_LABELS = {"base_params": "Référence", "high_translation": "Emphase Traduction",
                 "high_contrastive": "Emphase Contrastive", "high_cycles": "Emphase Cycle-Complet",
                 "high_alpha": "Emphase Attributs Continus"}
CHANNELS = ("H", "S", "L")
CRITERIA = ("memoire", "brut", "effet")
SMALL_EFFECT = 0.147  # |delta de Cliff| : seuil d'un effet « petit » (Romano et al., 2006)


# ----------------------------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------------------------
def _sl_from_rgb(dist):
    """S et L (échelle 0-255) recalculées depuis les canaux R, G, B enregistrés (entiers)."""
    hls = np.array([colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
                    for r, g, b in zip(dist["R"], dist["G"], dist["B"])]).reshape(-1, 3)
    return hls[:, 2] * 255, hls[:, 1] * 255


def _channels(dist, from_rgb):
    out = {"H": np.asarray(dist.get("H", []), dtype=np.float32)}
    if "S" in dist and "L" in dist:
        out["S"] = np.asarray(dist["S"], dtype=np.float32)
        out["L"] = np.asarray(dist["L"], dtype=np.float32)
    elif from_rgb and "R" in dist:
        s, l = _sl_from_rgb(dist)
        out["S"], out["L"] = s.astype(np.float32), l.astype(np.float32)
    else:
        return None
    return out


def _distributions(pkl_file, from_rgb):
    """{chemin: {attribut: {catégorie: {canal: array}}}} depuis analysis_results.pkl ou binned_results.pkl."""
    data = pickle.load(open(pkl_file, "rb"))
    out = {}
    for path in ("input", "translated", "half_cycle", "full_cycle"):
        for attr, _ in ATTRIBUTES:
            if pkl_file.name == "binned_results.pkl":
                dists = (data.get(f"{path}_colors_by_attr") or {}).get(attr) or {}
            else:
                dists = (data.get(attr) or {}).get(f"{path}_distributions") or {}
            for cat, dist in dists.items():
                ch = _channels(dist, from_rgb)
                if ch is None:
                    raise SystemExit(f"Pas de S/L dans {pkl_file} : évaluation sans --hls (utiliser --from-rgb).")
                out.setdefault(path, {}).setdefault(attr, {})[cat] = ch
    return out


def load_models(results_dir, from_rgb, cache_file):
    """Modèles « vide sémantique » et contrôles, chemins clean + bruité. Mis en cache dans cache_file."""
    if cache_file.exists():
        print(f"Cache : {cache_file}")
        return pickle.load(open(cache_file, "rb"))
    manifest = yaml.safe_load(open(REPO_ROOT / "experiments" / "thesis_runs.yaml"))["experiments"]
    models = {}
    for noise_dir, noisy in (("arbitrary", False), ("arbitrary_shape_noise_0.01", True)):
        for exp_name, exp in manifest.items():
            entries = [(run["run_id"], results_dir / noise_dir / exp_name /
                        f"results_{exp_name}_sans_couleurs_{run['run_id']}" / "analysis_results.pkl")
                       for run in exp["runs"]]
            control_name = f"{exp_name}_control" + ("_shape_noise_0.01" if noisy else "")
            entries.append(("control", results_dir / "regularity" / control_name /
                            "results_avec_couleurs_seed0" / "binned_results.pkl"))
            for run_id, f in entries:
                if not f.exists():
                    continue
                print(f"Lecture : {f}")
                dists = _distributions(f, from_rgb)
                model = models.setdefault((exp_name, run_id), {})
                if noisy:
                    model["translated_noisy"] = dists["translated"]
                else:
                    model.update(dists)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(models, open(cache_file, "wb"))
    return models


# ----------------------------------------------------------------------------------------------
# Statistiques
# ----------------------------------------------------------------------------------------------
def compare(v1, v2, channel, num_bins):
    """Test du mémoire, KS brut, écart des médianes (S et L sur 0-1, H en degrés) et delta de Cliff."""
    v1, v2 = v1[np.isfinite(v1)], v2[np.isfinite(v2)]
    if len(v1) < 2 or len(v2) < 2:
        return None
    p_memoire = comparison_metrics(v1, v2, num_bins)[4]
    raw = ks_2samp(v1, v2, method="asymp")
    u = mannwhitneyu(v1, v2, method="asymptotic").statistic
    scale = 1.0 if channel == "H" else 255.0
    return {"p_memoire": float(p_memoire), "D_brut": float(raw.statistic), "p_brut": float(raw.pvalue),
            "delta_mediane": float((np.median(v1) - np.median(v2)) / scale),
            "cliff_delta": float(2 * u / (len(v1) * len(v2)) - 1)}


def differs(stats, criterion, alpha):
    if stats is None:
        return False
    if criterion == "memoire":
        return stats["p_memoire"] < alpha
    if criterion == "brut":
        return stats["p_brut"] < alpha
    return stats["p_brut"] < alpha and abs(stats["cliff_delta"]) >= SMALL_EFFECT


def category_pair_stats(models, channels, num_bins):
    rows = []
    for (exp_name, run_id), model in sorted(models.items()):
        for path, _ in PATHS:
            for attr, _ in ATTRIBUTES:
                cats = (model.get(path) or {}).get(attr) or {}
                for c1, c2 in itertools.combinations(cats, 2):
                    for ch in channels:
                        st = compare(cats[c1][ch], cats[c2][ch], ch, num_bins)
                        if st:
                            rows.append(dict(config=exp_name, instance=run_id, chemin=path, attribut=attr,
                                             categorie_1=c1, categorie_2=c2, canal=ch, **st))
    return rows


def instance_pair_stats(models, channels, num_bins):
    rows = []
    by_exp = {}
    for (exp_name, run_id), model in models.items():
        if run_id != "control":
            by_exp.setdefault(exp_name, {})[run_id] = model
    for exp_name, runs in sorted(by_exp.items()):
        for path, _ in PATHS:
            for attr, _ in ATTRIBUTES:
                for r1, r2 in itertools.combinations(sorted(runs), 2):
                    cats1 = (runs[r1].get(path) or {}).get(attr) or {}
                    cats2 = (runs[r2].get(path) or {}).get(attr) or {}
                    for cat in cats1:
                        if cat not in cats2:
                            continue
                        for ch in channels:
                            st = compare(cats1[cat][ch], cats2[cat][ch], ch, num_bins)
                            if st:
                                rows.append(dict(config=exp_name, instance_1=r1, instance_2=r2, chemin=path,
                                                 attribut=attr, categorie=cat, canal=ch, **st))
    return rows


def level_rows(models):
    rows = []
    for (exp_name, run_id), model in sorted(models.items()):
        for path in ["input"] + [p for p, _ in PATHS]:
            cats = (model.get(path) or {}).get("shape") or {}
            for cat, ch in cats.items():
                for channel in ("S", "L"):
                    v = ch[channel] / 255.0
                    rows.append(dict(config=exp_name, instance=run_id, chemin=path, forme=cat, canal=channel,
                                     n=len(v), mediane=float(np.median(v)), q05=float(np.quantile(v, 0.05)),
                                     q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)),
                                     q95=float(np.quantile(v, 0.95))))
    return rows


# ----------------------------------------------------------------------------------------------
# Tableaux
# ----------------------------------------------------------------------------------------------
def write_csv(rows, f):
    import csv
    if not rows:
        return
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def level_table(levels, channel):
    """Médiane de S (ou L) par configuration et chemin (instances regroupées, 3 formes)."""
    header = ["Configuration", "Modèles"] + ["Entrée"] + [lab for _, lab in PATHS]
    rows = []
    groups = sorted({(r["config"], r["instance"] == "control") for r in levels})
    for exp_name, is_control in groups:
        sel = [r for r in levels if r["config"] == exp_name and (r["instance"] == "control") == is_control
               and r["canal"] == channel]
        row = [CONFIG_LABELS.get(exp_name, exp_name), "contrôle" if is_control else "vide sém."]
        for path in ["input"] + [p for p, _ in PATHS]:
            meds = [r["mediane"] for r in sel if r["chemin"] == path]
            row.append(f"{np.median(meds):.2f}" if meds else "")
        rows.append(row)
    return md_table(header, rows)


def regularity_tables(cat_rows, inst_rows, channel, criterion, alpha):
    """Tableau B.2 (% d'instances régulières) et 3.1 (arbitraire) pour un canal et un critère."""
    header = ["Configuration", "Chemin"] + [lab for _, lab in ATTRIBUTES]
    b2, arb, ctrl = [], [], []
    configs = sorted({r["config"] for r in cat_rows}, key=list(CONFIG_LABELS).index)
    for exp_name in configs:
        for path, path_label in PATHS:
            sel = [r for r in cat_rows if r["config"] == exp_name and r["chemin"] == path and r["canal"] == channel]
            if not sel:
                continue
            instances = sorted({r["instance"] for r in sel if r["instance"] != "control"})
            row_b2 = [CONFIG_LABELS[exp_name], path_label]
            row_arb = [CONFIG_LABELS[exp_name], path_label]
            row_ctrl = [CONFIG_LABELS[exp_name], path_label]
            for attr, _ in ATTRIBUTES:
                contrib = {}
                for inst in instances:
                    contrib[inst] = set()
                    for r in sel:
                        if r["instance"] == inst and r["attribut"] == attr and differs(r, criterion, alpha):
                            contrib[inst].update((r["categorie_1"], r["categorie_2"]))
                n_reg = sum(1 for c in contrib.values() if c)
                row_b2.append(f"{100 * n_reg / len(instances):.0f}" if instances else "")
                union = set().union(*contrib.values()) if contrib else set()
                if not union:
                    row_arb.append("-")
                else:
                    specific = {r["categorie"] for r in inst_rows
                                if r["config"] == exp_name and r["chemin"] == path and r["canal"] == channel
                                and r["attribut"] == attr and differs(r, criterion, alpha)}
                    row_arb.append(f"{len(specific)}/{len(union)}")
                n_ctrl = sum(1 for r in sel if r["instance"] == "control" and r["attribut"] == attr
                             and differs(r, criterion, alpha))
                n_pairs = sum(1 for r in sel if r["instance"] == "control" and r["attribut"] == attr)
                row_ctrl.append(f"{n_ctrl}/{n_pairs}" if n_pairs else "")
            b2.append(row_b2)
            arb.append(row_arb)
            if any(c for c in row_ctrl[2:]):
                ctrl.append(row_ctrl)
    return md_table(header, b2), md_table(header, arb), md_table(header, ctrl) if ctrl else "(aucun contrôle)"


def effect_summary(cat_rows, inst_rows, channel):
    """Plus grand |delta de Cliff| et |écart de médianes| par configuration et chemin."""
    header = ["Configuration", "Modèles", "Chemin", "Forme : max |δ| entre formes", "Forme : max |Δméd.|",
              "Orientation : max |δ|", "Forme : max |δ| entre instances"]
    rows = []
    groups = sorted({(r["config"], r["instance"] == "control") for r in cat_rows},
                    key=lambda g: (list(CONFIG_LABELS).index(g[0]), g[1]))
    for exp_name, is_control in groups:
        for path, path_label in PATHS:
            sel = [r for r in cat_rows if r["config"] == exp_name and (r["instance"] == "control") == is_control
                   and r["chemin"] == path and r["canal"] == channel]
            if not sel:
                continue
            shape = [r for r in sel if r["attribut"] == "shape"]
            rot = [r for r in sel if r["attribut"] == "rotation"]
            inst = [r for r in inst_rows if r["config"] == exp_name and r["chemin"] == path
                    and r["canal"] == channel and r["attribut"] == "shape"] if not is_control else []
            unit = "°" if channel == "H" else ""
            rows.append([CONFIG_LABELS[exp_name], "contrôle" if is_control else "vide sém.", path_label,
                         f"{max(abs(r['cliff_delta']) for r in shape):.2f}" if shape else "",
                         f"{max(abs(r['delta_mediane']) for r in shape):.3f}{unit}" if shape else "",
                         f"{max(abs(r['cliff_delta']) for r in rot):.2f}" if rot else "",
                         f"{max(abs(r['cliff_delta']) for r in inst):.2f}" if inst else ""])
    return md_table(header, rows)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def figures(models, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = ["diamond", "egg", "triangle"]
    configs = [c for c in CONFIG_LABELS if any(k[0] == c for k in models)]

    # 1. Niveau de saturation par chemin
    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 4), sharey=True, squeeze=False)
    for ax, exp_name in zip(axes[0], configs):
        keys = sorted(k for k in models if k[0] == exp_name)
        positions, data, labels, colors = [], [], [], []
        for i, (path, label) in enumerate(PATHS):
            for j, key in enumerate(keys):
                cats = (models[key].get(path) or {}).get("shape")
                if not cats:
                    continue
                v = np.concatenate([cats[s]["S"] for s in shapes if s in cats]) / 255
                positions.append(i * (len(keys) + 1) + j)
                data.append(v[::10])
                colors.append("grey" if key[1] == "control" else f"C{j}")
            labels.append((i * (len(keys) + 1) + (len(keys) - 1) / 2, label))
        if data:
            parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.9)
            for body, c in zip(parts["bodies"], colors):
                body.set_facecolor(c)
                body.set_alpha(0.6)
        ax.axhline(0.907, color="k", ls=":", lw=1)
        ax.set_xticks([p for p, _ in labels])
        ax.set_xticklabels([l for _, l in labels], rotation=30, ha="right")
        ax.set_title(CONFIG_LABELS[exp_name], fontsize=10)
    axes[0][0].set_ylabel("Saturation S (barycentre)")
    fig.suptitle("Saturation reconstruite par chemin (couleurs : instances ; gris : contrôle ; pointillés : entrée)")
    fig.tight_layout()
    fig.savefig(out_dir / "S_par_chemin.png", dpi=120)
    plt.close(fig)

    # 2. S par forme et par instance, traduction et traduction bruitée
    for exp_name in configs:
        keys = sorted(k for k in models if k[0] == exp_name)
        paths_here = [(p, l) for p, l in PATHS if p in ("translated", "translated_noisy", "full_cycle")
                      and any(models[k].get(p) for k in keys)]
        fig, axes = plt.subplots(len(paths_here), len(keys), figsize=(3.2 * len(keys), 2.6 * len(paths_here)),
                                 squeeze=False, sharex="row")
        for i, (path, label) in enumerate(paths_here):
            for j, key in enumerate(keys):
                ax = axes[i][j]
                cats = (models[key].get(path) or {}).get("shape") or {}
                allv = np.concatenate([cats[s]["S"] for s in shapes if s in cats]) / 255 if cats else np.array([0, 1])
                lo, hi = np.quantile(allv, [0.005, 0.995]) if len(allv) > 2 else (0, 1)
                for s in shapes:
                    if s in cats:
                        ax.hist(cats[s]["S"] / 255, bins=60, range=(lo, hi), histtype="step", density=True, label=s)
                ax.set_title(f"{'contrôle' if key[1] == 'control' else key[1]} — {label}", fontsize=9)
                if i == len(paths_here) - 1:
                    ax.set_xlabel("S")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{CONFIG_LABELS[exp_name]} : saturation par forme")
        fig.tight_layout()
        fig.savefig(out_dir / f"S_par_forme_{exp_name}.png", dpi=110)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "saturation",
                        help="Dossier contenant arbitrary/, arbitrary_shape_noise_0.01/ et regularity/")
    parser.add_argument("--out", type=Path, default=None, help="défaut : <results-dir>/analysis")
    parser.add_argument("--from-rgb", action="store_true",
                        help="Recalculer S et L depuis R, G, B si elles n'ont pas été extraites (aperçu)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-bins", type=int, default=15)
    args = parser.parse_args()
    out = args.out or args.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    models = load_models(args.results_dir, args.from_rgb, out / "cache_distributions.pkl")
    if not models:
        raise SystemExit(f"Aucun résultat dans {args.results_dir}")
    print(f"{len(models)} modèles : {sorted(models)}")

    levels = level_rows(models)
    write_csv(levels, out / "niveaux_S_L.csv")
    cat_rows = category_pair_stats(models, CHANNELS, args.num_bins)
    write_csv(cat_rows, out / "paires_categories.csv")
    inst_rows = instance_pair_stats(models, CHANNELS, args.num_bins)
    write_csv(inst_rows, out / "paires_instances.csv")

    md = [f"# Saturation et luminosité des couleurs reconstruites\n",
          f"Source : `{args.results_dir}`" + (" (S et L recalculées depuis R, G, B entiers)" if args.from_rgb else ""),
          f"\nSeuil alpha = {args.alpha} ; effet « petit » : |δ de Cliff| ≥ {SMALL_EFFECT}.\n",
          "## Niveau de saturation (médiane de S sur 0-1 ; entrée : 0,907)\n", level_table(levels, "S"),
          "\n## Niveau de luminosité (médiane de L sur 0-1 ; entrée : 0,706)\n", level_table(levels, "L")]
    for channel in ("S", "L", "H"):
        md.append(f"\n## Canal {channel} : tailles d'effet\n")
        md.append(effect_summary(cat_rows, inst_rows, channel))
        for criterion in CRITERIA:
            b2, arb, ctrl = regularity_tables(cat_rows, inst_rows, channel, criterion, args.alpha)
            md += [f"\n### Canal {channel}, critère « {criterion} »\n",
                   "% d'instances avec au moins une paire de catégories qui diffère (régularité) :\n", b2,
                   "\nArbitraire (catégories spécifiques / catégories ayant contribué à la régularité) :\n", arb,
                   "\nModèles Complets : paires de catégories qui diffèrent / paires testées :\n", ctrl]
    (out / "synthese.md").write_text("\n".join(md) + "\n")
    figures(models, out / "figures")
    print(f"Résultats dans {out}")


if __name__ == "__main__":
    main()
