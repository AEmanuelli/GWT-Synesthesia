"""Associations forme → couleur : tiennent-elles quelle que soit la géométrie ? reviennent-elles d'un modèle à l'autre ?

Données : traduction avec le résidu fixe de 1 % (graine 42), évaluation `--translation-only` (fichiers
binned_results.pkl). Pour chaque modèle et chaque forme, l'évaluation garde la couleur des 51 200
traductions de la forme dans l'ordre du jeu d'évaluation ; on retrouve ainsi l'orientation, la taille
et la position de chaque traduction (correspondance vérifiée sur les autres regroupements du fichier).
Les 25 copies d'une même géométrie (une par teinte d'entrée, absente des attributs) donnent la même
traduction.

  1. Invariance : pour chaque modèle et chaque forme colorée, la teinte dominante est-elle la même pour
     chaque orientation (16), chaque taille (8) et chaque position (16) ? Part des niveaux dont la teinte
     dominante est à ±30° de celle de la forme ; constance (part des traductions à ±30°) par niveau.
  2. Récurrence : d'un modèle à l'autre, une forme prend-elle la même teinte ? Teintes dominantes des
     modèles où la forme est colorée, test de Rayleigh d'uniformité, et regroupement par secteur.

    python experiments/association_recurrence.py
Sorties : results/publication/recurrence/ (synthese.md, recurrence.csv, figures).
"""
import argparse
import ast
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
from shape_noise_sweep import CONFIG_LABELS, SHAPES, SHAPE_LABELS, md_table  # noqa: E402

S_COLOURED = 0.3
TOL = 30.0
MIN_COLOURED = 0.5      # forme « colorée » : au moins la moitié des traductions avec S ≥ 0,3
CONFIG_ORDER = ["base_params", "high_cycles", "high_alpha", "high_translation", "high_contrastive"]
FOCUS = ("base_params", "high_cycles")


def circ_mean(h, weights=None):
    z = np.exp(1j * np.radians(h))
    z = np.average(z, weights=weights) if weights is not None else z.mean()
    return float(np.degrees(np.angle(z)) % 360), float(np.abs(z))


def dist(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def rayleigh(h):
    """Test de Rayleigh (H0 : teintes uniformes sur le cercle), approximation de Zar."""
    n = len(h)
    if n < 3:
        return np.nan
    r = circ_mean(h)[1]
    z = n * r ** 2
    return float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * r) ** 2)) - (1 + 2 * n)))


def dataset_factors():
    rows = list(csv.DictReader(open(paths.EVALSET_DIR / "attributes.csv")))
    cls = np.array([int(r["class"]) for r in rows])
    size = np.array([int(r["size"]) for r in rows])
    rot = np.array([int(round(float(r["rotation"]) / (2 * np.pi / 16))) % 16 for r in rows])
    loc = [tuple(ast.literal_eval(r["location"])) for r in rows]
    locs = sorted(set(loc))
    pos = np.array([locs.index(p) for p in loc])
    return cls, size, rot, pos


def per_image(data, factors):
    """Couleur (H, S, RGB) de chaque image du jeu, reconstituée depuis les regroupements par forme ;
    vérifiée sur les regroupements par orientation et par taille."""
    cls, size, rot, _ = factors
    t = data["translated_colors_by_attr"]
    n = len(cls)
    H, S = np.full(n, np.nan), np.full(n, np.nan)
    RGB = np.full((n, 3), np.nan)
    for s, name in enumerate(SHAPES):
        idx = np.where(cls == s)[0]
        pool = t["shape"][name]
        H[idx] = pool["H"]
        S[idx] = pool["S"]
        RGB[idx] = np.column_stack([pool["R"], pool["G"], pool["B"]])
    ok = True
    for r, cat in enumerate(sorted(t["rotation"], key=lambda c: int(c.split("x")[0]))):
        idx = np.where(rot == r)[0]
        ok &= np.allclose(H[idx], t["rotation"][cat]["H"], atol=1e-3)
    sizes = sorted(set(size))
    for k, cat in enumerate(sorted(t["size"], key=int)):
        idx = np.where(size == sizes[k])[0]
        ok &= np.allclose(H[idx], t["size"][cat]["H"], atol=1e-3)
    S = S / 255 if np.nanmax(S) > 1.5 else S
    return H, S, RGB, bool(ok)


def analyze(models, factors, out_dir):
    cls, size, rot, pos = factors
    rows, verified = [], []
    for exp_name, instance, f in models:
        data = pickle.load(open(f, "rb"))
        H, S, RGB, ok = per_image(data, factors)
        verified.append(ok)
        del data
        for s in range(3):
            sel = cls == s
            col = sel & (S >= S_COLOURED)
            frac_col = col.sum() / sel.sum()
            row = dict(config=exp_name, instance=instance, forme=s, colorees=frac_col)
            if frac_col < MIN_COLOURED:
                rows.append(row)
                continue
            dom, _ = circ_mean(H[col])
            row.update(teinte=dom, constance=float(np.mean(col[sel] & (dist(H[sel], dom) <= TOL))),
                       rgb=np.median(RGB[col & (dist(H, dom) <= TOL)], axis=0))
            for name, fac, levels in (("orientation", rot, range(16)), ("taille", size, sorted(set(size))),
                                      ("position", pos, range(16))):
                hues, cons, swatch = [], [], []
                for lv in levels:
                    lsel = sel & (fac == lv)
                    lcol = lsel & (S >= S_COLOURED)
                    if lcol.sum() < 10:
                        hues.append(np.nan)
                        cons.append(0.0)
                        swatch.append([128, 128, 128])
                        continue
                    hues.append(circ_mean(H[lcol])[0])
                    cons.append(float(np.mean(lcol[lsel] & (dist(H[lsel], dom) <= TOL))))
                    swatch.append(np.median(RGB[lcol], axis=0))
                hues = np.array(hues)
                row[f"{name}_meme_teinte"] = float(np.mean(dist(hues, dom) <= TOL))   # NaN compte comme non
                row[f"{name}_constance_min"] = float(np.min(cons))
                row[f"{name}_constance_max"] = float(np.max(cons))
                row[f"{name}_nuancier"] = np.array(swatch)
            rows.append(row)
    return rows, all(verified)


def fr(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")


def who(r):
    return f"{CONFIG_LABELS.get(r['config'], r['config'])} g{r['instance'][-1]}"


def colour_name(h):
    names = [(15, "rouge"), (45, "orange"), (70, "jaune"), (95, "vert-jaune"), (150, "vert"), (175, "turquoise"),
             (200, "cyan"), (230, "bleu clair"), (260, "bleu"), (290, "violet"), (330, "magenta"), (360, "rouge")]
    return next(n for lim, n in names if h < lim)


def write_outputs(rows, verified, out_dir):
    assoc = [r for r in rows if "teinte" in r]
    L = ["# Associations forme → couleur : géométrie et récurrence", "",
         f"Traduction avec le résidu fixe de 1 % (graine 42), 51 200 traductions par forme et par modèle "
         f"(2 048 géométries × 25 copies). Forme colorée : au moins {100 * MIN_COLOURED:.0f} % des traductions "
         f"avec S ≥ {fr(S_COLOURED, 1)}. Correspondance traduction → géométrie vérifiée : "
         f"{'oui' if verified else 'NON'}.", "",
         "## 1. La couleur tient-elle quelle que soit la géométrie ?", "",
         "Part des 16 orientations, des 8 tailles et des 16 positions dont la teinte dominante est à ±30° de "
         "celle de la forme ; entre parenthèses, constance (part des traductions à ±30°) du pire et du meilleur "
         "niveau.", ""]
    table = []
    for r in sorted(assoc, key=lambda r: -r["constance"]):
        table.append([who(r), SHAPE_LABELS[r["forme"]], f"{colour_name(r['teinte'])} ({r['teinte']:.0f}°)",
                      f"{100 * r['constance']:.0f} %"] +
                     [f"{100 * r[f'{n}_meme_teinte']:.0f} % ({100 * r[f'{n}_constance_min']:.0f}–"
                      f"{100 * r[f'{n}_constance_max']:.0f} %)" for n in ("orientation", "taille", "position")])
    L.append(md_table(["Modèle", "Forme", "Teinte dominante", "Constance", "Orientations", "Tailles",
                       "Positions"], table))
    L += ["", "## 2. Une forme prend-elle la même teinte d'un modèle à l'autre ?", "",
          "Teintes dominantes des modèles où la forme est colorée ; test de Rayleigh (H0 : teintes uniformes) ; "
          "R̄ : concentration des teintes entre modèles (1 : toutes identiques).", ""]
    table = []
    for scope, cfgs in (("6 modèles retenus", FOCUS), ("15 modèles", tuple(CONFIG_ORDER))):
        for s in range(3):
            hs = [(r["teinte"], who(r)) for r in assoc if r["forme"] == s and r["config"] in cfgs]
            n_all = len([r for r in rows if r["forme"] == s and r["config"] in cfgs])
            if not hs:
                table.append([scope, SHAPE_LABELS[s], f"0/{n_all}", "–", "–", "–"])
                continue
            h = np.array([x for x, _ in hs])
            mu, rb = circ_mean(h)
            table.append([scope, SHAPE_LABELS[s], f"{len(hs)}/{n_all}",
                          " ; ".join(f"{colour_name(x)} {x:.0f}°" for x, _ in sorted(hs)),
                          f"{fr(rb)} (moyenne {mu:.0f}°)", fr(rayleigh(h), 3)])
    L.append(md_table(["Modèles", "Forme", "Colorée dans", "Teintes dominantes", "R̄ entre modèles", "p (Rayleigh)"],
                      table))
    (out_dir / "synthese.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(out_dir / "recurrence.csv", "w", newline="", encoding="utf-8") as fh:
        keys = [k for k in rows[0].keys()] + [k for r in rows for k in r if k not in rows[0]]
        keys = [k for k in dict.fromkeys(keys) if not k.endswith("nuancier") and k != "rgb"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    swatch_figure(assoc, out_dir / "associations_par_geometrie.png")
    wheel_figure(rows, out_dir / "teintes_entre_modeles.png")


def swatch_figure(assoc, out_file, top=None):
    """Une ligne par association colorée : couleur produite pour chaque orientation, taille et position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assoc = sorted(assoc, key=lambda r: (CONFIG_ORDER.index(r["config"]), r["instance"], r["forme"]))
    assoc = [r for r in assoc if r["config"] in FOCUS] if top is None else assoc[:top]
    n = len(assoc)
    fig, axes = plt.subplots(n, 3, figsize=(11, 0.42 * n + 1.0), squeeze=False,
                             gridspec_kw={"width_ratios": [16, 8, 16]})
    for i, r in enumerate(assoc):
        for j, name in enumerate(("orientation", "taille", "position")):
            ax = axes[i][j]
            ax.imshow(r[f"{name}_nuancier"][None].astype(float) / 255, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title({"orientation": "16 orientations", "taille": "8 tailles",
                              "position": "16 positions"}[name], fontsize=9)
        axes[i][0].set_ylabel(f"{who(r)} · {SHAPE_LABELS[r['forme']]} · {100 * r['constance']:.0f} %",
                              rotation=0, ha="right", va="center", fontsize=7.5)
    fig.suptitle("Couleur produite selon la géométrie (traduction avec un résidu fixe de 1 %) : "
                 "une case = médiane des traductions colorées de ce niveau", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def wheel_figure(rows, out_file):
    """Teinte dominante de chaque modèle, par forme, sur le cercle des teintes (taille : constance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "polar"}, figsize=(10, 3.8))
    markers = {"base_params": "o", "high_cycles": "D", "high_alpha": "s", "high_translation": "^",
               "high_contrastive": "v"}
    for s, ax in enumerate(axes):
        for r in rows:
            if r["forme"] != s or "teinte" not in r:
                continue
            focus = r["config"] in FOCUS
            ax.scatter(np.radians(r["teinte"]), 1.0 if focus else 0.75, s=40 + 260 * r["constance"],
                       color=r["rgb"] / 255, marker=markers[r["config"]], edgecolors="k", linewidths=0.6,
                       alpha=1.0 if focus else 0.55, label=CONFIG_LABELS.get(r["config"]))
        ax.set_ylim(0, 1.25)
        ax.set_yticklabels([])
        ax.set_title(SHAPE_LABELS[s], fontsize=10)
    uniq = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            uniq.setdefault(lab, h)
    fig.legend(uniq.values(), uniq.keys(), fontsize=7, loc="lower center", ncol=5)
    fig.suptitle("Teinte dominante de chaque modèle (cercle extérieur : réglages retenus ; taille : constance)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path,
                    default=paths.RESULTS_DIR / "saturation" / "traduction_seule" / "arbitrary_shape_noise_0.01")
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "publication" / "recurrence")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models = []
    for cfg in CONFIG_ORDER:
        for f in sorted((args.results_dir / cfg).glob("results_*_sans_couleurs_seed*/binned_results.pkl")):
            models.append((cfg, f.parent.name.split("_")[-1], f))
    factors = dataset_factors()
    rows, verified = analyze(models, factors, args.out)
    write_outputs(rows, verified, args.out)
    print(f"{len(models)} modèles ; correspondance vérifiée : {verified} ; résultats dans {args.out}")


if __name__ == "__main__":
    main()
