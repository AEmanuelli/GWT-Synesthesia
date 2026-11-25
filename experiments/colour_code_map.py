"""Carte et écriture du code couleur caché dans le résidu de forme.

Pour un modèle, une forme t et une géométrie (position, taille, orientation) fixées, on traduit
(attributs -> GW -> image) le vecteur de forme

    p_t = 1 - m,   p_j = m·w,   p_k = m·(1 - w)        (j < k : les deux formes non cibles)

pour une grille de masses résiduelles m (0,01 % à 3 %) et de répartitions w (0 : tout vers k ; 1 : tout
vers j), sans rien changer d'autre. On mesure la couleur produite (masque d'Otsu, 3 × 3 pixels au
barycentre) et si la forme est encore reconnue (image traduite -> GW -> attributs).

  1. Carte : quelle teinte pour quelle répartition et quelle masse ? Quelle part du cercle des teintes
     le code atteint-il ? À partir de quelle masse la traduction se colore-t-elle ?
  2. Invariance : la même (m, w) donne-t-elle la même teinte quelle que soit la géométrie ?
  3. Écriture : en choisissant (m, w) sur la moitié des géométries (carte), peut-on donner à la forme une
     teinte fixée à l'avance sur les autres géométries ? Réussite : teinte à ±30° de la cible, forme
     colorée (S ≥ 0,3) et reconnue.

    python experiments/colour_code_map.py                       # Emphase Cycle-Complet et Référence
    python experiments/colour_code_map.py --experiment high_cycles --max-models 1 --per-shape 2   # essai
    python experiments/colour_code_map.py --analyze-only
    python experiments/colour_code_map.py --demo high_cycles:seed0   # figure d'écriture (recharge le modèle)

Sorties : results/publication/carte/ (donnees/, carte.csv, synthese.md, figures).
"""
import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401
from shape_noise_sweep import (CONFIG_LABELS, SHAPE_LABELS, hls_arrays, load_model, md_table,  # noqa: E402
                               run_model, thesis_models)
from SSD_utils import generate_fixed_colors, normalize_position, normalize_rotation, normalize_size  # noqa: E402

M_GRID = np.r_[0.0, np.logspace(-4, np.log10(0.03), 20)]   # masse résiduelle retirée à la forme cible
W_GRID = np.linspace(0, 1, 41)                              # part de cette masse donnée à la forme j
M_REF = 0.01
S_COLOURED = 0.3
TOL = 30.0
N_COLORS = 25
TARGETS = np.arange(0, 360, 30)
DEFAULT_EXPERIMENTS = ["high_cycles", "base_params"]
CONFIG_ORDER = ["high_cycles", "base_params", "high_alpha", "high_translation", "high_contrastive"]


# ----------------------------------------------------------------------------------------------
# Entrées
# ----------------------------------------------------------------------------------------------
def geometries(per_shape, seed):
    """per_shape géométries par forme, tirées du jeu d'évaluation (taille ≥ 10 pour bien voir la couleur)."""
    rows = [r for r in csv.DictReader(open(paths.EVALSET_DIR / "attributes.csv")) if float(r["size"]) >= 10]
    rng = np.random.default_rng(seed)
    out = []
    for t in range(3):
        cand = [r for r in rows if int(r["class"]) == t]
        for g, i in enumerate(rng.choice(len(cand), per_shape, replace=False)):
            r = cand[i]
            px, py = normalize_position(list(ast.literal_eval(r["location"])))
            rx, ry = normalize_rotation(float(r["rotation"]))
            out.append(dict(shape=t, geo=g, cont=[px, py, normalize_size(float(r["size"])), rx, ry],
                            color_index=(7 * (t * per_shape + g)) % N_COLORS))
    return out


def shape_vector(t, m, w):
    j, k = [c for c in range(3) if c != t]
    p = np.zeros(3)
    p[t], p[j], p[k] = 1 - m, m * w, m * (1 - w)
    return p


def build_inputs(geos, colored, cells=None):
    """Vecteurs de forme de la grille (ou des cellules (m, w) données) pour chaque géométrie."""
    palette, _ = generate_fixed_colors(N_COLORS)
    oh, at, meta = [], [], []
    for gi, g in enumerate(geos):
        cont = np.array(g["cont"], dtype=np.float64)
        colour = palette[g["color_index"]] / 127.5 - 1
        grid = cells if cells is not None else [(m, w) for m in M_GRID for w in ([0.5] if m == 0 else W_GRID)]
        for m, w in grid:
            oh.append(shape_vector(g["shape"], m, w))
            at.append(np.r_[cont, colour] if colored else cont)
            meta.append((gi, g["shape"], g["geo"], m, w))
    return np.array(oh, np.float32), np.array(at, np.float32), np.array(meta, np.float64)


# ----------------------------------------------------------------------------------------------
# Calcul
# ----------------------------------------------------------------------------------------------
def compute(models, geos, data_dir, device, batch_size):
    for exp_name, instance, ckpt, colored in models:
        f = data_dir / f"{exp_name}_{instance}.npz"
        if f.exists():
            print(f"Déjà calculé : {f.name}")
            continue
        oh, at, meta = build_inputs(geos, colored)
        print(f"Carte : {exp_name}/{instance} ({len(oh)} traductions)")
        t0 = time.time()
        gw = load_model(ckpt, colored, device)
        bary, _, areas, decoded = run_model(gw, oh, at, device, batch_size)
        np.savez_compressed(f, meta=meta, rgb=bary, aire=areas, probas_redecodees=decoded[:, :3],
                            config=exp_name, instance=instance, colored=colored)
        print(f"  {time.time() - t0:.0f} s")
        del gw
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# Analyse
# ----------------------------------------------------------------------------------------------
def circ_mean(h):
    z = np.exp(1j * np.radians(h)).mean()
    return float(np.degrees(np.angle(z)) % 360), float(np.abs(z))


def circ_sd(h):
    r = circ_mean(h)[1]
    return float(np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12)))))


def dist(a, b):
    return np.abs((np.asarray(a) - b + 180) % 360 - 180)


def load(data_dir):
    res = {}
    for f in sorted(data_dir.glob("*.npz")):
        z = np.load(f)
        meta = z["meta"]
        h, _, s = hls_arrays(z["rgb"].astype(float))
        t = meta[:, 1].astype(int)
        res[(str(z["config"]), str(z["instance"]))] = dict(
            gi=meta[:, 0].astype(int), t=t, geo=meta[:, 2].astype(int), m=meta[:, 3], w=meta[:, 4],
            H=h, S=s, rgb=z["rgb"], rec=z["probas_redecodees"].argmax(1) == t, colored=bool(z["colored"]))
    return res


def cell_table(d, t, geo_sel):
    """Par cellule (m, w), m > 0 : teinte moyenne circulaire, R̄, part colorée et part reconnue sur geo_sel."""
    out = []
    base = (d["t"] == t) & np.isin(d["geo"], geo_sel) & (d["m"] > 0)
    for m in M_GRID[1:]:
        for w in W_GRID:
            sel = base & np.isclose(d["m"], m) & np.isclose(d["w"], w)
            col = sel & (d["S"] >= S_COLOURED)
            mu, r = circ_mean(d["H"][col]) if col.any() else (np.nan, 0.0)
            out.append(dict(m=m, w=w, hue=mu, R=r, colores=col.sum() / max(1, sel.sum()),
                            S=float(np.median(d["S"][sel])) if sel.any() else np.nan,
                            reconnue=d["rec"][sel].mean() if sel.any() else np.nan))
    return out


def analyze_model(d, n_geo):
    rows = []
    geo_a, geo_b = np.arange(n_geo // 2), np.arange(n_geo // 2, n_geo)
    for t in range(3):
        st = d["t"] == t
        clean = st & (d["m"] == 0)
        r = dict(forme=t, S_sans_residu=float(np.median(d["S"][clean])))
        # 1. Seuil : plus petite masse où la moitié des traductions (toutes répartitions, géométries) est colorée
        m_star = np.nan
        for m in M_GRID[1:]:
            sel = st & np.isclose(d["m"], m)
            if np.mean(d["S"][sel] >= S_COLOURED) >= 0.5:
                m_star = m
                break
        r["m_seuil"] = m_star
        m_ref = M_GRID[np.argmin(np.abs(M_GRID - M_REF))]
        at_ref = st & np.isclose(d["m"], m_ref)
        r["S_m_ref"] = float(np.median(d["S"][at_ref]))
        r["colores_m_ref"] = float(np.mean(d["S"][at_ref] >= S_COLOURED))
        r["reconnue_m_le_ref"] = float(d["rec"][st & (d["m"] <= m_ref)].mean())
        # 2. Invariance : dispersion circulaire de la teinte entre géométries, à (m_ref, w) fixés
        sds = []
        for w in W_GRID:
            sel = at_ref & np.isclose(d["w"], w) & (d["S"] >= S_COLOURED)
            if sel.sum() >= max(3, n_geo // 2):
                sds.append(circ_sd(d["H"][sel]))
        r["dispersion_geometries"] = float(np.median(sds)) if sds else np.nan
        # Couverture : secteurs de 30° atteints par une cellule colorée de façon fiable (toutes géométries)
        cells = cell_table(d, t, np.arange(n_geo))
        hues = [c["hue"] for c in cells if c["colores"] >= 0.75 and c["R"] >= 0.8]
        r["secteurs_atteints"] = int(len({int(h // 30) % 12 for h in hues}))
        # 3. Écriture : cellule choisie sur les géométries A, testée sur les géométries B
        cells_a = [c for c in cell_table(d, t, geo_a) if c["colores"] >= 0.75 and c["reconnue"] >= 0.9]
        reach, success, trials, chosen = 0, 0, 0, {}
        for target in TARGETS:
            if not cells_a:
                break
            # parmi les cellules à ±15° de la cible, la plus saturée ; sinon la plus proche (≤ TOL)
            close = [c for c in cells_a if dist(c["hue"], target) <= TOL / 2]
            best = max(close, key=lambda c: c["S"]) if close else min(cells_a, key=lambda c: dist(c["hue"], target))
            if dist(best["hue"], target) > TOL:
                continue
            reach += 1
            chosen[int(target)] = (best["m"], best["w"])
            sel = st & np.isin(d["geo"], geo_b) & np.isclose(d["m"], best["m"]) & np.isclose(d["w"], best["w"])
            ok = (d["S"][sel] >= S_COLOURED) & d["rec"][sel] & (dist(d["H"][sel], target) <= TOL)
            success += int(ok.sum())
            trials += int(sel.sum())
        r["cibles_atteignables"] = reach
        r["reussite_ecriture"] = success / trials if trials else np.nan
        r["reussite_ecriture_toutes_cibles"] = success / (len(TARGETS) * len(geo_b))
        r["cellules_choisies"] = chosen
        rows.append(r)
    return rows


def fr(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")


def pct(m):
    return "–" if not np.isfinite(m) else f"{100 * m:.2g} %".replace(".", ",")


def analyze(data_dir, out_dir, n_geo):
    res = load(data_dir)
    if not res:
        raise SystemExit(f"Aucun résultat dans {data_dir}")
    keys = sorted(res, key=lambda k: (CONFIG_ORDER.index(k[0]) if k[0] in CONFIG_ORDER else 9,
                                      k[1] == "control", k[1]))
    table, all_rows = [], []
    for key in keys:
        for r in analyze_model(res[key], n_geo):
            r.update(config=key[0], instance=key[1])
            all_rows.append(r)
            who = f"{CONFIG_LABELS.get(key[0], key[0])} — " + ("contrôle" if key[1] == "control"
                                                               else f"graine {key[1][-1]}")
            table.append([who, SHAPE_LABELS[r["forme"]], fr(r["S_sans_residu"]), pct(r["m_seuil"]),
                          fr(r["S_m_ref"]), f"{fr(100 * r['colores_m_ref'], 0)} %",
                          f"{fr(100 * r['reconnue_m_le_ref'], 0)} %",
                          f"{fr(r['dispersion_geometries'], 0)}°", f"{r['secteurs_atteints']}/12",
                          f"{r['cibles_atteignables']}/12",
                          f"{fr(100 * r['reussite_ecriture'], 0)} %" if np.isfinite(r["reussite_ecriture"]) else "–"])
    with open(out_dir / "carte.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [k for k in all_rows[0] if k != "cellules_choisies"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    L = ["# Carte et écriture du code couleur", "",
         f"Traduction avec p_t = 1 − m, p_j = m·w, p_k = m·(1 − w) ; {len(M_GRID) - 1} masses m de 0,01 % à 3 %, "
         f"{len(W_GRID)} répartitions w ; {n_geo} géométries par forme. Colorée : S ≥ {fr(S_COLOURED, 1)}. "
         f"Reconnue : forme re-décodée depuis l'image traduite.", "",
         "- seuil : plus petite masse où la moitié des traductions (toutes répartitions et géométries) est colorée ;",
         f"- à m = {pct(M_GRID[np.argmin(np.abs(M_GRID - M_REF))])} : S médiane, part colorée ; reconnue : pour m ≤ cette masse ;",
         "- dispersion : écart-type circulaire de la teinte entre géométries, à (m, w) fixés (médiane sur w) ;",
         "- secteurs : secteurs de 30° du cercle des teintes atteints de façon fiable (≥ 75 % colorées, R̄ ≥ 0,8) ;",
         f"- écriture : cellule (m, w) choisie sur {n_geo // 2} géométries pour chacune des 12 teintes cibles, "
         f"testée sur les {n_geo - n_geo // 2} autres (réussite : ±{TOL:.0f}°, colorée, reconnue), parmi les cibles atteignables.", "",
         md_table(["Modèle", "Forme", "S sans résidu", "seuil", "S (1 %)", "colorées (1 %)", "reconnue", "dispersion",
                   "secteurs", "cibles atteignables", "réussite de l'écriture"], table)]
    (out_dir / "synthese.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for key in keys:
        map_figure(res[key], key, out_dir / f"carte_{key[0]}_{key[1]}.png")
    return all_rows


def map_figure(d, key, out_file):
    """Couleur produite selon (w, m), pour deux géométries de chaque forme (même échelle, couleur réelle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = M_GRID[1:]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), squeeze=False)
    for t in range(3):
        for row, geo in enumerate((0, 1)):
            ax = axes[row][t]
            grid = np.zeros((len(ms), len(W_GRID), 3))
            for a, m in enumerate(ms):
                for b, w in enumerate(W_GRID):
                    sel = (d["t"] == t) & (d["geo"] == geo) & np.isclose(d["m"], m) & np.isclose(d["w"], w)
                    if sel.any():
                        grid[a, b] = d["rgb"][sel][0] / 255
            ax.imshow(grid, origin="lower", aspect="auto", extent=(0, 1, 0, len(ms)), interpolation="nearest")
            ticks = [0, 5, 10, 15, 19]
            ax.set_yticks([i + 0.5 for i in ticks])
            ax.set_yticklabels([pct(ms[i]) for i in ticks], fontsize=7)
            ax.set_xlabel("répartition w (0 : tout vers l'autre forme ; 1 : tout vers la première)", fontsize=7)
            ax.set_title(f"{SHAPE_LABELS[t]}, géométrie {geo + 1}", fontsize=9)
            if t == 0:
                ax.set_ylabel("masse résiduelle m", fontsize=8)
    who = "contrôle" if key[1] == "control" else f"graine {key[1][-1]}"
    fig.suptitle(f"{CONFIG_LABELS.get(key[0], key[0])}, {who} : couleur traduite selon la masse et la répartition du résidu",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def demo_figure(key, geos, out_dir, n_geo, device):
    """Écriture : la cellule (m, w) choisie sur les géométries A pour chaque teinte cible, appliquée aux
    géométries B ; une ligne par forme et par géométrie B, une colonne par cible (couleur réelle produite)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from synesthesia_figure import translate
    res = load(out_dir / "donnees")
    rows = analyze_model(res[key], n_geo)
    ckpt = next(m[2] for m in thesis_models([key[0]], include_controls=key[1] == "control") if m[1] == key[1])
    gw = load_model(ckpt, key[1] == "control", device)
    geo_b = [g for g in geos if g["geo"] >= n_geo // 2]
    n_show = min(2, len(geo_b) // 3)
    fig, axes = plt.subplots(3 * n_show + 1, len(TARGETS), figsize=(1.0 * len(TARGETS), 1.05 * (3 * n_show + 1)))
    from matplotlib.colors import hsv_to_rgb
    for c, target in enumerate(TARGETS):
        axes[0][c].imshow(np.full((8, 8, 3), hsv_to_rgb([target / 360, 0.75, 0.9])))
        axes[0][c].set_title(f"{target}°", fontsize=7)
    for t in range(3):
        chosen = rows[t]["cellules_choisies"]
        gs = [g for g in geo_b if g["shape"] == t][:n_show]
        for i, g in enumerate(gs):
            ax_row = axes[1 + t * n_show + i]
            cells = [chosen.get(int(target)) for target in TARGETS]
            todo = [(m, w) for cell in cells if cell is not None for m, w in [cell]]
            if todo:
                oh, at, _ = build_inputs([g], key[1] == "control", todo)
                ims = translate(gw, oh, at, device)
            k = 0
            for c, cell in enumerate(cells):
                if cell is None:
                    ax_row[c].imshow(np.zeros((32, 32, 3)))
                    ax_row[c].text(16, 16, "–", color="w", ha="center", va="center")
                else:
                    ax_row[c].imshow(ims[k])
                    k += 1
            ax_row[0].set_ylabel(SHAPE_LABELS[t], fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    who = "contrôle" if key[1] == "control" else f"graine {key[1][-1]}"
    fig.suptitle(f"Écriture d'une teinte par le résidu de forme — {CONFIG_LABELS.get(key[0], key[0])}, {who}\n"
                 "ligne du haut : teinte demandée ; dessous : traduction de géométries non vues lors du choix du résidu",
                 fontsize=8)
    fig.tight_layout()
    out_file = out_dir / f"ecriture_{key[0]}_{key[1]}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", action="append", help=f"Configuration(s) (défaut : {DEFAULT_EXPERIMENTS})")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--per-shape", type=int, default=8, help="Géométries par forme (moitié carte, moitié test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "publication" / "carte")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--demo", action="append", metavar="CONFIG:INSTANCE", help="Figure d'écriture pour ce modèle")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args()

    data_dir = args.out / "donnees"
    data_dir.mkdir(parents=True, exist_ok=True)
    geos = geometries(args.per_shape, args.seed)
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.demo:
        for spec in args.demo:
            cfg, inst = spec.split(":")[:2]
            print(f"Écrit : {demo_figure((cfg, inst), geos, args.out, args.per_shape, device)}")
        return
    if not args.analyze_only:
        models = thesis_models(args.experiment or DEFAULT_EXPERIMENTS, include_controls=not args.no_controls)
        if args.max_models:
            models = models[:args.max_models]
        compute(models, geos, data_dir, device, args.batch_size)
    analyze(data_dir, args.out, args.per_shape)
    print(f"Résultats dans {args.out}")


if __name__ == "__main__":
    main()
