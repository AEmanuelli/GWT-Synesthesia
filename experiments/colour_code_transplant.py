"""Greffe du code couleur : le Global Workspace lit-il la couleur cachée dans les attributs décodés ?

Pour chaque image du sous-ensemble de `residual_regression.py`, on connaît ce que le GW en décode
(probabilités de forme p, attributs continus ĉ : image -> VAE -> GW -> attributs) et les vraies valeurs
(forme t, attributs continus c). On traduit ensuite (attributs -> GW -> image) sept vecteurs d'attributs :

  propre          (one-hot t, c)              aucune information de couleur : témoin, gris attendu
  cycle           (p, ĉ)                      seconde moitié du cycle complet
  forme           (p, c)                      on ne garde que le résidu des probabilités de forme
  continus        (one-hot t, ĉ)              on ne garde que les écarts des attributs continus
  greffe_forme    (p_d, c)                    résidu de forme d'une autre image d (même forme, autre teinte)
  greffe_continus (one-hot t, c + ĉ_d - c_d)  écarts continus de d, posés sur la géométrie de l'image
  greffe_tout     (p_d, c + ĉ_d - c_d)        les deux

et on mesure si la couleur produite reprend la teinte de la source (l'image elle-même, ou le donneur d
pour les greffes) : R̄ = |moyenne de exp(i(h_sortie - h_source))| sur toutes les images (1 : teinte
transmise ; 0 : sans rapport). Témoin des greffes : R̄ par rapport à la teinte du receveur (0 attendu,
puisque ses attributs propres ne contiennent pas de couleur). Validation : la condition « cycle » doit
redonner la transmission du cycle complet mesurée par l'évaluation standard.

    python experiments/colour_code_transplant.py                        # calcul + analyse
    python experiments/colour_code_transplant.py --experiment high_cycles --max-models 1 --per-cell 4   # essai
    python experiments/colour_code_transplant.py --analyze-only

Entrées : results/saturation/regression/donnees/ (lancer `residual_regression.py` avant).
Sorties : results/publication/greffe/ (donnees/, greffe.csv, synthese.md, greffe.png).
"""
import argparse
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
from shape_noise_sweep import CONFIG_LABELS, hls_arrays, load_model, md_table, thesis_models  # noqa: E402
from synesthesia_figure import translate  # noqa: E402
from SSD_utils import extract_shape_color, segment_shape  # noqa: E402

CONDITIONS = {
    "propre": "propre (aucun code)",
    "cycle": "cycle (p, ĉ)",
    "forme": "résidu de forme seul",
    "continus": "écarts continus seuls",
    "greffe_forme": "greffe : résidu de forme",
    "greffe_continus": "greffe : écarts continus",
    "greffe_tout": "greffe : les deux",
}
GRAFTS = ("greffe_forme", "greffe_continus", "greffe_tout")
CONFIG_ORDER = ["base_params", "high_translation", "high_contrastive", "high_cycles", "high_alpha"]
S_COLOURED = 0.3
N_BOOT = 1000


# ----------------------------------------------------------------------------------------------
# Entrées
# ----------------------------------------------------------------------------------------------
def subset(img, per_cell, seed):
    """per_cell images par case (teinte, forme, taille) parmi celles de residual_regression.py."""
    rng = np.random.default_rng(seed)
    cells = {}
    for i, key in enumerate(zip(img["color_index"], img["forme"], img["taille"])):
        cells.setdefault(key, []).append(i)
    keep = [i for idx in cells.values() for i in rng.choice(idx, size=min(per_cell, len(idx)), replace=False)]
    return np.sort(np.array(keep))


def pick_donors(shape, colour, seed):
    """Pour chaque image, un donneur de même forme et d'une autre teinte (tiré au hasard)."""
    rng = np.random.default_rng(seed + 1)
    donor = np.empty(len(shape), dtype=int)
    for t in np.unique(shape):
        idx = np.where(shape == t)[0]
        for i in idx:
            j = idx[rng.integers(len(idx))]
            while colour[j] == colour[i]:
                j = idx[rng.integers(len(idx))]
            donor[i] = j
    return donor


def attribute_vectors(t, c, p, c_hat, donor):
    onehot = np.eye(3, dtype=np.float32)[t]
    dev = c_hat - c
    grafted = np.clip(c + dev[donor], -1, 1).astype(np.float32)
    return {
        "propre": (onehot, c),
        "cycle": (p, c_hat),
        "forme": (p, c),
        "continus": (onehot, c_hat),
        "greffe_forme": (p[donor], c),
        "greffe_continus": (onehot, grafted),
        "greffe_tout": (p[donor], grafted),
    }


def measure(images):
    """Couleur comme dans l'évaluation : masque d'Otsu, 3 x 3 pixels au barycentre (RGB 0-255)."""
    return np.array([extract_shape_color(x, segment_shape(x), True) * 255 for x in images], dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Calcul
# ----------------------------------------------------------------------------------------------
def compute(models, img, keep, donor, data_dir, device, batch_size, n_examples):
    t = img["forme"][keep].astype(int)
    c = img["attributs_vrais"][keep].astype(np.float32)
    for exp_name, instance, ckpt, colored in models:
        f = data_dir / f"{exp_name}_{instance}.npz"
        if f.exists():
            print(f"Déjà calculé : {f.name}")
            continue
        src = paths.RESULTS_DIR / "saturation" / "regression" / "donnees" / f"{exp_name}_{instance}.npz"
        if not src.exists():
            print(f"Absent (lancer residual_regression.py) : {src}")
            continue
        d = np.load(src)
        assert np.array_equal(d["numero"], img["numero"]), "sous-ensembles d'images différents"
        p = d["probas"][keep].astype(np.float32)
        c_hat = d["attributs_decodes"][keep].astype(np.float32)
        print(f"Greffes : {exp_name}/{instance} ({len(keep)} images × {len(CONDITIONS)} conditions)")
        t0 = time.time()
        gw = load_model(ckpt, colored, device)
        out, examples = {}, {}
        for cond, (oh, at) in attribute_vectors(t, c, p, c_hat, donor).items():
            rgb = []
            for start in range(0, len(oh), batch_size):
                ims = translate(gw, oh[start:start + batch_size], at[start:start + batch_size], device)
                rgb.append(measure(ims))
                if start == 0:
                    examples[cond] = (ims[:n_examples] * 255).round().astype(np.uint8)
            out[cond] = np.concatenate(rgb)
        np.savez_compressed(f, **{f"rgb_{k}": v for k, v in out.items()},
                            **{f"exemples_{k}": v for k, v in examples.items()},
                            indices=keep, donneur=donor, config=exp_name, instance=instance)
        print(f"  {time.time() - t0:.0f} s")
        del gw
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# Analyse
# ----------------------------------------------------------------------------------------------
def rbar(dh_deg):
    return float(np.abs(np.mean(np.exp(1j * np.radians(dh_deg)))))


def rbar_ci(dh_deg, seed=0):
    z = np.exp(1j * np.radians(dh_deg))
    rng = np.random.default_rng(seed)
    vals = [np.abs(z[rng.integers(0, len(z), len(z))].mean()) for _ in range(N_BOOT)]
    return np.percentile(vals, [2.5, 97.5])


def full_cycle_transmission():
    """R̄ du cycle complet de l'évaluation standard (moyenne des 3 formes), d'après synesthesia_figure.py."""
    f = paths.RESULTS_DIR / "saturation" / "synesthesie" / "associations.csv"
    if not f.exists():
        return {}
    acc = {}
    for r in csv.DictReader(open(f, encoding="utf-8")):
        v = r.get("transmission_cycle_complet")
        if r.get("condition") == "annexe_E" and v not in (None, "", "nan"):
            acc.setdefault((r["config"], r["instance"]), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def analyze(data_dir, img, out_dir):
    palette_h = hls_arrays(img["rgb_palette"].astype(float))[0]
    rows = []
    for f in sorted(data_dir.glob("*.npz")):
        z = np.load(f)
        keep, donor = z["indices"], z["donneur"]
        h_src = palette_h[keep]
        for cond in CONDITIONS:
            h, _, s = hls_arrays(z[f"rgb_{cond}"].astype(float))
            ref = h_src[donor] if cond in GRAFTS else h_src
            dh = (h - ref + 180) % 360 - 180
            lo, hi = rbar_ci(dh)
            row = dict(config=str(z["config"]), instance=str(z["instance"]), condition=cond, n=len(h),
                       R=rbar(dh), R_lo=lo, R_hi=hi, S_mediane=float(np.median(s)),
                       colores=float(np.mean(s >= S_COLOURED)))
            row["R_receveur"] = rbar((h - h_src + 180) % 360 - 180) if cond in GRAFTS else np.nan
            col = s >= S_COLOURED
            row["R_colores"] = rbar(dh[col]) if col.sum() >= 20 else np.nan
            rows.append(row)
    if not rows:
        raise SystemExit(f"Aucun résultat dans {data_dir}")
    with open(out_dir / "greffe.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_synthesis(rows, out_dir / "synthese.md")
    figure(rows, out_dir / "greffe.png")
    hue_figure(data_dir, img, out_dir / "greffe_par_teinte.png", out_dir / "synthese.md")
    return rows


def hue_figure(data_dir, img, out_file, synthesis_file, conds=("cycle", "forme", "continus")):
    """Part des traductions correctes (colorées, à ±30° de la teinte de la source) selon la teinte d'entrée :
    quels canaux transmettent quelles teintes ?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette_h = hls_arrays(img["rgb_palette"].astype(float))[0]
    files = sorted(data_dir.glob("*.npz"), key=lambda f: (CONFIG_ORDER.index(str(np.load(f)["config"]))
                                                          if str(np.load(f)["config"]) in CONFIG_ORDER else 9, f.name))
    fig, axes = plt.subplots(1, len(files), figsize=(2.6 * len(files), 2.9), sharey=True, squeeze=False)
    colours = {"cycle": "0.2", "forme": "#2a78d6", "continus": "#eb6834"}
    lines = ["", "## Transmission selon la teinte d'entrée", "",
             "Part des traductions colorées (S ≥ 0,3) dont la teinte est à ±30° de celle de l'image source, "
             "par teinte d'entrée (moyenne des teintes chaudes, 0–90° et 300–360°, et froides, 90–300°).", "",
             "| Instance | " + " | ".join(f"{CONDITIONS[c]} : chaudes / froides" for c in conds) + " |",
             "|---|" + "---|" * len(conds)]
    for ax, f in zip(axes[0], files):
        z = np.load(f)
        keep = z["indices"]
        h_src = palette_h[keep]
        order = np.argsort(np.unique(h_src))
        hues = np.unique(h_src)[order]
        cells = []
        for c in conds:
            h, _, s = hls_arrays(z[f"rgb_{c}"].astype(float))
            ok = (s >= S_COLOURED) & (np.abs((h - h_src + 180) % 360 - 180) <= 30)
            frac = np.array([ok[np.isclose(h_src, u)].mean() for u in hues])
            ax.plot(hues, frac, "o-", ms=3, lw=1.2, color=colours[c], label=CONDITIONS[c])
            warm = (hues < 90) | (hues >= 300)
            cells.append(f"{100 * frac[warm].mean():.0f} % / {100 * frac[~warm].mean():.0f} %")
        cfg, inst = str(z["config"]), str(z["instance"])
        lines.append(f"| {CONFIG_LABELS.get(cfg, cfg)} {inst} | " + " | ".join(cells) + " |")
        ax.set_title(f"{CONFIG_LABELS.get(cfg, cfg)}\ngraine {inst[-1]}", fontsize=8)
        ax.set_xlabel("teinte d'entrée (°)", fontsize=8)
        ax.set_xticks([0, 120, 240, 360])
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("part transmise (±30°)", fontsize=8)
    axes[0][0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)
    with open(synthesis_file, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def fr(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")


def write_synthesis(rows, out_file):
    by = {(r["config"], r["instance"], r["condition"]): r for r in rows}
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in by)]
    insts = sorted({k[1] for k in by})
    ref = full_cycle_transmission()
    L = ["# Greffe du code couleur", "",
         "Traduction (attributs → GW → image) de vecteurs d'attributs construits à partir de ce que le GW décode "
         "d'une image (voir `experiments/colour_code_transplant.py`). R̄ : transmission de la teinte de la source "
         "(l'image elle-même ; le donneur pour les greffes), 1 = transmise, 0 = sans rapport, IC 95 % par bootstrap "
         "des images. S : saturation médiane. « colorées » : part des traductions avec S ≥ 0,3.", ""]
    L += ["## R̄ par configuration (moyenne des instances ; min–max)", ""]
    header = ["Configuration"] + [CONDITIONS[c] for c in CONDITIONS]
    table = []
    for cfg in configs:
        cells = [CONFIG_LABELS.get(cfg, cfg)]
        for cond in CONDITIONS:
            vals = [by[(cfg, i, cond)]["R"] for i in insts if (cfg, i, cond) in by]
            cells.append(f"{fr(np.mean(vals))} ({fr(min(vals))}–{fr(max(vals))})" if vals else "–")
        table.append(cells)
    L.append(md_table(header, table))
    L += ["", "## Détail par instance", "",
          "R̄ [IC 95 %] ; S médiane ; part colorée. Greffes : entre parenthèses, R̄ par rapport au receveur "
          "(témoin, 0 attendu). Dernière colonne : R̄ du cycle complet de l'évaluation standard (validation de la "
          "condition « cycle »).", ""]
    header = ["Instance"] + [CONDITIONS[c] for c in CONDITIONS] + ["R̄ cycle complet (évaluation)"]
    table = []
    for cfg in configs:
        for inst in insts:
            if (cfg, inst, "cycle") not in by:
                continue
            cells = [f"{CONFIG_LABELS.get(cfg, cfg)} {inst}"]
            for cond in CONDITIONS:
                r = by[(cfg, inst, cond)]
                txt = f"{fr(r['R'])} [{fr(r['R_lo'])}–{fr(r['R_hi'])}] ; S {fr(r['S_mediane'])} ; {fr(100 * r['colores'], 0)} %"
                if cond in GRAFTS:
                    txt += f" ({fr(r['R_receveur'])})"
                cells.append(txt)
            cells.append(fr(ref.get((cfg, inst), np.nan)))
            table.append(cells)
    L.append(md_table(header, table))
    if ref:
        pairs = [(by[(c, i, "cycle")]["R"], ref[(c, i)]) for c, i, cond in by if cond == "cycle" and (c, i) in ref]
        if len(pairs) >= 3:
            a, b = np.array(pairs).T
            from scipy.stats import spearmanr
            L += ["", f"Validation : R̄ de la condition « cycle » contre R̄ du cycle complet de l'évaluation standard, "
                      f"{len(pairs)} instances : ρ de Spearman = {fr(spearmanr(a, b).statistic)}, écart absolu médian "
                      f"= {fr(float(np.median(np.abs(a - b))), 3)}."]
    out_file.write_text("\n".join(L) + "\n", encoding="utf-8")


def figure(rows, out_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(r["config"], r["instance"], r["condition"]): r for r in rows}
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in by)]
    insts = sorted({k[1] for k in by})
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(2, len(configs), figsize=(3.4 * len(configs), 6.2), sharey="row", squeeze=False)
    x = np.arange(len(conds))
    width = 0.8 / max(1, len(insts))
    for j, cfg in enumerate(configs):
        for k, inst in enumerate(insts):
            R = [by[(cfg, inst, c)]["R"] if (cfg, inst, c) in by else np.nan for c in conds]
            lo = [by[(cfg, inst, c)]["R_lo"] if (cfg, inst, c) in by else np.nan for c in conds]
            hi = [by[(cfg, inst, c)]["R_hi"] if (cfg, inst, c) in by else np.nan for c in conds]
            S = [by[(cfg, inst, c)]["S_mediane"] if (cfg, inst, c) in by else np.nan for c in conds]
            pos = x - 0.4 + width * (k + 0.5)
            err = np.array([np.subtract(R, lo), np.subtract(hi, R)])
            axes[0][j].bar(pos, R, width, color=f"C{int(inst[-1])}", label=f"graine {inst[-1]}",
                           yerr=err, error_kw=dict(lw=0.6, capsize=1))
            axes[1][j].bar(pos, S, width, color=f"C{int(inst[-1])}")
        axes[0][j].set_title(CONFIG_LABELS.get(cfg, cfg), fontsize=10)
        for ax in axes[:, j]:
            ax.set_xticks(x)
            ax.set_xticklabels([CONDITIONS[c] for c in conds], rotation=40, ha="right", fontsize=7.5)
            ax.axvspan(3.5, 6.5, color="0.93", zorder=0)
            ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("R̄ : teinte de la source transmise")
    axes[1][0].set_ylabel("S médiane de la traduction")
    axes[0][0].set_ylim(0, 1.02)
    axes[1][0].set_ylim(0, 1.0)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Greffe du code couleur : traduction d'attributs décodés, entiers ou en partie\n"
                 "(fond gris : code pris sur une autre image, de même forme et d'une autre teinte)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def examples_figure(model_key, img, out_file, device, shape_name="egg", n=6):
    """Figure d'exemples : n images d'une même forme, de teintes espacées ; pour chacune, l'image d'entrée et
    sa traduction dans chaque condition ; la greffe prend le code de l'image suivante (donneur, cyclique)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shape_noise_sweep import SHAPES, SHAPE_LABELS
    exp_name, instance = model_key
    ckpt = next(m[2] for m in thesis_models([exp_name], include_controls=False) if m[1] == instance)
    d = np.load(paths.RESULTS_DIR / "saturation" / "regression" / "donnees" / f"{exp_name}_{instance}.npz")
    t = SHAPES.index(shape_name)
    hues = np.linspace(0, 25, n, endpoint=False).astype(int)
    chosen = []
    for ci in hues:  # la plus grande taille disponible, pour bien voir la couleur
        idx = np.where((img["forme"] == t) & (img["color_index"] == ci))[0]
        chosen.append(idx[np.argmax(img["taille"][idx])])
    chosen = np.array(chosen)
    donor = np.roll(np.arange(n), -1)
    vec = attribute_vectors(np.full(n, t), img["attributs_vrais"][chosen].astype(np.float32),
                            d["probas"][chosen].astype(np.float32),
                            d["attributs_decodes"][chosen].astype(np.float32), donor)
    gw = load_model(ckpt, False, device)
    shown = ["propre", "cycle", "forme", "continus", "greffe_forme", "greffe_continus"]
    outs = {c: translate(gw, *vec[c], device) for c in shown}
    cols = ["entrée"] + [CONDITIONS[c] for c in shown]
    fig, axes = plt.subplots(n, len(cols), figsize=(1.35 * len(cols), 1.45 * n))
    for r in range(n):
        axes[r][0].imshow(img["images"][chosen[r]])
        for j, c in enumerate(shown, start=1):
            axes[r][j].imshow(outs[c][r])
            if c.startswith("greffe"):  # couleur du donneur, en coin
                col = img["rgb_palette"][chosen[donor[r]]] / 255
                axes[r][j].add_patch(plt.Rectangle((24, 0), 8, 8, color=col, ec="white", lw=0.8))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for j, c in enumerate(cols):
        axes[0][j].set_title(c, fontsize=7.5)
    fig.suptitle(f"{CONFIG_LABELS.get(exp_name, exp_name)}, graine {instance[-1]} — {SHAPE_LABELS[t]}s : "
                 "traduction des attributs décodés (carré en coin : teinte du donneur)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", action="append", help="Configuration(s) (défaut : toutes)")
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--per-cell", type=int, default=16, help="Images par case teinte × forme × taille (≤ 32)")
    ap.add_argument("--seed", type=int, default=0, help="Tirage des images et des donneurs")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--n-examples", type=int, default=24, help="Traductions gardées comme exemples")
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "publication" / "greffe")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--examples", action="append", metavar="CONFIG:INSTANCE[:FORME]",
                    help="Figure d'exemples pour ce modèle (ex. high_cycles:seed0:egg), sans refaire le calcul")
    args = ap.parse_args()
    if args.examples:
        img = dict(np.load(paths.RESULTS_DIR / "saturation" / "regression" / "donnees" / "images.npz"))
        device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
        for spec in args.examples:
            parts = spec.split(":")
            shape = parts[2] if len(parts) > 2 else "egg"
            out_file = args.out / f"exemples_{parts[0]}_{parts[1]}_{shape}.png"
            examples_figure((parts[0], parts[1]), img, out_file, device, shape)
            print(f"Écrit : {out_file}")
        return

    data_dir = args.out / "donnees"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_file = paths.RESULTS_DIR / "saturation" / "regression" / "donnees" / "images.npz"
    img = dict(np.load(img_file))
    keep = subset(img, args.per_cell, args.seed)
    donor = pick_donors(img["forme"][keep], img["color_index"][keep], args.seed)
    if not args.analyze_only:
        device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
        models = [m for m in thesis_models(args.experiment, include_controls=False)]
        if args.max_models:
            models = models[:args.max_models]
        compute(models, img, keep, donor, data_dir, device, args.batch_size, args.n_examples)
    analyze(data_dir, img, args.out)
    print(f"Résultats dans {args.out}")


if __name__ == "__main__":
    main()
