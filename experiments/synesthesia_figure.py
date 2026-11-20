"""Synesthésie artificielle : quelles associations forme -> couleur sont les plus nettes ?

Chemin de traduction (attributs -> GW -> image) des 15 modèles « vide sémantique », lu dans
l'évaluation --hls --translation-only (51 200 traductions par forme et par modèle), sans bruit et
avec le bruit de l'annexe E (graine 42, 1 % sur le one-hot de la forme). Pour chaque modèle et
chaque forme, les couleurs sont résumées dans le plan de chrominance (a = R - (G+B)/2,
b = sqrt(3)/2 (G - B), unités RGB 0-255) :
  - chroma systématique : norme du vecteur chrominance moyen, grande seulement si la couleur est à
    la fois vive et constante (score de classement) ;
  - constance : part des traductions colorées (chroma >= 40) dont la teinte est à ±30° de la teinte
    dominante (direction du vecteur moyen) ;
  - couleur représentative : médiane R, G, B des traductions constantes, nommée d'après sa teinte
    (« gris » si moins de la moitié des traductions sont colorées, « variable » si moins de la
    moitié des traductions colorées ont la teinte dominante) ;
  - bande : 240 traductions tirées au hasard, d'abord celles de la teinte dominante (±30°), puis les
    autres teintes dans l'ordre du cercle, puis les grises.
Deux contrôles complètent le tableau :
  - balayage (shape_noise_sweep.py) : couleur quand tout le résidu (ε = 0,01) part vers l'une ou
    l'autre des deux autres formes ; « robuste » si les deux directions donnent la même couleur ;
  - cycle complet du chapitre 3 (image -> GW -> attributs sans couleur -> GW -> image) :
    concentration R̄ de l'écart de teinte entrée -> sortie (1 = la couleur d'entrée traverse les
    attributs sans couleur, cachée dans le résidu des probabilités de forme).
Les associations les plus marquantes (5 par forme, constance >= 60 %) sont redessinées avec de
vraies traductions produites par les checkpoints, pour les mêmes attributs chez tous les modèles
(jeu d'évaluation, taille >= 10 px, une orientation par quadrant, exemples non triés).

    python experiments/synesthesia_figure.py
    python experiments/synesthesia_figure.py --stats-only     # tableau seulement, sans checkpoint

Sorties dans results/saturation/synesthesie/ : associations.csv, synthese.md, galerie.png (les
associations les plus marquantes, par forme) et palettes.png (les 15 modèles).
"""

import argparse
import ast
import colorsys
import csv
import gc
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
from shape_noise_sweep import (CONFIG_LABELS, EPSILONS, KIND_SHAPE, N_DIRECTIONS, SHAPES,  # noqa: E402
                               annex_e_noise, load_model, thesis_models)
from SSD_utils import (extract_shape_color, normalize_position, normalize_rotation,  # noqa: E402
                       normalize_size, segment_shape)

SHAPE_NAMES = {"diamond": "losange", "egg": "œuf", "triangle": "triangle"}
SHAPE_TITLES = {"diamond": "Losanges", "egg": "Œufs", "triangle": "Triangles"}
SHORT_LABELS = {"base_params": "Référence", "high_translation": "Emph. Traduction",
                "high_contrastive": "Emph. Contrastive", "high_cycles": "Cycle-Complet",
                "high_alpha": "Attr. Continus"}
CONDITIONS = {"sans_bruit": "arbitrary", "annexe_E": "arbitrary_shape_noise_0.01"}
C_MIN = 40.0           # chroma (unités RGB 0-255) d'une couleur visible
HUE_TOL = 30.0         # tolérance de teinte (degrés) de la constance
MIN_CONSTANCY = 0.60   # seuil pour figurer dans la galerie
TOP_PER_SHAPE = 5
N_EXAMPLES, MIN_SIZE = 4, 10
STRIP_N = 240          # traductions par bande de couleurs
SWEEP_EPS = 0.01
ROBUST_CHROMA, ROBUST_HUE = 30.0, 45.0
HUE_NAMES = [(15, "rouge"), (40, "orange"), (65, "jaune"), (90, "vert-jaune"), (150, "vert"),
             (178, "turquoise"), (200, "cyan"), (245, "bleu"), (270, "bleu-violet"), (290, "violet"),
             (330, "magenta"), (350, "cramoisi"), (361, "rouge")]
NUMERIC = {"n", "chroma_systematique", "constance", "colorees", "teinte", "teinte_dominante", "R", "G", "B",
           "L", "S", "S_mediane", "residu_1_teinte", "residu_1_chroma", "residu_2_teinte", "residu_2_chroma",
           "transmission_cycle_complet"}


# ----------------------------------------------------------------------------------------------
# Couleurs
# ----------------------------------------------------------------------------------------------
def opponent(rgb):
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    return rgb[:, 0] - (rgb[:, 1] + rgb[:, 2]) / 2, np.sqrt(3) / 2 * (rgb[:, 1] - rgb[:, 2])


def circ_diff(h1, h2):
    return np.abs((np.asarray(h1) - np.asarray(h2) + 180) % 360 - 180)


def color_name(rgb, constancy=1.0, colored=1.0):
    rgb = np.asarray(rgb, dtype=np.float64)
    if colored < 0.5 or rgb.max() - rgb.min() < C_MIN:
        return "gris"
    if constancy < 0.5 * colored:
        return "variable"
    h, l, _ = colorsys.rgb_to_hls(*(rgb / 255))
    hue, value = h * 360, rgb.max() / 255
    name = next(n for limit, n in HUE_NAMES if hue < limit)
    if 15 <= hue < 90 and value < 0.6:  # jaunes et orangés sombres
        return "olive" if hue >= 40 else name + " foncé"
    if value < 0.5:
        return name + " foncé"
    return name + " clair" if l > 0.65 else name


def summarize(rgb, s_values):
    """Couleur dominante, constance et couleur représentative d'un ensemble de traductions."""
    a, b = opponent(rgb)
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    mean_a, mean_b = a.mean(), b.mean()
    dominant = np.degrees(np.arctan2(mean_b, mean_a)) % 360
    colored = chroma >= C_MIN
    constant = colored & (circ_diff(hue, dominant) <= HUE_TOL)
    rep = np.median(rgb[constant] if constant.sum() >= 20 else rgb, axis=0)
    h, l, s = colorsys.rgb_to_hls(*(rep / 255))
    return {"n": len(rgb), "chroma_systematique": float(np.hypot(mean_a, mean_b)),
            "constance": float(constant.mean()), "colorees": float(colored.mean()),
            "teinte": float(h * 360), "teinte_dominante": float(dominant),
            "R": float(rep[0]), "G": float(rep[1]), "B": float(rep[2]), "L": float(l), "S": float(s),
            "S_mediane": float(np.median(s_values) / 255) if len(s_values) else float("nan"),
            "nom": color_name(rep, constant.mean(), colored.mean())}


def colour_strip(rgb, dominant, seed=0):
    """STRIP_N traductions tirées au hasard : teinte dominante (±HUE_TOL) d'abord, puis le reste du cercle des
    teintes dans le même sens, puis les grises (des claires aux sombres)."""
    rng = np.random.default_rng(seed)
    sub = rgb[rng.choice(len(rgb), size=min(STRIP_N, len(rgb)), replace=False)]
    a, b = opponent(sub)
    gray = np.hypot(a, b) < C_MIN
    signed = (np.degrees(np.arctan2(b, a)) - dominant + 180) % 360 - 180
    key = np.where(gray, 2000 + 255 - sub.mean(axis=1),
                   np.where(np.abs(signed) <= HUE_TOL, signed, 1000 + signed % 360))
    return np.round(sub[np.argsort(key, kind="stable")]).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# Lecture des résultats
# ----------------------------------------------------------------------------------------------
def shape_rgb(data, path, shape):
    dist = data[f"{path}_colors_by_attr"]["shape"][shape]
    return np.stack([np.asarray(dist[c], dtype=np.float64) for c in "RGB"], axis=1)


def cycle_transmission(pkl_file):
    """R̄ de l'écart de teinte entrée -> sortie du cycle complet, par forme (échantillons alignés)."""
    data = pickle.load(open(pkl_file, "rb"))
    out = {}
    for shape in SHAPES:
        rgb_in, rgb_out = shape_rgb(data, "input", shape), shape_rgb(data, "full_cycle", shape)
        if len(rgb_out) == 0 or len(rgb_in) != len(rgb_out):
            continue
        (a1, b1), (a2, b2) = opponent(rgb_in), opponent(rgb_out)
        out[shape] = float(np.abs(np.mean(np.exp(1j * (np.arctan2(b2, a2) - np.arctan2(b1, a1))))))
    del data
    gc.collect()
    return out


def sweep_directions(sweep_dir, exp_name, instance):
    """{(forme, forme qui reçoit le résidu): (teinte, chroma)} du balayage, à ε = SWEEP_EPS."""
    f = sweep_dir / "donnees" / f"{exp_name}_{instance}.npz"
    if not f.exists():
        return {}
    z = np.load(f)
    meta, rgb = z["meta"], z["rgb_barycentre"].astype(np.float64)
    kind, target, direction = meta[:, 2].astype(int), meta[:, 1].astype(int), meta[:, 3].astype(int)
    eps = EPSILONS[np.argmin(np.abs(EPSILONS - SWEEP_EPS))]
    step = 360 / N_DIRECTIONS
    out = {}
    for t, shape in enumerate(SHAPES):
        j, k = [c for c in range(3) if c != t]
        # 300° : tout le résidu vers j ; 240° : tout vers k (voir tangent_basis)
        for recipient, deg in ((j, 300), (k, 240)):
            sel = ((kind == KIND_SHAPE) & (target == t) & (direction == int(round(deg / step)))
                   & np.isclose(meta[:, 5], eps))
            a, b = opponent(rgb[sel])
            out[(shape, SHAPES[recipient])] = (float(np.degrees(np.arctan2(b.mean(), a.mean())) % 360),
                                               float(np.hypot(a.mean(), b.mean())))
    return out


def collect_stats(results_dir, chapter3_dir, sweep_dir):
    """Lignes du tableau et bandes de couleurs {(config, instance, condition, forme): (STRIP_N, 3)}."""
    manifest = yaml.safe_load(open(paths.THESIS_RUNS_FILE))["experiments"]
    rows, strips = [], {}
    for exp_name, exp in manifest.items():
        for run in exp["runs"]:
            instance = run["run_id"]
            run_dir = f"results_{exp_name}_sans_couleurs_{instance}"
            f3 = chapter3_dir / exp_name / run_dir / "binned_results.pkl"
            transmission = cycle_transmission(f3) if f3.exists() else {}
            sweep = sweep_directions(sweep_dir, exp_name, instance)
            for condition, sub_dir in CONDITIONS.items():
                f = results_dir / sub_dir / exp_name / run_dir / "binned_results.pkl"
                if not f.exists():
                    print(f"Absent, ignoré : {f}")
                    continue
                print(f"Lecture : {f}")
                data = pickle.load(open(f, "rb"))
                for shape in SHAPES:
                    rgb = shape_rgb(data, "translated", shape)
                    s_values = np.asarray(data["translated_colors_by_attr"]["shape"][shape].get("S", []))
                    row = {"config": exp_name, "instance": instance, "condition": condition, "forme": shape,
                           **summarize(rgb, s_values)}
                    strips[(exp_name, instance, condition, shape)] = colour_strip(rgb, row["teinte_dominante"])
                    others = [o for o in SHAPES if o != shape]
                    for i, other in enumerate(others, start=1):
                        hue, chroma = sweep.get((shape, other), (float("nan"), float("nan")))
                        row.update({f"residu_{i}_forme": other, f"residu_{i}_teinte": hue,
                                    f"residu_{i}_chroma": chroma})
                    row["robuste"] = bool(
                        sweep and all(row[f"residu_{i}_chroma"] >= ROBUST_CHROMA for i in (1, 2))
                        and circ_diff(row["residu_1_teinte"], row["residu_2_teinte"]) <= ROBUST_HUE)
                    row["transmission_cycle_complet"] = transmission.get(shape, float("nan"))
                    rows.append(row)
                del data
                gc.collect()
    return rows, strips


def write_csv(rows, f):
    with open(f, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(f):
    rows = []
    for r in csv.DictReader(open(f)):
        rows.append({k: (float(v) if k in NUMERIC else (v == "True" if k == "robuste" else v))
                     for k, v in r.items()})
    return rows


def select_gallery(stats):
    """Par forme : les TOP_PER_SHAPE associations les plus vives et constantes (bruit de l'annexe E)."""
    selection = {}
    for shape in SHAPES:
        cand = [r for r in stats if r["condition"] == "annexe_E" and r["forme"] == shape
                and r["constance"] >= MIN_CONSTANCY]
        best = sorted(cand, key=lambda r: -r["chroma_systematique"])[:TOP_PER_SHAPE]
        selection[shape] = sorted(best, key=lambda r: r["teinte"])
    return selection


# ----------------------------------------------------------------------------------------------
# Vraies traductions
# ----------------------------------------------------------------------------------------------
def pick_examples(csv_file, seed=0):
    """N_EXAMPLES attributs par forme (taille >= MIN_SIZE, une orientation par quadrant)."""
    rows = list(csv.DictReader(open(csv_file)))
    rng = np.random.default_rng(seed)
    picks = {}
    for t, shape in enumerate(SHAPES):
        chosen = []
        for quadrant in range(N_EXAMPLES):
            cand = [r for r in rows if int(r["class"]) == t and float(r["size"]) >= MIN_SIZE
                    and int(round(float(r["rotation"]) / (np.pi / 8))) % 16 // 4 == quadrant]
            chosen.append(cand[rng.integers(len(cand))])
        picks[shape] = chosen
    return picks


def example_inputs(picks, noisy):
    """One-hot (avec ou sans le bruit de l'annexe E) et attributs continus normalisés."""
    noise = annex_e_noise()
    one_hots, attrs = [], []
    for t, shape in enumerate(SHAPES):
        for r in picks[shape]:
            onehot = np.eye(3)[t]
            if noisy:  # comme _process_translation_path_batch (SSD_utils.py)
                onehot = np.clip(onehot + noise, 1e-8, None)
                onehot = onehot / onehot.sum()
            px, py = normalize_position(list(ast.literal_eval(r["location"])))
            rx, ry = normalize_rotation(float(r["rotation"]))
            one_hots.append(onehot)
            attrs.append([px, py, normalize_size(float(r["size"])), rx, ry])
    return np.array(one_hots, dtype=np.float32), np.array(attrs, dtype=np.float32)


def translate(gw, one_hots, attrs, device):
    """Traduction attributs -> GW -> image (comme shape_noise_sweep.run_model)."""
    from shimmer.modules.selection import SingleDomainSelection
    selection = SingleDomainSelection().to(device)
    with torch.no_grad():
        oh = torch.from_numpy(one_hots).to(device)
        at = torch.from_numpy(attrs).to(device)
        z = gw.gw_mod.encode_and_fuse({"attr": gw.encode_domain([oh, at], "attr")}, selection_module=selection)
        images = gw.domain_mods["v_latents"].decode_images(gw.gw_mod.decode(z)["v_latents"])
    return images.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()


def generate_examples(picks, device, cache_file):
    """{(config, instance, condition): images (3 * N_EXAMPLES, 32, 32, 3) uint8}, en cache."""
    if cache_file.exists():
        z = np.load(cache_file)
        return {tuple(k.split("|")): z[k] for k in z.files}
    images = {}
    for exp_name, instance, ckpt, _ in thesis_models(include_controls=False):
        print(f"Traductions : {exp_name}/{instance}")
        gw = load_model(ckpt, False, device)
        for condition in CONDITIONS:
            imgs = translate(gw, *example_inputs(picks, condition == "annexe_E"), device)
            images[(exp_name, instance, condition)] = np.round(imgs * 255).astype(np.uint8)
        del gw
    np.savez_compressed(cache_file, **{"|".join(k): v for k, v in images.items()})
    return images


def check_examples(stats, images):
    """Exemples dessinés à ±HUE_TOL de la teinte dominante, contre l'attendu d'après la constance."""
    inside, expected, total = 0, 0.0, 0
    for r in stats:
        key = (r["config"], r["instance"], r["condition"])
        if r["condition"] != "annexe_E" or r["constance"] < MIN_CONSTANCY or key not in images:
            continue
        t = SHAPES.index(r["forme"])
        for img in images[key][t * N_EXAMPLES:(t + 1) * N_EXAMPLES].astype(np.float32) / 255:
            a, b = opponent(extract_shape_color(img, segment_shape(img), True) * 255)
            inside += bool(np.hypot(a[0], b[0]) >= C_MIN
                           and circ_diff(np.degrees(np.arctan2(b[0], a[0])) % 360, r["teinte_dominante"]) <= HUE_TOL)
            expected += r["constance"]
            total += 1
    return inside, expected, total


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
class Canvas:
    """Figure dont les axes sont placés en pouces depuis le coin supérieur gauche."""

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor("white")

    def _axes(self, x, y, w, h):
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.75")
            spine.set_linewidth(0.5)
        return ax

    def image(self, img, x, y, size):
        self._axes(x, y, size, size).imshow(img, interpolation="nearest")

    def strip(self, colors, x, y, w, h):
        self._axes(x, y, w, h).imshow(colors[None], interpolation="nearest", aspect="auto")

    def rect(self, x, y, w, h, color):
        self.fig.add_artist(plt.Rectangle((x / self.w, 1 - (y + h) / self.h), w / self.w, h / self.h,
                                          transform=self.fig.transFigure, facecolor=color,
                                          edgecolor="0.4", linewidth=0.4))

    def text(self, x, y, s, **kw):
        self.fig.text(x / self.w, 1 - y / self.h, s, **kw)


def noise_robust(csv_file, min_other_seeds=3):
    """Associations sûres (graine 42, 0,01) qui restent sûres avec la même couleur pour au moins
    `min_other_seeds` des 5 autres graines du bruit (0,01) et aux deux autres amplitudes (graine 42),
    d'après `saturation_robustness.py` (bruit_conditions.csv). Ensemble vide si le fichier manque."""
    if csv_file is None or not Path(csv_file).exists():
        return set()
    import csv
    rows = list(csv.DictReader(open(csv_file, encoding="utf-8")))
    by = {}
    for r in rows:
        key = (r["config"], r["instance"], r["forme"])
        same = r["sure"] == "True" and float(r["dh_lo"]) >= -30 and float(r["dh_hi"]) <= 30
        by.setdefault(key, {})[(int(r["graine"]), round(float(r["amplitude"]), 3))] = (r["sure"] == "True", same)
    out = set()
    for key, cond in by.items():
        if not cond.get((42, 0.01), (False, False))[0]:
            continue
        seeds = sum(cond.get((g, 0.01), (False, False))[1] for g in (0, 1, 2, 3, 4))
        amps = all(cond.get((42, a), (False, False))[1] for a in (0.003, 0.03))
        if seeds >= min_other_seeds and amps:
            out.add(key)
    return out


def model_label(r, short=True):
    labels = SHORT_LABELS if short else CONFIG_LABELS
    return f"{labels[r['config']]}, graine {r['instance'][-1]}"


def fr(x, fmt=".2f"):
    """Nombre au format français (virgule décimale)."""
    return format(x, fmt).replace(".", ",")


def gallery_figure(stats, strips, images, out_file, stable=frozenset()):
    selection = select_gallery(stats)
    img, gap, arrow, strip_h = 0.62, 0.05, 0.34, 0.09
    label_h, row_sp, block_sp, margin = 0.30, 0.24, 0.5, 0.35
    title_h, head_h, foot_h = 1.0, 0.62, 1.15
    noisy_w = N_EXAMPLES * img + (N_EXAMPLES - 1) * gap
    entry_w = img + arrow + noisy_w
    row_h = label_h + img + 0.05 + strip_h + row_sp
    n_rows = max(len(v) for v in selection.values())
    width = 2 * margin + 3 * entry_w + 2 * block_sp
    height = title_h + head_h + n_rows * row_h + foot_h
    c = Canvas(width, height)
    c.text(width / 2, 0.38, "Associations forme → couleur en traduction bruitée (attributs sans couleur)",
           ha="center", va="center", fontsize=15, weight="bold")
    c.text(width / 2, 0.72, "Traduction attributs → image, sans couleur dans les attributs. Pour chaque forme, "
           f"les {TOP_PER_SHAPE} associations les plus vives et constantes parmi les 15 modèles.",
           ha="center", va="center", fontsize=9.5, color="0.3")
    for b, shape in enumerate(SHAPES):
        x0 = margin + b * (entry_w + block_sp)
        c.text(x0 + entry_w / 2, title_h + 0.2, SHAPE_TITLES[shape], ha="center", va="center",
               fontsize=13, weight="bold")
        c.text(x0 + img / 2, title_h + head_h - 0.1, "sans bruit", ha="center", va="bottom",
               fontsize=7.5, color="0.35")
        c.text(x0 + img + arrow + noisy_w / 2, title_h + head_h - 0.1,
               "avec un bruit fixe de 1 % sur la forme", ha="center", va="bottom", fontsize=7.5,
               color="0.35")
        t = SHAPES.index(shape)
        for i, r in enumerate(selection[shape]):
            y = title_h + head_h + i * row_h
            c.text(x0, y + label_h / 2, model_label(r), ha="left", va="center", fontsize=8.5, weight="bold")
            mark = (" ★" if r["robuste"] else "") + (" ◆" if (r["config"], r["instance"], r["forme"]) in stable else "")
            c.text(x0 + entry_w - 0.24, y + label_h / 2, f"{r['nom']} · {100 * r['constance']:.0f} %{mark}",
                   ha="right", va="center", fontsize=8.5)
            c.rect(x0 + entry_w - 0.18, y + label_h / 2 - 0.09, 0.18, 0.18,
                   (r["R"] / 255, r["G"] / 255, r["B"] / 255))
            key = (r["config"], r["instance"])
            c.image(images[key + ("sans_bruit",)][t * N_EXAMPLES], x0, y + label_h, img)
            c.text(x0 + img + arrow / 2, y + label_h + img / 2, "→", ha="center", va="center", fontsize=13,
                   color="0.45")
            for e, im in enumerate(images[key + ("annexe_E",)][t * N_EXAMPLES:(t + 1) * N_EXAMPLES]):
                c.image(im, x0 + img + arrow + e * (img + gap), y + label_h, img)
            ys = y + label_h + img + 0.05
            c.strip(strips[key + ("sans_bruit", shape)], x0, ys, img, strip_h)
            c.strip(strips[key + ("annexe_E", shape)], x0 + img + arrow, ys, noisy_w, strip_h)
    foot = (f"Chaque ligne : la même description (forme, position, taille, orientation) traduite par un modèle ; "
            f"{N_EXAMPLES} exemples non triés du jeu d'évaluation, identiques pour tous les modèles.\n"
            f"Bande : {STRIP_N} des 51 200 traductions de la forme, tirées au hasard ; d'abord celles de la teinte "
            "dominante, puis les autres teintes, puis les grises. Pourcentage : part des traductions\n"
            "dont la teinte est à ±30° de la teinte dominante.   ★ : même couleur quelle que soit la forme vers "
            f"laquelle part le résidu (balayage, ε = {fr(SWEEP_EPS, 'g')}).\n"
            "◆ : même couleur pour au moins 4 graines du bruit sur 6 et aux trois amplitudes (0,003 ; 0,01 ; 0,03), "
            "constance sûre (borne basse de Wilson ≥ 60 %).\n"
            "Attr. Continus et Cycle-Complet : configurations Emphase Attributs Continus et Emphase Cycle-Complet. "
            "Emphase Traduction et Emphase Contrastive restent grises.")
    c.text(margin, height - foot_h + 0.14, foot, ha="left", va="top", fontsize=7.5, color="0.3", linespacing=1.5)
    c.fig.savefig(out_file, dpi=200)
    plt.close(c.fig)
    return selection


def palettes_figure(stats, strips, images, out_file):
    by_key = {(r["config"], r["instance"], r["forme"]): r for r in stats if r["condition"] == "annexe_E"}
    clean_s = [r["S_mediane"] for r in stats if r["condition"] == "sans_bruit"]
    configs = [c for c in CONFIG_LABELS if any(k[0] == c for k in by_key)]
    instances = sorted({k[1] for k in by_key})
    small, sgap, mgap, strip_h = 0.3, 0.02, 0.12, 0.07
    montage = 2 * small + sgap
    cell_w = 3 * montage + 2 * mgap
    text_h, col_sp, row_sp, left, margin = 0.5, 0.3, 0.16, 0.7, 0.3
    title_h, head_h, shape_h, foot_h = 0.9, 0.45, 0.22, 0.75
    width = margin + left + len(configs) * cell_w + (len(configs) - 1) * col_sp + margin
    height = title_h + head_h + shape_h + len(instances) * (montage + text_h + row_sp) + foot_h
    c = Canvas(width, height)
    c.text(width / 2, 0.35, "Palette de chaque modèle : la couleur imaginée pour chaque forme",
           ha="center", va="center", fontsize=14, weight="bold")
    c.text(width / 2, 0.66, "Traduction attributs → image avec un bruit fixe de 1 % sur la forme ; "
           "dans chaque case : losange, œuf, triangle (4 exemples chacun)",
           ha="center", va="center", fontsize=9, color="0.3")
    for ci, config in enumerate(configs):
        x0 = margin + left + ci * (cell_w + col_sp)
        c.text(x0 + cell_w / 2, title_h + head_h / 2, CONFIG_LABELS[config], ha="center", va="center",
               fontsize=10, weight="bold")
        for s, shape in enumerate(SHAPES):
            c.text(x0 + s * (montage + mgap) + montage / 2, title_h + head_h + shape_h / 2, SHAPE_NAMES[shape],
                   ha="center", va="center", fontsize=7.5, color="0.35")
        for ii, instance in enumerate(instances):
            y0 = title_h + head_h + shape_h + ii * (montage + text_h + row_sp)
            if ci == 0:
                c.text(margin, y0 + montage / 2, f"graine {instance[-1]}", ha="left", va="center",
                       fontsize=9.5, weight="bold")
            key = (config, instance, "annexe_E")
            if key not in images:
                continue
            for s, shape in enumerate(SHAPES):
                r = by_key[(config, instance, shape)]
                xs = x0 + s * (montage + mgap)
                for e, im in enumerate(images[key][s * N_EXAMPLES:(s + 1) * N_EXAMPLES]):
                    c.image(im, xs + (e % 2) * (small + sgap), y0 + (e // 2) * (small + sgap), small)
                c.strip(strips[key + (shape,)], xs, y0 + montage + 0.04, montage, strip_h)
                strong = r["constance"] >= MIN_CONSTANCY
                style = dict(ha="center", va="top", fontsize=6.5, color="0.1" if strong else "0.5")
                c.text(xs + montage / 2, y0 + montage + 0.17, r["nom"], weight="bold" if strong else "normal",
                       **style)
                c.text(xs + montage / 2, y0 + montage + 0.31, f"{100 * r['constance']:.0f} %", **style)
    foot = (f"Bande : {STRIP_N} des 51 200 traductions de la forme ; d'abord celles de la teinte dominante, puis les "
            "autres teintes, puis les grises. Pourcentage : part des traductions dont la teinte est à ±30°\n"
            "de la teinte dominante "
            f"(en gras si ≥ {100 * MIN_CONSTANCY:.0f} %) ; « gris » : moins de la moitié des traductions colorées ; "
            "« variable » : moins de la moitié des traductions colorées ont la teinte dominante.\n"
            f"Sans bruit, toutes ces traductions sont grises (saturation médiane {fr(min(clean_s))} à "
            f"{fr(max(clean_s))}). Les Modèles Complets (couleur donnée dans les attributs) ne sont pas concernés.")
    c.text(margin, height - foot_h + 0.12, foot, ha="left", va="top", fontsize=7.5, color="0.3", linespacing=1.5)
    c.fig.savefig(out_file, dpi=200)
    plt.close(c.fig)


# ----------------------------------------------------------------------------------------------
# Synthèse
# ----------------------------------------------------------------------------------------------
def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def write_synthesis(stats, selection, checks, out_file):
    noisy = [r for r in stats if r["condition"] == "annexe_E"]
    clean = [r for r in stats if r["condition"] == "sans_bruit"]
    lines = ["# Synesthésie artificielle : associations forme → couleur", "",
             "Chemin de traduction (attributs → GW → image) des 15 modèles sans couleur dans les attributs, "
             "51 200 traductions par forme et par modèle (évaluation `--hls --translation-only`). Couleur mesurée "
             "comme dans l'évaluation (barycentre du masque d'Otsu) et résumée dans le plan de chrominance "
             "(a = R − (G+B)/2, b = √3/2 (G − B), unités RGB 0-255) : **chroma systématique** = norme du vecteur "
             f"chrominance moyen (score de classement) ; **constance** = part des traductions colorées (chroma ≥ "
             f"{C_MIN:g}) dont la teinte est à ±{HUE_TOL:g}° de la teinte dominante ; **couleur** = médiane R, G, B "
             "des traductions constantes (« gris » : moins de la moitié des traductions colorées ; « variable » : "
             "moins de la moitié des traductions colorées ont la teinte dominante). **Robuste** (balayage, ε = "
             f"{SWEEP_EPS:g}) : la couleur est la même (chroma ≥ {ROBUST_CHROMA:g} et teintes à moins de "
             f"{ROBUST_HUE:g}°) que le résidu parte entièrement vers l'une ou vers l'autre des deux autres formes. "
             "Figures : `galerie.png` (les plus marquantes) et `palettes.png` (les 15 modèles).",
             "",
             f"## 1. Les plus marquantes (bruit de l'annexe E, {TOP_PER_SHAPE} par forme, constance ≥ "
             f"{100 * MIN_CONSTANCY:.0f} %)", ""]
    rows = []
    for shape in SHAPES:
        for r in sorted(selection[shape], key=lambda r: -r["chroma_systematique"]):
            rows.append([SHAPE_TITLES[shape], model_label(r, short=False), r["nom"],
                         f"rgb({r['R']:.0f}, {r['G']:.0f}, {r['B']:.0f})", f"{r['teinte']:.0f}°",
                         f"{100 * r['constance']:.0f} %", f"{100 * r['colorees']:.0f} %",
                         f"{r['chroma_systematique']:.0f}", f"{r['S_mediane']:.2f}",
                         "oui" if r["robuste"] else "non"])
    lines += [md_table(["Forme", "Modèle", "Couleur", "RGB", "Teinte", "Constance", "Colorées",
                        "Chroma syst.", "S médiane", "Robuste"], rows), ""]
    if checks and checks[2]:
        inside, expected, total = checks
        lines += [f"Exemples dessinés (non triés) des associations de constance ≥ {100 * MIN_CONSTANCY:.0f} % : "
                  f"{inside} sur {total} à ±{HUE_TOL:g}° de la teinte dominante (attendu d'après la constance : "
                  f"{expected:.0f}).", ""]
    lines += ["## 2. Palette de chaque modèle (bruit de l'annexe E)", "",
              "Couleur représentative et constance ; en gras, constance ≥ "
              f"{100 * MIN_CONSTANCY:.0f} %.", ""]
    rows = []
    for config in CONFIG_LABELS:
        for instance in sorted({r["instance"] for r in noisy if r["config"] == config}):
            row = [f"{CONFIG_LABELS[config]}, graine {instance[-1]}"]
            for shape in SHAPES:
                r = next(x for x in noisy if (x["config"], x["instance"], x["forme"]) == (config, instance, shape))
                cell = f"{r['nom']} {100 * r['constance']:.0f} %"
                row.append(f"**{cell}**" if r["constance"] >= MIN_CONSTANCY else cell)
            rows.append(row)
    lines += [md_table(["Modèle"] + [SHAPE_NAMES[s] for s in SHAPES], rows), ""]
    lines += ["## 3. Sans bruit", "",
              f"Chroma systématique au plus {max(r['chroma_systematique'] for r in clean):.0f} (contre jusqu'à "
              f"{max(r['chroma_systematique'] for r in noisy):.0f} avec le bruit), constance au plus "
              f"{100 * max(r['constance'] for r in clean):.1f} %, saturation médiane de "
              f"{min(r['S_mediane'] for r in clean):.2f} à {max(r['S_mediane'] for r in clean):.2f} : "
              "toutes les traductions sans bruit sont grises.", ""]
    lines += ["## 4. La couleur dépend-elle de la forme qui reçoit le résidu ? (balayage)", "",
              f"Teinte et chroma de la couleur moyenne (16 exemples du balayage) quand tout le résidu (ε = "
              f"{SWEEP_EPS:g}) part vers l'une ou l'autre des deux autres formes. Associations de constance ≥ "
              f"{100 * MIN_CONSTANCY:.0f} % avec le bruit de l'annexe E.", ""]
    rows = []
    for r in sorted((x for x in noisy if x["constance"] >= MIN_CONSTANCY),
                    key=lambda x: (SHAPES.index(x["forme"]), -x["chroma_systematique"])):
        rows.append([SHAPE_TITLES[r["forme"]], model_label(r, short=False), r["nom"],
                     f"vers {SHAPE_NAMES[r['residu_1_forme']]} : {r['residu_1_teinte']:.0f}° "
                     f"({r['residu_1_chroma']:.0f})",
                     f"vers {SHAPE_NAMES[r['residu_2_forme']]} : {r['residu_2_teinte']:.0f}° "
                     f"({r['residu_2_chroma']:.0f})", "oui" if r["robuste"] else "non"])
    lines += [md_table(["Forme", "Modèle", "Couleur (annexe E)", "Tout le résidu vers…", "… ou vers",
                        "Robuste"], rows), "", "Entre parenthèses : chroma de la couleur moyenne.", ""]
    trans = [r for r in noisy if not np.isnan(r["transmission_cycle_complet"])]
    if trans:
        lines += ["## 5. Cycle complet : la couleur d'entrée traverse-t-elle les attributs sans couleur ?", "",
                  "R̄ de l'écart de teinte entrée → sortie du cycle complet (image → GW → attributs → GW → image, "
                  "résultats du chapitre 3) : 1 = la teinte d'entrée est restituée, 0 = sans rapport.", ""]
        rows = []
        for config in CONFIG_LABELS:
            for instance in sorted({r["instance"] for r in trans if r["config"] == config}):
                vals = [next(x["transmission_cycle_complet"] for x in trans
                             if (x["config"], x["instance"], x["forme"]) == (config, instance, s)) for s in SHAPES]
                rows.append([f"{CONFIG_LABELS[config]}, graine {instance[-1]}"] + [f"{v:.2f}" for v in vals])
        lines += [md_table(["Modèle"] + [SHAPE_NAMES[s] for s in SHAPES], rows), ""]
    out_file.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=paths.RESULTS_DIR / "saturation" / "traduction_seule",
                        help="Évaluation --hls --translation-only (arbitrary/ et arbitrary_shape_noise_0.01/)")
    parser.add_argument("--chapter3-dir", type=Path, default=paths.RESULTS_DIR / "arbitrary",
                        help="Résultats du chapitre 3 (cycle complet) ; ignorés s'ils sont absents")
    parser.add_argument("--sweep-dir", type=Path, default=paths.RESULTS_DIR / "saturation" / "balayage")
    parser.add_argument("--dataset-csv", type=Path, default=paths.EVALSET_DIR / "attributes.csv")
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "saturation" / "synesthesie")
    parser.add_argument("--stats-only", action="store_true", help="Tableau seulement (pas de checkpoint chargé)")
    parser.add_argument("--recompute", action="store_true", help="Ignorer les fichiers intermédiaires en cache")
    parser.add_argument("--cpu", action="store_true", help="Traductions sur CPU même si un GPU est disponible")
    parser.add_argument("--noise-robustness", type=Path,
                        default=paths.RESULTS_DIR / "saturation" / "robustesse" / "bruit_conditions.csv",
                        help="Résultats de saturation_robustness.py : marque ◆ les associations stables "
                             "selon la graine et l'amplitude du bruit (ignoré si absent)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stats_file, strips_file = args.out / "associations.csv", args.out / "bandes.npz"
    if stats_file.exists() and strips_file.exists() and not args.recompute:
        print(f"Statistiques : {stats_file}")
        z = np.load(strips_file)
        strips = {tuple(k.split("|")): z[k] for k in z.files}
    else:
        rows, strips = collect_stats(args.results_dir, args.chapter3_dir, args.sweep_dir)
        if not rows:
            raise SystemExit(f"Aucun résultat dans {args.results_dir}")
        write_csv(rows, stats_file)
        np.savez_compressed(strips_file, **{"|".join(k): v for k, v in strips.items()})
    stats = read_csv(stats_file)
    selection, checks = select_gallery(stats), None
    if not args.stats_only:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        cache = args.out / "exemples.npz"
        if args.recompute and cache.exists():
            cache.unlink()
        images = generate_examples(pick_examples(args.dataset_csv), device, cache)
        checks = check_examples(stats, images)
        selection = gallery_figure(stats, strips, images, args.out / "galerie.png",
                                   noise_robust(args.noise_robustness))
        palettes_figure(stats, strips, images, args.out / "palettes.png")
    write_synthesis(stats, selection, checks, args.out / "synthese.md")
    print(f"Écrit dans {args.out}")


if __name__ == "__main__":
    main()
