"""Robustesse (étape 4 du protocole « saturation ») : extraction de la couleur et bruit de l'annexe E.

Chemin de traduction (attributs -> GW -> image) des 19 modèles, sur un sous-ensemble tiré du jeu
d'évaluation (--per-shape attributs par forme, graine 0), dans 19 conditions :
  - sans bruit ;
  - bruit de l'annexe E pour 6 graines du vecteur de bruit (42, celle du mémoire, puis 0 à 4) et
    3 amplitudes (0,003 ; 0,01, celle du mémoire ; 0,03). Comme dans _process_translation_path_batch
    (SSD_utils.py) : un seul vecteur randn(3) * amplitude pour toutes les images, ajouté au one-hot,
    borné à 1e-8 puis renormalisé.
La couleur de chaque image traduite est extraite de deux façons (même masque d'Otsu) :
  - « barycentre » : 3 x 3 pixels au barycentre du masque (extract_shape_color, comme l'évaluation) ;
  - « interieur » : le pixel le plus intérieur du masque (maximum de la transformée de distance),
    moins sensible aux bords flous.
Analyses (volets 1 et 2) :
  - médianes de S ; régularité (S diffère entre formes, entre orientations) et arbitraire (S d'une
    même forme diffère entre instances) avec le test du mémoire et le critère d'effet (KS brut
    significatif et |delta de Cliff| >= 0,147) ; constance des associations forme -> couleur
    (synesthesia_figure.summarize) ; comparaison des deux extractions ;
  - pour chaque graine et amplitude : associations de constance >= 60 %, teinte dominante ; la teinte
    d'une forme dépend-elle de la graine du bruit ? Quelles associations du mémoire (graine 42,
    0,01) survivent aux autres graines et amplitudes ?

    python experiments/saturation_robustness.py                 # calcul (reprend) + analyse
    python experiments/saturation_robustness.py --analyze-only
    python experiments/saturation_robustness.py --max-models 1 --per-shape 64   # essai rapide

Sorties dans results/saturation/robustesse/extraction_bruit/ : donnees/<modèle>.npz, synthese.md,
tableaux CSV, figures.
"""

import argparse
import colorsys
import itertools
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401
from saturation_analysis import SMALL_EFFECT, compare  # noqa: E402
from shape_noise_sweep import CONFIG_LABELS, SHAPES, load_model, md_table, thesis_models  # noqa: E402
from synesthesia_figure import (C_MIN, HUE_TOL, MIN_CONSTANCY, SHAPE_NAMES, circ_diff, opponent,  # noqa: E402
                                summarize, translate)
from SSD_utils import (extract_shape_color, normalize_position, normalize_rotation,  # noqa: E402
                       normalize_size, segment_shape)

NOISE_SEEDS = (42, 0, 1, 2, 3, 4)
AMPLITUDES = (0.003, 0.01, 0.03)
THESIS_SEED, THESIS_AMP = 42, 0.01
EXTRACTIONS = ("barycentre", "interieur")
ALPHA, NUM_BINS = 0.05, 15
N_ORIENT = 16


def conditions():
    """[(nom, graine, amplitude)] ; graine None = sans bruit."""
    out = [("sans_bruit", None, 0.0)]
    out += [(f"g{s}_a{a:g}", s, a) for a in AMPLITUDES for s in NOISE_SEEDS]
    return out


def noise_vector(seed, amp):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (torch.randn(3, generator=generator) * amp).numpy()


def noisy_one_hots(shapes, seed, amp):
    oh = np.eye(3)[shapes]
    if seed is None:
        return oh.astype(np.float32)
    oh = np.clip(oh + noise_vector(seed, amp), 1e-8, None)
    return (oh / oh.sum(axis=1, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Entrées et extraction
# ----------------------------------------------------------------------------------------------
def select_attributes(per_shape, seed=0):
    df = pd.read_csv(paths.EVALSET_DIR / "attributes.csv")
    sub = df.groupby("class", group_keys=False).sample(n=per_shape, random_state=seed).sort_index()
    import ast
    pos = np.array([ast.literal_eval(s) for s in sub["location"]], dtype=np.float64)
    rot = sub["rotation"].to_numpy(np.float64)
    size = sub["size"].to_numpy(np.float64)
    attrs = np.column_stack([np.array([normalize_position(list(p)) for p in pos]),
                             [normalize_size(s) for s in size],
                             np.array([normalize_rotation(r) for r in rot])]).astype(np.float32)
    orient = np.round(rot / (2 * np.pi / N_ORIENT)).astype(int) % N_ORIENT
    return dict(forme=sub["class"].to_numpy(), taille=size, orientation=orient, attrs=attrs)


def innermost_color(img, mask):
    """Couleur du pixel le plus intérieur du masque (maximum de la transformée de distance)."""
    if mask.sum() < 1:
        return img.mean(axis=(0, 1))
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return img[y, x]


def extract(images):
    """(N, 2, 3) : couleur RGB 0-255, barycentre puis intérieur."""
    out = np.empty((len(images), 2, 3), dtype=np.float32)
    for i, img in enumerate(images):
        mask = segment_shape(img)
        out[i, 0] = extract_shape_color(img, mask, True) * 255
        out[i, 1] = innermost_color(img, mask) * 255
    return out


def compute(models, inputs, data_dir, device, batch_size):
    data_dir.mkdir(parents=True, exist_ok=True)
    conds = conditions()
    for exp_name, instance, ckpt, colored in models:
        f = data_dir / f"{exp_name}_{instance}.npz"
        if f.exists():
            print(f"Déjà calculé : {f.name}")
            continue
        t0 = time.time()
        gw = load_model(ckpt, colored, device)
        attrs = inputs["attrs"]
        if colored:  # Modèles Complets : couleur donnée (palette du jeu de test, une par image)
            from SSD_utils import generate_fixed_colors
            palette, _ = generate_fixed_colors(25)
            rgb_in = palette[np.arange(len(attrs)) % 25] / 127.5 - 1
            attrs = np.concatenate([attrs, rgb_in], axis=1).astype(np.float32)
        rgb = np.empty((len(conds), len(attrs), 2, 3), dtype=np.float32)
        for c, (_, seed, amp) in enumerate(conds):
            oh = noisy_one_hots(inputs["forme"], seed, amp)
            for s in range(0, len(attrs), batch_size):
                imgs = translate(gw, oh[s:s + batch_size], attrs[s:s + batch_size], device)
                rgb[c, s:s + batch_size] = extract(imgs)
        del gw
        np.savez_compressed(f, rgb=rgb, config=exp_name, instance=instance, colored=colored)
        print(f"{exp_name}/{instance} : {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------
# Analyse
# ----------------------------------------------------------------------------------------------
def saturation(rgb):
    """S (HLS) sur 0-255, comme les distributions de l'évaluation."""
    x = np.clip(rgb / 255, 0, 1)
    return np.array([colorsys.rgb_to_hls(*p)[2] for p in x]) * 255


def load_all(data_dir):
    out = {}
    for f in sorted(data_dir.glob("*.npz")):
        z = np.load(f)
        out[(str(z["config"]), str(z["instance"]))] = z["rgb"]
    return out


def model_order(key):
    order = list(CONFIG_LABELS)
    return order.index(key[0]), key[1] == "control", key[1]


def label(key):
    return f"{CONFIG_LABELS[key[0]]}, {'contrôle' if key[1] == 'control' else 'graine ' + key[1][-1]}"


def regularity(s, inputs, attr):
    """Paires de catégories (forme ou orientation) : (n paires, n test du mémoire, n critère d'effet)."""
    cats = inputs["forme"] if attr == "forme" else inputs["orientation"]
    n = n_mem = n_eff = 0
    for c1, c2 in itertools.combinations(np.unique(cats), 2):
        st = compare(s[cats == c1], s[cats == c2], "S", NUM_BINS)
        if st is None:
            continue
        n += 1
        n_mem += st["p_memoire"] < ALPHA
        n_eff += st["p_brut"] < ALPHA and abs(st["cliff_delta"]) >= SMALL_EFFECT
    return n, n_mem, n_eff


def volet1(models, inputs, conds):
    """Médianes de S, régularité, arbitraire et constance, pour les deux extractions."""
    idx = {name: i for i, (name, _, _) in enumerate(conds)}
    rows, assoc, arb = [], [], []
    for key in sorted(models, key=model_order):
        for cond in ("sans_bruit", f"g{THESIS_SEED}_a{THESIS_AMP:g}"):
            for e, ext in enumerate(EXTRACTIONS):
                rgb = models[key][idx[cond], :, e]
                s = saturation(rgb)
                n_f, mem_f, eff_f = regularity(s, inputs, "forme")
                n_o, mem_o, eff_o = regularity(s, inputs, "orientation")
                rows.append(dict(config=key[0], instance=key[1], condition=cond, extraction=ext,
                                 S_mediane=np.median(s) / 255, forme_memoire=mem_f, forme_effet=eff_f,
                                 orientation_memoire=mem_o, orientation_effet=eff_o, n_paires_orientation=n_o))
                if key[1] != "control":
                    for t, shape in enumerate(SHAPES):
                        sel = inputs["forme"] == t
                        r = summarize(rgb[sel], s[sel])
                        assoc.append(dict(config=key[0], instance=key[1], condition=cond, extraction=ext,
                                          forme=shape, constance=r["constance"], colorees=r["colorees"],
                                          teinte_dominante=r["teinte_dominante"], nom=r["nom"],
                                          S_mediane=r["S_mediane"], R=r["R"], G=r["G"], B=r["B"]))
    # arbitraire : S d'une même forme entre instances d'une configuration
    for exp_name in CONFIG_LABELS:
        inst = sorted(k for k in models if k[0] == exp_name and k[1] != "control")
        for cond in ("sans_bruit", f"g{THESIS_SEED}_a{THESIS_AMP:g}"):
            for e, ext in enumerate(EXTRACTIONS):
                for t, shape in enumerate(SHAPES):
                    sel = inputs["forme"] == t
                    for k1, k2 in itertools.combinations(inst, 2):
                        st = compare(saturation(models[k1][idx[cond], sel, e]),
                                     saturation(models[k2][idx[cond], sel, e]), "S", NUM_BINS)
                        arb.append(dict(config=exp_name, instance_1=k1[1], instance_2=k2[1], condition=cond,
                                        extraction=ext, forme=shape, **st))
    return pd.DataFrame(rows), pd.DataFrame(assoc), pd.DataFrame(arb)


def volet2(models, inputs, conds):
    """Constance et teinte dominante pour chaque graine et amplitude du bruit (extraction du mémoire)."""
    rows = []
    for key in sorted(models, key=model_order):
        if key[1] == "control":
            continue
        for c, (name, seed, amp) in enumerate(conds):
            if seed is None:
                continue
            rgb = models[key][c, :, 0]
            for t, shape in enumerate(SHAPES):
                sel = inputs["forme"] == t
                s = saturation(rgb[sel])
                r = summarize(rgb[sel], s)
                rows.append(dict(config=key[0], instance=key[1], graine=seed, amplitude=amp, forme=shape,
                                 constance=r["constance"], colorees=r["colorees"],
                                 teinte_dominante=r["teinte_dominante"], nom=r["nom"], S_mediane=r["S_mediane"],
                                 R=r["R"], G=r["G"], B=r["B"]))
    return pd.DataFrame(rows)


def circ_spread(hues):
    """R̄ des teintes (degrés)."""
    if len(hues) == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * np.radians(hues)))))


def fr(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace(".", ",")


def recipient(seed, amp, t):
    """Forme non cible qui reçoit le plus de masse (vecteur de bruit borné puis renormalisé)."""
    oh = noisy_one_hots(np.array([t]), seed, amp)[0]
    others = [c for c in range(3) if c != t]
    return SHAPES[max(others, key=lambda c: oh[c])]


def write_outputs(models, inputs, out_dir, args):
    conds = conditions()
    v1, assoc, arb = volet1(models, inputs, conds)
    v2 = volet2(models, inputs, conds)
    v1.to_csv(out_dir / "extraction_S.csv", index=False)
    assoc.to_csv(out_dir / "extraction_associations.csv", index=False)
    arb.to_csv(out_dir / "extraction_arbitraire.csv", index=False)
    v2.to_csv(out_dir / "bruit_associations.csv", index=False)
    noisy = f"g{THESIS_SEED}_a{THESIS_AMP:g}"
    L = [f"# Robustesse : extraction de la couleur et bruit de l'annexe E\n",
         f"Chemin de traduction, {len(inputs['forme'])} attributs tirés du jeu d'évaluation ({args.per_shape} par "
         f"forme, graine 0), {len(models)} modèles. Associations : constance = part des traductions à ±{HUE_TOL:.0f}° "
         f"de la teinte dominante (chroma ≥ {C_MIN:.0f}) ; « association » si constance ≥ {MIN_CONSTANCY:.0%}.\n"]

    # --- Volet 1 -----------------------------------------------------------------------------
    L.append("## 1. Autre extraction : pixel le plus intérieur du masque\n")
    L.append("### Médiane de S (moyenne des instances ; contrôles : les 4 Modèles Complets)\n")
    rows = []
    for exp_name in CONFIG_LABELS:
        for ctrl in (False, True):
            sub = v1[(v1.config == exp_name) & ((v1.instance == "control") == ctrl)]
            if sub.empty:
                continue
            vals = [fr(sub[(sub.condition == c) & (sub.extraction == e)].S_mediane.mean())
                    for c in ("sans_bruit", noisy) for e in EXTRACTIONS]
            rows.append([CONFIG_LABELS[exp_name] + (", contrôle" if ctrl else "")] + vals)
    L.append(md_table(["", "sans bruit, barycentre", "sans bruit, intérieur", "annexe E, barycentre",
                       "annexe E, intérieur"], rows) + "\n")
    L.append("### Régularité de S : paires qui diffèrent (test du mémoire / critère d'effet), somme sur les instances\n")
    rows = []
    for exp_name in CONFIG_LABELS:
        for ctrl in (False, True):
            sub = v1[(v1.config == exp_name) & ((v1.instance == "control") == ctrl)]
            if sub.empty:
                continue
            n_inst = sub.instance.nunique()
            cells = []
            for c in ("sans_bruit", noisy):
                for e in EXTRACTIONS:
                    x = sub[(sub.condition == c) & (sub.extraction == e)]
                    cells.append(f"F {x.forme_memoire.sum()}/{x.forme_effet.sum()} sur {3 * n_inst} ; "
                                 f"O {x.orientation_memoire.sum()}/{x.orientation_effet.sum()} sur "
                                 f"{x.n_paires_orientation.sum()}")
            rows.append([CONFIG_LABELS[exp_name] + (", contrôle" if ctrl else "")] + cells)
    L.append(md_table(["", "sans bruit, barycentre", "sans bruit, intérieur", "annexe E, barycentre",
                       "annexe E, intérieur"], rows) + "\n")
    L.append("### Arbitraire de S : paires d'instances qui diffèrent pour une même forme (mémoire / effet), "
             "|δ| maximal\n")
    rows = []
    for exp_name in CONFIG_LABELS:
        cells = []
        for c in ("sans_bruit", noisy):
            for e in EXTRACTIONS:
                x = arb[(arb.config == exp_name) & (arb.condition == c) & (arb.extraction == e)]
                eff = ((x.p_brut < ALPHA) & (x.cliff_delta.abs() >= SMALL_EFFECT)).sum()
                cells.append(f"{(x.p_memoire < ALPHA).sum()}/{eff} sur {len(x)} (|δ| ≤ {fr(x.cliff_delta.abs().max())})")
        rows.append([CONFIG_LABELS[exp_name]] + cells)
    L.append(md_table(["", "sans bruit, barycentre", "sans bruit, intérieur", "annexe E, barycentre",
                       "annexe E, intérieur"], rows) + "\n")
    L.append("### Associations forme → couleur (annexe E), selon l'extraction\n")
    a = assoc[assoc.condition == noisy]
    piv = a.pivot_table(index=["config", "instance", "forme"], columns="extraction",
                        values=["constance", "teinte_dominante"], aggfunc="first")
    rows, n_b, n_i, n_both, d_hue = [], 0, 0, 0, []
    for (c, i, f), r in piv.iterrows():
        cb, ci = r[("constance", "barycentre")], r[("constance", "interieur")]
        hb, hi = r[("teinte_dominante", "barycentre")], r[("teinte_dominante", "interieur")]
        n_b += cb >= MIN_CONSTANCY
        n_i += ci >= MIN_CONSTANCY
        if cb >= MIN_CONSTANCY or ci >= MIN_CONSTANCY:
            both = cb >= MIN_CONSTANCY and ci >= MIN_CONSTANCY
            n_both += both
            d_hue.append(float(circ_diff(hb, hi)))
            name_i = a[(a.config == c) & (a.instance == i) & (a.forme == f) & (a.extraction == "interieur")].nom.iloc[0]
            name_b = a[(a.config == c) & (a.instance == i) & (a.forme == f) & (a.extraction == "barycentre")].nom.iloc[0]
            rows.append([label((c, i)), SHAPE_NAMES[f], f"{name_b}, {cb:.0%}", f"{name_i}, {ci:.0%}",
                         f"{circ_diff(hb, hi):.0f}°"])
    L.append(md_table(["Modèle", "forme", "barycentre", "intérieur", "écart de teinte dominante"], rows) + "\n")
    L.append(f"Associations (constance ≥ {MIN_CONSTANCY:.0%}) : {n_b} avec le barycentre, {n_i} avec le pixel "
             f"intérieur, {n_both} dans les deux ; écart médian de teinte dominante {np.median(d_hue):.0f}° "
             f"(max {max(d_hue):.0f}°).\n")

    # --- Volet 2 -----------------------------------------------------------------------------
    L.append("## 2. Graines et amplitudes du bruit de l'annexe E\n")
    vecs = [[str(s)] + [" ; ".join(fr(v, 4) for v in noise_vector(s, 0.01))] +
            [", ".join(f"{SHAPE_NAMES[sh]} → {SHAPE_NAMES[recipient(s, 0.01, t)]}" for t, sh in enumerate(SHAPES))]
            for s in NOISE_SEEDS]
    L.append("Vecteurs de bruit (amplitude 0,01 ; losange, œuf, triangle) et forme qui reçoit le plus de "
             "masse résiduelle :\n")
    L.append(md_table(["graine", "bruit", "résidu principal"], vecs) + "\n")
    L.append("### Médiane de S selon la graine et l'amplitude (moyenne sur instances et formes)\n")
    rows = []
    for exp_name in CONFIG_LABELS:
        x = v2[v2.config == exp_name]
        rows.append([CONFIG_LABELS[exp_name]] + [fr(x[(x.graine == s) & (x.amplitude == amp)].S_mediane.mean())
                                                 for amp in AMPLITUDES for s in (42, 0, 1, 2, 3, 4)][:0] +
                    [f"{fr(x[x.amplitude == amp].groupby('graine').S_mediane.mean().min())} à "
                     f"{fr(x[x.amplitude == amp].groupby('graine').S_mediane.mean().max())} "
                     f"(g42 : {fr(x[(x.amplitude == amp) & (x.graine == 42)].S_mediane.mean())})"
                     for amp in AMPLITUDES])
    L.append(md_table(["", *[f"amplitude {fr(a, 3)} (min à max sur 6 graines)" for a in AMPLITUDES]], rows) + "\n")
    L.append("### Nombre d'associations (constance ≥ 60 %) par graine et amplitude, 15 modèles\n")
    rows = []
    for amp in AMPLITUDES:
        x = v2[v2.amplitude == amp]
        rows.append([fr(amp, 3)] + [int((x[x.graine == s].constance >= MIN_CONSTANCY).sum()) for s in NOISE_SEEDS])
    L.append(md_table(["amplitude", *[f"graine {s}" for s in NOISE_SEEDS]], rows) + "\n")
    # dépendance à la graine : pour chaque modèle × forme, teintes dominantes des graines où la forme est colorée
    L.append("### La couleur dépend-elle de la graine ? (amplitude 0,01)\n")
    x = v2[v2.amplitude == THESIS_AMP]
    rows, rbars, same_recipient = [], [], []
    for (c, i, f), g in x.groupby(["config", "instance", "forme"]):
        col = g[g.constance >= MIN_CONSTANCY]
        if len(col) < 2:
            continue
        rbar = circ_spread(col.teinte_dominante.to_numpy())
        rbars.append(rbar)
        # même forme receveuse -> même couleur ?
        by_rec = {}
        for _, r in col.iterrows():
            by_rec.setdefault(recipient(int(r.graine), THESIS_AMP, SHAPES.index(f)), []).append(r.teinte_dominante)
        within = [circ_spread(np.array(h)) for h in by_rec.values() if len(h) >= 2]
        same_recipient += within
        rows.append([label((c, i)), SHAPE_NAMES[f], f"{len(col)}/6",
                     ", ".join(f"g{int(r.graine)} {r.nom} ({r.teinte_dominante:.0f}°)" for _, r in col.iterrows()),
                     fr(rbar)])
    L.append(md_table(["Modèle", "forme", "graines avec association", "couleurs", "R̄ des teintes"], rows) + "\n")
    L.append(f"R̄ des teintes dominantes entre graines (1 = même couleur quelle que soit la graine) : médiane "
             f"{fr(np.median(rbars)) if rbars else '–'} sur {len(rbars)} couples modèle × forme ; entre graines dont "
             f"le résidu part vers la même forme : médiane {fr(np.median(same_recipient)) if same_recipient else '–'} "
             f"({len(same_recipient)} groupes).\n")
    # survie des associations du mémoire
    L.append("### Associations du mémoire (graine 42, 0,01) : survie aux autres graines et amplitudes\n")
    ref = v2[(v2.graine == THESIS_SEED) & (v2.amplitude == THESIS_AMP) & (v2.constance >= MIN_CONSTANCY)]
    rows, survive_all = [], 0
    for _, r in ref.iterrows():
        g = v2[(v2.config == r.config) & (v2.instance == r.instance) & (v2.forme == r.forme)]
        ok = (g.constance >= MIN_CONSTANCY) & (circ_diff(g.teinte_dominante, r.teinte_dominante) <= HUE_TOL)
        n_seed = int(ok[(g.amplitude == THESIS_AMP) & (g.graine != THESIS_SEED)].sum())
        n_amp = int(ok[(g.graine == THESIS_SEED) & (g.amplitude != THESIS_AMP)].sum())
        n_all = int(ok.sum())
        survive_all += n_seed == len(NOISE_SEEDS) - 1
        rows.append([label((r.config, r.instance)), SHAPE_NAMES[r.forme], f"{r.nom} ({r.constance:.0%})",
                     f"{n_seed}/5", f"{n_amp}/2", f"{n_all}/18"])
    L.append(md_table(["Modèle", "forme", "couleur (g42, 0,01)", "autres graines (0,01)",
                       "autres amplitudes (g42)", "toutes conditions"], rows) + "\n")
    L.append(f"{len(ref)} associations à la graine 42 ; {survive_all} gardent leur couleur avec les 5 autres "
             "graines.\n")
    (out_dir / "synthese.md").write_text("\n".join(L))
    figure_seeds(v2, out_dir / "teintes_par_graine.png")
    print("\n".join(L))


def figure_seeds(v2, out_file):
    """Couleur représentative par modèle × forme (lignes) et graine × amplitude (colonnes)."""
    x = v2.copy()
    x["cle"] = [(c, i, f) for c, i, f in zip(x.config, x.instance, x.forme)]
    keys = sorted(x.cle.unique(), key=lambda k: (model_order(k[:2]), SHAPES.index(k[2])))
    cols = [(a, s) for a in AMPLITUDES for s in NOISE_SEEDS]
    fig, ax = plt.subplots(figsize=(10, 0.22 * len(keys) + 1.5))
    for r, k in enumerate(keys):
        for c, (a, s) in enumerate(cols):
            row = x[(x.cle == k) & (x.amplitude == a) & (x.graine == s)].iloc[0]
            color = np.array([row.R, row.G, row.B]) / 255
            ax.add_patch(plt.Rectangle((c, r), 0.95, 0.9, color=np.clip(color, 0, 1)))
            if row.constance >= MIN_CONSTANCY:
                ax.plot(c + 0.47, r + 0.45, "o", ms=2.5, color="white", mec="black", mew=0.4)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(keys), 0)
    ax.set_yticks(np.arange(len(keys)) + 0.45)
    ax.set_yticklabels([f"{label(k[:2])}, {SHAPE_NAMES[k[2]]}" for k in keys], fontsize=5)
    ax.set_xticks(np.arange(len(cols)) + 0.47)
    ax.set_xticklabels([f"g{s}\n{a:g}" for a, s in cols], fontsize=6)
    ax.set_title("Couleur représentative de la traduction selon la graine et l'amplitude du bruit\n"
                 "(point : association de constance ≥ 60 %)", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--per-shape", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "saturation" / "robustesse" / "extraction_bruit")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = select_attributes(args.per_shape)
    models = thesis_models(args.experiment)[:args.max_models]
    if not args.analyze_only:
        compute(models, inputs, args.out / "donnees", device, args.batch_size)
    write_outputs(load_all(args.out / "donnees"), inputs, args.out, args)


if __name__ == "__main__":
    main()
