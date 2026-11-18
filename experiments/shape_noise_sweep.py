"""Balayage du bruit sur les attributs de forme (version courte de l'étape causale de l'annexe E).

Sur un petit sous-ensemble d'attributs (3 formes x 4 orientations x 2 tailles x 2 positions = 48
exemples), on perturbe l'entrée du chemin de traduction (attributs -> GW -> image) et on mesure la
couleur reconstruite (même masque d'Otsu et même barycentre que l'évaluation) :
  - forme : one-hot + eps * u, u dans le plan tangent au simplexe (somme nulle), 36 directions
    régulièrement espacées x 9 amplitudes eps de 0,001 à 0,1 (échelle log). Pas de projection : une
    direction est dite « admissible » si le vecteur reste dans le simplexe (composantes >= 0) ;
  - témoin 1 : même perturbation sur les 5 attributs continus (36 directions aléatoires de R^5) ;
  - témoin 2 : les Modèles Complets (couleur donnée dans les attributs) ;
  - repère : le bruit de l'annexe E (graine 42, x 0,01, renormalisé).
À chaque perturbation, l'image traduite est ré-encodée (image -> GW -> attributs) pour vérifier
que la forme reste reconnue : l'effet doit porter sur la couleur, pas sur l'identité de la forme.

Base du plan tangent pour une forme cible t (autres formes j < k) :
  e1 = (e_j - e_k) / sqrt(2)          (répartition de la masse entre les deux autres formes)
  e2 = (2 e_t - e_j - e_k) / sqrt(6)  (masse ajoutée à la forme cible)
  u(theta) = cos(theta) e1 + sin(theta) e2 ; admissible pour theta entre 240° et 300°.

    python experiments/shape_noise_sweep.py                  # calcul (reprend où il s'est arrêté) + analyse
    python experiments/shape_noise_sweep.py --analyze-only
    python experiments/shape_noise_sweep.py --experiment high_cycles --max-models 1   # essai rapide

Sorties dans results/saturation/balayage/ : donnees/<modèle>.npz, synthese.md, figures/.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401
from SSD_utils import (extract_shape_color, generate_fixed_colors, normalize_position,  # noqa: E402
                       normalize_rotation, normalize_size, segment_shape)

SHAPES = ["diamond", "egg", "triangle"]
CONFIG_LABELS = {"base_params": "Référence", "high_translation": "Emphase Traduction",
                 "high_contrastive": "Emphase Contrastive", "high_cycles": "Emphase Cycle-Complet",
                 "high_alpha": "Emphase Attributs Continus"}
N_DIRECTIONS = 36
EPSILONS = np.logspace(-3, -1, 9)
ROTATIONS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
SIZES = [9, 12]
POSITIONS = [(13, 13), (18, 18)]
N_COLORS = 25  # palette du jeu de test (Modèles Complets)
KIND_BASE, KIND_SHAPE, KIND_CONT, KIND_ANNEX_E = 0, 1, 2, 3


# ----------------------------------------------------------------------------------------------
# Modèles
# ----------------------------------------------------------------------------------------------
def thesis_models(experiments=None, include_controls=True):
    """[(config, instance, checkpoint, avec_couleurs)] pour les checkpoints présents."""
    manifest = yaml.safe_load(open(paths.THESIS_RUNS_FILE))["experiments"]
    out = []
    for exp_name, exp in manifest.items():
        if experiments and exp_name not in experiments:
            continue
        entries = [(run, False) for run in exp["runs"]]
        if include_controls and exp.get("control"):
            entries.append((exp["control"], True))
        for run, colored in entries:
            candidates = [paths.gw_checkpoint(run["original_run_name"], run.get("original_ckpt", run["ckpt"])),
                          paths.gw_checkpoint(run["run_dir"], run["ckpt"])]
            ckpt = next((c for c in candidates if c.exists()), None)
            if ckpt is None:
                print(f"Absent, ignoré : {exp_name}/{run['run_id']} ({candidates[0]})")
                continue
            out.append((exp_name, "control" if colored else run["run_id"], ckpt, colored))
    return out


def load_model(ckpt, colored, device):
    from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig
    from SSD_eval_regularity import load_global_workspace
    config = {
        "domains": [
            LoadedDomainConfig(domain_type=DomainModuleVariant.v_latents, checkpoint_path=paths.DOMAIN_V_CKPT),
            LoadedDomainConfig(domain_type=DomainModuleVariant.attr_legacy if colored
                               else DomainModuleVariant.attr_legacy_no_color,
                               checkpoint_path=paths.DOMAIN_ATTR_CKPT, args={"temperature": 1, "alpha": 1}),
        ],
        "global_workspace": {"latent_dim": 12, "encoders": {"hidden_dim": 256, "n_layers": 2},
                             "decoders": {"hidden_dim": 256, "n_layers": 2}},
    }
    gw = load_global_workspace(Path(ckpt), config)
    return gw.to(device).eval()


# ----------------------------------------------------------------------------------------------
# Entrées perturbées
# ----------------------------------------------------------------------------------------------
def base_samples():
    """48 exemples : forme, orientation, taille, position, indice de couleur (Modèles Complets)."""
    out = []
    for i, (t, rot, size, pos) in enumerate(itertools.product(range(3), ROTATIONS, SIZES, POSITIONS)):
        out.append(dict(shape=t, rotation=rot, size=size, position=pos, color_index=(7 * i) % N_COLORS))
    return out


def tangent_basis(t):
    j, k = [c for c in range(3) if c != t]
    e1, e2 = np.zeros(3), np.zeros(3)
    e1[j], e1[k] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    e2[t], e2[j], e2[k] = 2 / np.sqrt(6), -1 / np.sqrt(6), -1 / np.sqrt(6)
    return e1, e2


def annex_e_noise():
    generator = torch.Generator()
    generator.manual_seed(42)  # comme _process_translation_path_batch (SSD_utils.py)
    return (torch.randn(3, generator=generator) * 0.01).numpy()


def build_inputs(colored):
    """Toutes les entrées du balayage : one-hot (N, 3), attributs (N, 5 ou 8) et métadonnées."""
    rgb_palette, _ = generate_fixed_colors(N_COLORS)
    rng = np.random.default_rng(0)
    cont_dirs = rng.normal(size=(N_DIRECTIONS, 5))
    cont_dirs /= np.linalg.norm(cont_dirs, axis=1, keepdims=True)
    thetas = 2 * np.pi * np.arange(N_DIRECTIONS) / N_DIRECTIONS
    noise_e = annex_e_noise()
    one_hots, attrs, meta = [], [], []
    for b, s in enumerate(base_samples()):
        t = s["shape"]
        onehot = np.eye(3)[t]
        px, py = normalize_position(list(s["position"]))
        rx, ry = normalize_rotation(s["rotation"])
        cont = np.array([px, py, normalize_size(s["size"]), rx, ry])
        color = rgb_palette[s["color_index"]] / 127.5 - 1
        e1, e2 = tangent_basis(t)

        def add(oh, ct, kind, direction=-1, theta=np.nan, eps=0.0):
            one_hots.append(oh)
            attrs.append(np.concatenate([ct, color]) if colored else ct)
            meta.append((b, t, kind, direction, theta, eps, float(np.all(oh >= 0))))

        add(onehot, cont, KIND_BASE)
        for m, eps in itertools.product(range(N_DIRECTIONS), EPSILONS):
            u = np.cos(thetas[m]) * e1 + np.sin(thetas[m]) * e2
            add(onehot + eps * u, cont, KIND_SHAPE, m, thetas[m], eps)
            add(onehot, cont + eps * cont_dirs[m], KIND_CONT, m, np.nan, eps)
        noisy = np.clip(onehot + noise_e, 1e-8, None)
        add(noisy / noisy.sum(), cont, KIND_ANNEX_E, -1, np.nan, float(np.linalg.norm(noisy / noisy.sum() - onehot)))
    meta = np.array(meta, dtype=float)
    return (np.array(one_hots, dtype=np.float32), np.array(attrs, dtype=np.float32), meta)


# ----------------------------------------------------------------------------------------------
# Passage dans le modèle et mesure de la couleur
# ----------------------------------------------------------------------------------------------
def run_model(gw, one_hots, attrs, device, batch_size=2048):
    from shimmer.modules.selection import SingleDomainSelection
    selection = SingleDomainSelection().to(device)
    colors_bary, colors_mask, areas, decoded = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(one_hots), batch_size):
            oh = torch.from_numpy(one_hots[start:start + batch_size]).to(device)
            at = torch.from_numpy(attrs[start:start + batch_size]).to(device)
            # Traduction : attributs -> GW -> image (comme _process_translation_path_batch)
            z_attr = gw.encode_domain([oh, at], "attr")
            z = gw.gw_mod.encode_and_fuse({"attr": z_attr}, selection_module=selection)
            v = gw.gw_mod.decode(z)["v_latents"]
            images = gw.domain_mods["v_latents"].decode_images(v)
            # Re-décodage des attributs depuis l'image traduite : image -> GW -> attributs
            v_back = gw.domain_mods["v_latents"].visual_module.encode(images)
            z_back = gw.gw_mod.encode_and_fuse({"v_latents": v_back}, selection_module=selection)
            decoded.append(gw.gw_mod.decode(z_back)["attr"].cpu().numpy())
            for img in images.permute(0, 2, 3, 1).cpu().numpy():
                mask = segment_shape(img)
                colors_bary.append(extract_shape_color(img, mask, True) * 255)
                colors_mask.append(extract_shape_color(img, mask, False) * 255)
                areas.append(mask.sum())
    return (np.array(colors_bary, dtype=np.float32), np.array(colors_mask, dtype=np.float32),
            np.array(areas, dtype=np.int32), np.concatenate(decoded).astype(np.float32))


def compute(models, out_dir, device):
    data_dir = out_dir / "donnees"
    data_dir.mkdir(parents=True, exist_ok=True)
    for exp_name, instance, ckpt, colored in models:
        f = data_dir / f"{exp_name}_{instance}.npz"
        if f.exists():
            print(f"Déjà calculé : {f.name}")
            continue
        print(f"Balayage : {exp_name}/{instance} ({ckpt})")
        gw = load_model(ckpt, colored, device)
        one_hots, attrs, meta = build_inputs(colored)
        bary, mask, areas, decoded = run_model(gw, one_hots, attrs, device)
        np.savez_compressed(f, one_hots=one_hots, attrs=attrs, meta=meta, rgb_barycentre=bary, rgb_masque=mask,
                            aire=areas, attributs_decodes=decoded, config=exp_name, instance=instance,
                            colored=colored)
        del gw
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# Analyse
# ----------------------------------------------------------------------------------------------
def hls_arrays(rgb):
    """H (degrés), L et S (0-1) de couleurs RGB 0-255, vectorisé (mêmes formules que colorsys)."""
    x = rgb / 255.0
    mx, mn = x.max(1), x.min(1)
    l = (mx + mn) / 2
    d = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(d == 0, 0, np.where(l <= 0.5, d / (mx + mn), d / (2 - mx - mn)))
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        h = np.where(mx == r, (g - b) / d, np.where(mx == g, 2 + (b - r) / d, 4 + (r - g) / d))
    h = np.where(d == 0, 0, (h / 6) % 1) * 360
    return h, l, s


def opponent(rgb):
    """Plan de chrominance : a = R - (G + B) / 2, b = sqrt(3)/2 (G - B) (angle = teinte, norme = chroma)."""
    return rgb[:, 0] - (rgb[:, 1] + rgb[:, 2]) / 2, np.sqrt(3) / 2 * (rgb[:, 1] - rgb[:, 2])


def circ_diff(a, b):
    return (a - b + 180) % 360 - 180


def load_sweeps(out_dir):
    sweeps = {}
    for f in sorted((out_dir / "donnees").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = z["meta"]
        rgb = z["rgb_barycentre"].astype(float)
        h, l, s = hls_arrays(rgb)
        a, b_ = opponent(rgb)
        d = dict(config=str(z["config"]), instance=str(z["instance"]), colored=bool(z["colored"]),
                 base=meta[:, 0].astype(int), shape=meta[:, 1].astype(int), kind=meta[:, 2].astype(int),
                 direction=meta[:, 3].astype(int), theta=meta[:, 4], eps=meta[:, 5], feasible=meta[:, 6] > 0,
                 H=h, L=l, S=s, a=a, b=b_, decoded=z["attributs_decodes"])
        base_idx = np.where(d["kind"] == KIND_BASE)[0]
        ref = {k: np.empty(int(d["base"].max()) + 1) for k in ("H", "L", "S", "a", "b")}
        for k in ref:
            ref[k][d["base"][base_idx]] = d[k][base_idx]
        d["dH"] = circ_diff(d["H"], ref["H"][d["base"]])
        d["dS"] = d["S"] - ref["S"][d["base"]]
        d["dL"] = d["L"] - ref["L"][d["base"]]
        d["dC"] = np.hypot(d["a"] - ref["a"][d["base"]], d["b"] - ref["b"][d["base"]])
        d["S0"] = ref["S"][d["base"]]
        d["recognized"] = d["decoded"][:, :3].argmax(1) == d["shape"]
        d["p_target"] = d["decoded"][np.arange(len(d["shape"])), d["shape"]]
        sweeps[(d["config"], d["instance"])] = d
    return sweeps


FEASIBLE_DEG = list(range(240, 301, 10))  # directions admissibles : le vecteur reste dans le simplexe


def direction_deg(d):
    return np.where(np.isfinite(d["theta"]), np.round(np.degrees(np.nan_to_num(d["theta"]))), -1).astype(int)


def mean_direction(angles):
    z = np.exp(1j * np.asarray(angles))
    return np.angle(z.mean()), np.abs(z.mean())


def permutation_test(angles, labels, n_perm=5000, seed=0):
    """Les directions moyennes diffèrent-elles entre groupes ? Statistique : somme des longueurs
    résultantes des groupes moins la longueur résultante totale (proche de 0 si mêmes directions)."""
    rng = np.random.default_rng(seed)
    z = np.exp(1j * np.asarray(angles))
    labels = np.asarray(labels)

    def stat(lab):
        return sum(np.abs(z[lab == g].sum()) for g in np.unique(lab)) - np.abs(z.sum())

    observed = stat(labels)
    null = np.array([stat(rng.permutation(labels)) for _ in range(n_perm)])
    return (1 + np.sum(null >= observed)) / (1 + n_perm)


def hue_at(d, eps, deg):
    """Teinte (degrés) et saturation par exemple pour une direction et une amplitude données."""
    sel = (d["kind"] == KIND_SHAPE) & np.isclose(d["eps"], eps) & (direction_deg(d) == deg)
    order = np.argsort(d["base"][sel])
    return d["base"][sel][order], d["H"][sel][order], d["S"][sel][order], d["shape"][sel][order]


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def group_label(key):
    exp_name, instance = key
    return f"{CONFIG_LABELS.get(exp_name, exp_name)} — {'contrôle' if instance == 'control' else instance}"


def analyze(out_dir, eps_ref):
    sweeps = load_sweeps(out_dir)
    if not sweeps:
        raise SystemExit(f"Aucune donnée dans {out_dir / 'donnees'}")
    keys = sorted(sweeps, key=lambda k: (list(CONFIG_LABELS).index(k[0]), k[1] == "control", k[1]))
    eps_ref = EPSILONS[np.argmin(np.abs(EPSILONS - eps_ref))]
    shown = [EPSILONS[0], EPSILONS[2], EPSILONS[4], EPSILONS[6], EPSILONS[8]]
    md = ["# Balayage du bruit sur les attributs de forme (chemin de traduction)\n",
          f"{len(sweeps)} modèles ; 48 exemples (3 formes × 4 orientations × 2 tailles × 2 positions) ; "
          f"36 directions du plan tangent au simplexe × {len(EPSILONS)} amplitudes ε de {EPSILONS[0]:g} à "
          f"{EPSILONS[-1]:g}. Couleur mesurée comme dans l'évaluation (barycentre du masque d'Otsu). "
          "Directions **admissibles** (240° à 300°) : le vecteur reste dans le simplexe, la masse quitte la "
          "forme cible vers les deux autres (à 240° tout vers l'une, à 300° tout vers l'autre, à 270° moitié-moitié), "
          "comme dans les probabilités produites par le décodeur d'attributs. Les autres directions donnent des "
          "composantes négatives, jamais vues à l'entraînement. ΔC : déplacement dans le plan de chrominance "
          "(a = R − (G+B)/2, b = √3/2 (G − B), unités RGB 0-255), défini même pour une couleur presque grise.\n"]

    # 1. Reconnaissance de la forme
    rows = []
    for key in keys:
        d = sweeps[key]
        base = d["kind"] == KIND_BASE
        shape = d["kind"] == KIND_SHAPE
        row = [group_label(key), f"{np.median(d['S'][base]):.3f}", f"{100 * d['recognized'][base].mean():.0f}"]
        for feasible in (True, False):
            for eps in (EPSILONS[4], EPSILONS[-1]):
                sel = shape & np.isclose(d["eps"], eps) & (d["feasible"] == feasible)
                row.append(f"{100 * d['recognized'][sel].mean():.0f}")
        row.append(f"{100 * d['recognized'][d['kind'] == KIND_ANNEX_E].mean():.0f}")
        rows.append(row)
    md += ["## 1. La forme reste-t-elle reconnue ?\n",
           "Forme re-décodée depuis l'image traduite (image → GW → attributs), en % des exemples.\n",
           md_table(["Modèle", "S sans bruit", "% sans bruit", f"% admissibles ε={EPSILONS[4]:g}",
                     f"% admissibles ε={EPSILONS[-1]:g}", f"% non admissibles ε={EPSILONS[4]:g}",
                     f"% non admissibles ε={EPSILONS[-1]:g}", "% bruit annexe E"], rows)]

    # 2. Saturation et amplitude
    rows = []
    for key in keys:
        d = sweeps[key]
        feas = (d["kind"] == KIND_SHAPE) & d["feasible"]
        row = [group_label(key), f"{np.median(d['S'][d['kind'] == KIND_BASE]):.3f}"]
        row += [f"{np.median(d['S'][feas & np.isclose(d['eps'], e)]):.3f}" for e in shown]
        sel_s = feas & np.isclose(d["eps"], EPSILONS[0])
        sel_c = (d["kind"] == KIND_CONT) & np.isclose(d["eps"], EPSILONS[0])
        row += [f"{np.median(d['dC'][sel_s]):.1f}", f"{np.median(d['dC'][sel_c]):.2f}",
                f"{np.median(d['dC'][(d['kind'] == KIND_CONT) & np.isclose(d['eps'], EPSILONS[-1])]):.1f}",
                f"{np.median(d['S'][d['kind'] == KIND_ANNEX_E]):.3f}"]
        rows.append(row)
    md += ["\n## 2. Saturation de la couleur traduite\n",
           "Médiane de S (0-1) sur les exemples et les directions admissibles ; ΔC médian pour la plus petite "
           "amplitude, perturbation de la forme (directions admissibles) contre perturbation des attributs "
           "continus (témoin).\n",
           md_table(["Modèle", "S sans bruit"] + [f"S ε={e:g}" for e in shown] +
                    [f"ΔC forme ε={EPSILONS[0]:g}", f"ΔC continus ε={EPSILONS[0]:g}",
                     f"ΔC continus ε={EPSILONS[-1]:g}", "S bruit annexe E"], rows)]

    # 3. Teinte selon la direction (cône admissible)
    rows = []
    for key in keys:
        d = sweeps[key]
        hues = np.array([hue_at(d, eps_ref, deg)[1] for deg in FEASIBLE_DEG])  # (7, 48)
        sats = np.array([hue_at(d, eps_ref, deg)[2] for deg in FEASIBLE_DEG])
        steps = np.abs(circ_diff(hues[1:], hues[:-1]))
        span = np.abs(circ_diff(hues[-1], hues[0]))
        rows.append([group_label(key), f"{np.median(sats):.2f}", f"{np.median(steps):.1f}",
                     f"{np.median(span):.1f}", f"{np.mean(sats.min(0) > 0.3) * 100:.0f}"])
    md += [f"\n## 3. La teinte dépend-elle de la répartition du résidu ? (ε = {eps_ref:g})\n",
           "Pour chaque exemple, teinte obtenue dans les 7 directions admissibles (240° à 300°, pas de 10°). "
           "Pas : écart de teinte entre directions voisines ; écart 240°-300° : entre les deux répartitions "
           "extrêmes (tout le résidu sur l'une ou l'autre des formes non cibles).\n",
           md_table(["Modèle", "S médiane", "pas médian (°)", "écart 240°-300° médian (°)",
                     "% d'exemples avec S > 0,3 partout"], rows)]

    # 4. Teinte propre à l'instance, et à la forme
    rows_inst, rows_shape = [], []
    for exp_name in CONFIG_LABELS:
        inst_keys = [k for k in keys if k[0] == exp_name and k[1] != "control"]
        if not inst_keys:
            continue
        for t, shape_name in enumerate(SHAPES):
            means_270, pvalues = "", []
            for deg in (240, 270, 300):
                angles, labels, means = [], [], []
                for k in inst_keys:
                    _, h, s, shapes = hue_at(sweeps[k], eps_ref, deg)
                    a = np.radians(h[shapes == t])
                    angles.append(a)
                    labels += [k[1]] * len(a)
                    mu, rbar = mean_direction(a)
                    means.append(f"{k[1]} {np.degrees(mu) % 360:.0f}° ({rbar:.2f})")
                if deg == 270:
                    means_270 = " ; ".join(means)
                pvalues.append(f"{permutation_test(np.concatenate(angles), labels):.4f}"
                               if len(inst_keys) > 1 else "")
            rows_inst.append([CONFIG_LABELS[exp_name], shape_name, means_270] + pvalues)
        for k in inst_keys:
            _, h, s, shapes = hue_at(sweeps[k], eps_ref, 270)
            means = []
            for t, shape_name in enumerate(SHAPES):
                mu, rbar = mean_direction(np.radians(h[shapes == t]))
                means.append(f"{shape_name} {np.degrees(mu) % 360:.0f}° ({rbar:.2f})")
            rows_shape.append([CONFIG_LABELS[exp_name], k[1], " ; ".join(means),
                               f"{permutation_test(np.radians(h), shapes):.4f}"])
    md += [f"\n## 4. Teinte propre à l'instance ? (ε = {eps_ref:g})\n",
           "Pour chaque forme : 16 exemples par instance. Teinte moyenne circulaire et longueur résultante R̄ "
           "(1 : même teinte pour tous les exemples de l'instance) dans la direction 270° (résidu partagé "
           "moitié-moitié). p : test de permutation des étiquettes d'instance (H0 : même teinte moyenne dans "
           "les trois instances), pour les directions 240°, 270° et 300°.\n",
           md_table(["Configuration", "Forme", "Teinte moyenne à 270° (R̄) par instance", "p 240°", "p 270°",
                     "p 300°"], rows_inst),
           f"\n## 5. Teinte propre à la forme dans une instance ? (ε = {eps_ref:g}, direction 270°)\n",
           "p : test de permutation des étiquettes de forme (H0 : même teinte moyenne pour les trois formes).\n",
           md_table(["Configuration", "Instance", "Teinte moyenne par forme (R̄)", "p"], rows_shape)]
    (out_dir / "synthese.md").write_text("\n".join(md) + "\n")
    figures(sweeps, keys, eps_ref, out_dir / "figures")
    print("\n".join(md))


def figures(sweeps, keys, eps_ref, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    configs = [c for c in CONFIG_LABELS if any(k[0] == c for k in keys)]

    def style(key):
        if key[1] == "control":
            return dict(color="grey", ls="--", lw=2.2)
        return dict(color=f"C{configs.index(key[0])}", ls="-", lw=1.3)

    # Saturation et reconnaissance en fonction de eps
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labelled = set()
    for key in keys:
        d = sweeps[key]
        label = "contrôles" if key[1] == "control" else CONFIG_LABELS[key[0]]
        lab = None if label in labelled else label
        labelled.add(label)
        feas = (d["kind"] == KIND_SHAPE) & d["feasible"]
        s_shape = [np.median(d["S"][feas & np.isclose(d["eps"], e)]) for e in EPSILONS]
        s_cont = [np.median(d["S"][(d["kind"] == KIND_CONT) & np.isclose(d["eps"], e)]) for e in EPSILONS]
        s0 = np.median(d["S"][d["kind"] == KIND_BASE])
        axes[0].plot(np.r_[EPSILONS[0] / 3, EPSILONS], np.r_[s0, s_shape], label=lab, **style(key))
        axes[1].plot(np.r_[EPSILONS[0] / 3, EPSILONS], np.r_[s0, s_cont], **style(key))
        for feasible, ls in ((True, "-"), (False, ":")):
            rate = [100 * d["recognized"][(d["kind"] == KIND_SHAPE) & (d["feasible"] == feasible)
                                          & np.isclose(d["eps"], e)].mean() for e in EPSILONS]
            axes[2].plot(EPSILONS, rate, color=style(key)["color"], ls=ls, alpha=0.7)
    axes[0].set_title("S : perturbation de la forme (directions admissibles)")
    axes[1].set_title("S : perturbation des attributs continus (témoin)")
    axes[2].set_title("Forme reconnue (— admissibles, ··· non admissibles)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("ε (premier point : sans bruit)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("S médiane")
    axes[1].set_ylim(axes[0].get_ylim())
    axes[2].set_ylabel("%")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "saturation_vs_epsilon.png", dpi=120)
    plt.close(fig)

    # Trajectoires dans le plan de chrominance (36 directions, eps max), par instance et par forme
    for exp_name in configs:
        inst_keys = [k for k in keys if k[0] == exp_name]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        for t, ax in enumerate(axes):
            for i, key in enumerate(inst_keys):
                d = sweeps[key]
                base = (d["kind"] == KIND_BASE) & (d["shape"] == t)
                da, db = [], []
                for m in range(N_DIRECTIONS):
                    sel = (d["kind"] == KIND_SHAPE) & np.isclose(d["eps"], eps_ref) & (d["direction"] == m) \
                        & (d["shape"] == t)
                    da.append(np.mean(d["a"][sel]) - np.mean(d["a"][base]))
                    db.append(np.mean(d["b"][sel]) - np.mean(d["b"][base]))
                da, db = np.array(da + da[:1]), np.array(db + db[:1])
                color = "grey" if key[1] == "control" else f"C{i}"
                ax.plot(da, db, "-", lw=0.8, color=color, label="contrôle" if key[1] == "control" else key[1])
                feas = [m for m in range(N_DIRECTIONS) if 240 <= 360 * m / N_DIRECTIONS <= 300]
                ax.plot(da[feas], db[feas], "o-", ms=4, lw=2, color=color)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_aspect("equal", "datalim")
            ax.set_title(SHAPES[t])
            ax.set_xlabel("Δa (unités RGB)")
        axes[0].set_ylabel("Δb (unités RGB)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{CONFIG_LABELS[exp_name]} : déplacement moyen de la couleur traduite (ε = {eps_ref:g}), "
                     "36 directions ; trait épais : directions admissibles")
        fig.tight_layout()
        fig.savefig(fig_dir / f"chrominance_{exp_name}.png", dpi=110)
        plt.close(fig)

    # Teinte par exemple et par instance, direction 270° : roue des teintes (couleur réelle)
    fig, axes = plt.subplots(len(configs), 3, subplot_kw={"projection": "polar"},
                             figsize=(10, 3.3 * len(configs)), squeeze=False)
    markers = ["o", "s", "^"]
    for r, exp_name in enumerate(configs):
        inst_keys = [k for k in keys if k[0] == exp_name and k[1] != "control"]
        for t in range(3):
            ax = axes[r][t]
            for i, key in enumerate(inst_keys):
                d = sweeps[key]
                sel = (d["kind"] == KIND_SHAPE) & np.isclose(d["eps"], eps_ref) & (direction_deg(d) == 270) \
                    & (d["shape"] == t)
                hsv = np.column_stack([d["H"][sel] / 360, d["S"][sel], np.full(sel.sum(), 0.85)])
                import matplotlib.colors as mcolors
                ax.scatter(np.radians(d["H"][sel]), np.full(sel.sum(), 1 + 0.25 * i), s=28,
                           c=mcolors.hsv_to_rgb(hsv), marker=markers[i % 3], edgecolors="k", linewidths=0.3,
                           label=key[1])
                mu, rbar = mean_direction(np.radians(d["H"][sel]))
                ax.plot([mu, mu], [0, rbar], color="k", lw=1.5, ls=["-", "--", ":"][i % 3])
            ax.set_ylim(0, 1.7)
            ax.set_yticklabels([])
            ax.set_title(f"{CONFIG_LABELS[exp_name]} — {SHAPES[t]}", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Teinte traduite, résidu partagé moitié-moitié (270°), ε = {eps_ref:g} ; "
                 "un cercle par instance, trait : teinte moyenne (longueur R̄)")
    fig.tight_layout()
    fig.savefig(fig_dir / "teintes_par_instance.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append", help="Configuration(s) (défaut : toutes)")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "saturation" / "balayage")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--eps-ref", type=float, default=0.01,
                        help="Amplitude utilisée pour comparer les teintes entre directions et instances")
    parser.add_argument("--threads", type=int, default=None, help="Threads PyTorch sur CPU")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.analyze_only:
        models = thesis_models(args.experiment, include_controls=not args.no_controls)[:args.max_models]
        compute(models, args.out, device)
    analyze(args.out, args.eps_ref)


if __name__ == "__main__":
    main()
