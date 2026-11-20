"""Lecture des résidus par régression (étape 2 du protocole « saturation »).

Question : peut-on relire la couleur d'une image d'entrée dans les attributs que le Global Workspace en
décode (image -> VAE -> GW -> décodeur d'attributs), alors que ces attributs ne contiennent pas de
couleur ? Autrement dit, la couleur est-elle cachée dans le résidu des probabilités de forme ?

Images : sous-ensemble stratifié du jeu de test, 32 images par case (teinte, forme, taille), soit
19 200 images ; orientation et position sont tirées au hasard dans chaque case, la teinte reste donc
indépendante des autres attributs. Pour chaque modèle, on garde les sorties du décodeur d'attributs
(logits de forme, dont les probabilités du modèle sont le softmax, voir SSD_eval/category_probs.py ;
5 attributs continus ; couleur décodée chez les Modèles Complets) et le latent GW.

Variables explicatives (groupes) :
  - résidu : log-rapports des deux probabilités de forme non cibles à la probabilité cible,
    log(p_j / p_t) et log(p_k / p_t), j < k (égaux aux différences de logits) ;
  - continus : écarts des 5 attributs continus décodés (position x et y, taille, cos et sin de
    l'orientation) à leurs vraies valeurs ;
  - les deux ;
  - « + contexte » : les mêmes, plus les vraies valeurs des 5 attributs continus. Indépendantes de la
    teinte par construction (témoin : contexte seul), elles permettent de séparer ce qui, dans le
    résidu, vient de la géométrie de ce qui vient de la couleur ;
  - repères : latent GW (12 dimensions, ce que le GW sait de la couleur avant le décodeur d'attributs) ;
    couleur décodée (Modèles Complets).
Cibles : couleur d'entrée (palette du jeu de test d'après color_index) dans le plan de chrominance,
a = R - (G+B)/2 et b = sqrt(3)/2 (G - B), et luminosité I = (R+G+B)/3 (unités RGB 0-255). L et S
(HLS) étant fixés à la génération (L = 180, S = 230), I ne varie ici qu'avec la teinte (157 pour les
primaires rouge, vert, bleu ; 201 pour jaune, cyan, magenta).
Régressions, ajustées séparément pour chaque forme cible (le sens des deux log-rapports en dépend) :
  - ridge (linéaire ; pénalité choisie pour chaque cible par leave-one-out exact) ;
  - k plus proches voisins (variables centrées réduites, k choisi de même) : non linéaire, mais se dégrade
    quand on lui ajoute des variables peu informatives ;
  - petit perceptron multicouche (2 couches cachées de 64 unités, mini-lots, arrêt précoce sur une partie de
    l'apprentissage), qui n'a pas ce défaut et sait lire le latent GW (environ 2 min 30 par modèle sur un
    GeForce 940MX, 4 à 6 min sur 4 cœurs de CPU).
R² en validation croisée à 5 plis, sur les prédictions hors pli ; ligne de base : même calcul avec les
couleurs mélangées entre images.

    python experiments/residual_regression.py                  # calcul (reprend où il s'est arrêté) + analyse
    python experiments/residual_regression.py --analyze-only   # sans passage dans les modèles (régressions en cache)
    python experiments/residual_regression.py --experiment high_cycles --max-models 1   # essai rapide
    python experiments/residual_regression.py --analyze-only --device cpu --experiment high_cycles   # en parallèle
                                                               # d'un autre processus qui traite d'autres modèles

Sorties dans results/saturation/regression/ : donnees/ (images, sorties des modèles, R² par modèle),
r2.csv, r2.md (tableaux), synthese.md, r2_par_modele.png.
"""

import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401
from shape_noise_sweep import CONFIG_LABELS, load_model, md_table, opponent, thesis_models  # noqa: E402
from SSD_utils import (extract_shape_color, generate_fixed_colors, normalize_position,  # noqa: E402
                       normalize_rotation, normalize_size, segment_shape)

N_COLORS = 25          # teintes du jeu de test
N_FOLDS = 5
RIDGE_ALPHAS = np.logspace(-3, 5, 17)
KNN_KS = np.array([5, 10, 20, 40, 80])
# Perceptron multicouche (mlp_cross_val_predict)
MLP_HIDDEN, MLP_LR, MLP_WEIGHT_DECAY, MLP_BATCH = 64, 3e-3, 1e-4, 256
MLP_MAX_STEPS, MLP_EVAL_EVERY, MLP_PATIENCE, MLP_VAL_FRACTION = 8000, 50, 1500, 0.15
GROUPS = {  # nom court -> libellé
    "residu": "résidu",
    "continus": "continus",
    "les_deux": "les deux",
    "contexte": "contexte seul (témoin)",
    "residu_contexte": "résidu + contexte",
    "continus_contexte": "continus + contexte",
    "les_deux_contexte": "les deux + contexte",
    "latent_gw": "latent GW (repère)",
    "couleur_decodee": "couleur décodée (Modèles Complets)",
}
REGRESSIONS = ("ridge", "kNN", "MLP")
TRANSMISSION_CSV = paths.RESULTS_DIR / "saturation" / "synesthesie" / "associations.csv"


# ----------------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------------
def select_subset(per_cell, seed):
    """per_cell images par case (teinte, forme, taille), triées par numéro (lecture plus rapide sur HDD)."""
    df = pd.read_csv(paths.EVALSET_DIR / "attributes.csv")
    sub = df.groupby(["color_index", "class", "size"], group_keys=False).sample(n=per_cell, random_state=seed)
    return sub.sort_index()


def load_image_set(data_dir, per_cell, seed):
    """Images du sous-ensemble, vraies valeurs des attributs et couleur d'entrée (mise en cache)."""
    f = data_dir / "images.npz"
    if f.exists():
        z = np.load(f)
        if int(z["par_case"]) != per_cell or int(z["graine"]) != seed:
            raise SystemExit(f"{f} a été calculé avec --per-cell {int(z['par_case'])} --seed {int(z['graine'])} : "
                             "changer --out ou supprimer le dossier donnees/.")
        return {k: z[k] for k in z.files}
    sub = select_subset(per_cell, seed)
    print(f"Lecture de {len(sub)} images ({paths.EVALSET_DIR})")
    images = np.empty((len(sub), 32, 32, 3), dtype=np.uint8)
    t0 = time.time()
    for i, name in enumerate(sub["filename"]):
        with Image.open(paths.EVALSET_DIR / name) as im:
            images[i] = np.asarray(im.convert("RGB"))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(sub)} ({time.time() - t0:.0f} s)")
    position = np.array([ast.literal_eval(s) for s in sub["location"]], dtype=np.float64)
    rotation = sub["rotation"].to_numpy(np.float64)
    size = sub["size"].to_numpy(np.float64)
    # Mêmes normalisations que l'entrée du modèle (SSD_utils.py)
    true_cont = np.column_stack([np.array([normalize_position(p) for p in position]),
                                 [normalize_size(s) for s in size],
                                 np.array([normalize_rotation(r) for r in rotation])])
    palette, _ = generate_fixed_colors(N_COLORS)
    # Couleur mesurée comme dans l'évaluation (masque d'Otsu, 3x3 pixels au barycentre), pour contrôle
    measured = np.array([extract_shape_color(x, segment_shape(x), True) * 255
                         for x in images.astype(np.float32) / 255])
    data = dict(numero=sub.index.to_numpy(), images=images, forme=sub["class"].to_numpy(),
                color_index=sub["color_index"].to_numpy(), taille=size, rotation=rotation, position=position,
                attributs_vrais=true_cont.astype(np.float32),
                rgb_palette=palette[sub["color_index"].to_numpy()].astype(np.float32),
                rgb_mesuree=measured.astype(np.float32), par_case=per_cell, graine=seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **data)
    return data


# ----------------------------------------------------------------------------------------------
# Passage dans les modèles
# ----------------------------------------------------------------------------------------------
def softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run_model(gw, images, device, batch_size):
    """Image -> VAE -> GW -> décodeur d'attributs, par lots (sorties brutes du décodeur et latents)."""
    from shimmer.modules.gw_module import GWDecoder_legacy
    from shimmer.modules.selection import SingleDomainSelection
    selection = SingleDomainSelection().to(device)
    decoder = gw.gw_mod.gw_decoders["attr"]
    vae = gw.domain_mods["v_latents"].visual_module
    out = {"latent_vae": [], "latent_gw": [], "sorties_decodeur": [], "probas": []}
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start:start + batch_size]).to(device)
            x = x.permute(0, 3, 1, 2).float().div(255).contiguous()  # comme ToTensor dans l'évaluation
            v = vae.encode(x)  # moyenne de l'encodeur du VAE
            z = gw.gw_mod.encode_and_fuse({"v_latents": v}, selection_module=selection)
            raw = GWDecoder_legacy.forward(decoder, z)  # logits de forme (sans le softmax), attributs (tanh)
            probs = gw.gw_mod.decode(z)["attr"][:, :3]  # probabilités de forme du modèle (category_probs.py)
            for key, tensor in zip(out, (v, z, raw, probs)):
                out[key].append(tensor.cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def compute(models, images, data_dir, device, batch_size):
    for exp_name, instance, ckpt, colored in models:
        f = data_dir / f"{exp_name}_{instance}.npz"
        if f.exists():
            print(f"Déjà calculé : {f.name}")
            continue
        print(f"Image -> GW -> attributs : {exp_name}/{instance} ({ckpt})")
        t0 = time.time()
        gw = load_model(ckpt, colored, device)
        out = run_model(gw, images["images"], device, batch_size)
        raw = out.pop("sorties_decodeur")
        gap = np.abs(softmax(raw[:, :3].astype(np.float64)) - out["probas"]).max()
        np.savez_compressed(f, logits_forme=raw[:, :3], attributs_decodes=raw[:, 3:8],
                            couleur_decodee=raw[:, 8:11] if colored else np.zeros((len(raw), 0), np.float32),
                            numero=images["numero"], config=exp_name, instance=instance, colored=colored,
                            checkpoint=str(ckpt), **out)
        print(f"  {time.time() - t0:.0f} s ; écart softmax(logits) - probabilités du modèle : {gap:.1e}")
        del gw
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# Régressions
# ----------------------------------------------------------------------------------------------
def standardize(train, test):
    mu, sd = train.mean(0), train.std(0)
    sd[sd < 1e-12] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def ridge_predict(x_train, y_train, x_test):
    """Ridge sur variables centrées réduites ; pénalité choisie pour chaque cible par leave-one-out exact."""
    a, b = standardize(x_train, x_test)
    y_mean = y_train.mean(0)
    yc = y_train - y_mean
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ yc
    errors = []
    for alpha in RIDGE_ALPHAS:
        shrink = s ** 2 / (s ** 2 + alpha)
        fitted = u @ (shrink[:, None] * uty)
        leverage = (u ** 2) @ shrink + 1.0 / len(a)  # diagonale de la matrice chapeau (constante comprise)
        errors.append((((yc - fitted) / (1 - leverage)[:, None]) ** 2).mean(0))
    best = RIDGE_ALPHAS[np.argmin(np.array(errors), axis=0)]
    coef = vt.T @ ((s[:, None] / (s[:, None] ** 2 + best[None, :])) * uty)
    return b @ coef + y_mean


def knn_predict(x_train, y_train, x_test):
    """Moyenne des k plus proches voisins (variables centrées réduites) ; k choisi pour chaque cible par
    leave-one-out sur l'ensemble d'apprentissage."""
    a, b = standardize(x_train, x_test)
    n, k_max = len(a), int(KNN_KS.max())
    tree = cKDTree(a)
    _, idx = tree.query(a, k=k_max + 1, workers=-1)
    own = idx == np.arange(n)[:, None]
    own[~own.any(1), -1] = True  # point absent de ses propres voisins (ex aequo) : on retire le dernier
    idx = idx[~own].reshape(n, k_max)
    loo = np.cumsum(y_train[idx], axis=1)[:, KNN_KS - 1] / KNN_KS[None, :, None]
    best = KNN_KS[np.argmin(((loo - y_train[:, None, :]) ** 2).mean(0), axis=0)]
    _, idx_test = tree.query(b, k=k_max, workers=-1)
    neighbours = y_train[idx_test]
    return np.stack([neighbours[:, :k, j].mean(1) for j, k in enumerate(best)], axis=1)


PREDICTORS = {"ridge": ridge_predict, "kNN": knn_predict}


def fold_tasks(shapes, folds):
    """(apprentissage, test) pour chaque pli et chaque forme cible."""
    tasks = []
    for test in folds:
        in_test = np.zeros(len(shapes), dtype=bool)
        in_test[test] = True
        for s in range(3):
            tasks.append((np.where(~in_test & (shapes == s))[0], np.where(in_test & (shapes == s))[0]))
    return tasks


def cross_val_predict(x, y, shapes, folds, predict):
    """Prédictions hors pli ; un modèle par forme cible et par pli."""
    pred = np.empty_like(y)
    for train, held_out in fold_tasks(shapes, folds):
        pred[held_out] = predict(x[train], y[train], x[held_out])
    return pred


def _padded(arrays):
    """Empile des tableaux (n_i, d) de longueurs différentes : (m, max n_i, d) et masque (m, max n_i)."""
    out = np.zeros((len(arrays), max(len(a) for a in arrays), arrays[0].shape[1]), dtype=np.float32)
    mask = np.zeros(out.shape[:2], dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, :len(a)], mask[i, :len(a)] = a, 1
    return out, mask


def mlp_cross_val_predict(x, targets, shapes, folds, seed, device):
    """Petit perceptron multicouche (2 couches cachées de MLP_HIDDEN unités, ReLU), un réseau par pli, par
    forme cible et par jeu de cibles (vraies couleurs, couleurs mélangées) ; ces réseaux indépendants sont
    entraînés ensemble (produits matriciels par lots). Variables et cibles centrées réduites sur
    l'apprentissage ; AdamW sur des mini-lots de MLP_BATCH exemples ; arrêt précoce sur 15 % de
    l'apprentissage mis de côté (on garde les poids du meilleur point). Renvoie une prédiction hors pli
    par jeu de cibles."""
    rng = np.random.default_rng(seed)
    blocks = {k: [] for k in ("x_fit", "y_fit", "x_val", "y_val", "x_test")}
    tasks, target_stats = [], []
    for j, y in enumerate(targets):
        for train, held_out in fold_tasks(shapes, folds):
            train = rng.permutation(train)
            n_val = int(round(MLP_VAL_FRACTION * len(train)))
            val, fit = train[:n_val], train[n_val:]
            mx, sx = x[fit].mean(0), x[fit].std(0)
            my, sy = y[fit].mean(0), y[fit].std(0)
            sx[sx < 1e-12], sy[sy < 1e-12] = 1.0, 1.0
            tasks.append((j, held_out))
            target_stats.append((my, sy))
            for key, idx in (("fit", fit), ("val", val), ("test", held_out)):
                blocks[f"x_{key}"].append((x[idx] - mx) / sx)
                if key != "test":
                    blocks[f"y_{key}"].append((y[idx] - my) / sy)
    t = {}
    for key, arrays in blocks.items():
        values, mask = _padded(arrays)
        t[key], t["m_" + key[2:]] = torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device)
    n_fit = t["m_fit"].sum(1)
    m, p, q, h = len(tasks), x.shape[1], targets[0].shape[1], MLP_HIDDEN
    generator = torch.Generator().manual_seed(seed)
    shapes_init = [((m, p, h), p), ((m, 1, h), p), ((m, h, h), h), ((m, 1, h), h), ((m, h, q), h), ((m, 1, q), h)]
    params = [((torch.rand(s, generator=generator) * 2 - 1) / np.sqrt(fan_in)).to(device).requires_grad_()
              for s, fan_in in shapes_init]  # initialisation de nn.Linear
    batch_generator = torch.Generator().manual_seed(seed)  # sur CPU : mêmes mini-lots quel que soit l'appareil
    rows = torch.arange(m, device=device)[:, None]

    def forward(inputs, w):
        hidden = torch.relu(torch.baddbmm(w[1], inputs, w[0]))
        hidden = torch.relu(torch.baddbmm(w[3], hidden, w[2]))
        return torch.baddbmm(w[5], hidden, w[4])

    optimizer = torch.optim.AdamW(params, lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    best = [w.detach().clone() for w in params]
    best_val = torch.full((m,), float("inf"), device=device)
    last_improvement = 0
    for step in range(1, MLP_MAX_STEPS + 1):
        idx = (torch.rand((m, MLP_BATCH), generator=batch_generator).to(device) * n_fit[:, None]).long()
        optimizer.zero_grad()
        pred = forward(t["x_fit"][rows, idx], params)
        ((pred - t["y_fit"][rows, idx]) ** 2).mean((1, 2)).sum().backward()
        optimizer.step()
        if step % MLP_EVAL_EVERY:
            continue
        with torch.no_grad():
            val = (((forward(t["x_val"], params) - t["y_val"]) ** 2).mean(2) * t["m_val"]).sum(1) / t["m_val"].sum(1)
            improved = val < best_val * (1 - 1e-3)
            if improved.any():
                last_improvement, best_val = step, torch.where(improved, val, best_val)
                for b, w in zip(best, params):
                    b[improved] = w.detach()[improved]
        if step - last_improvement >= MLP_PATIENCE:
            break
    with torch.no_grad():
        out = forward(t["x_test"], best).cpu().numpy().astype(np.float64)
    preds = [np.empty_like(y) for y in targets]
    for i, (j, held_out) in enumerate(tasks):
        my, sy = target_stats[i]
        preds[j][held_out] = out[i, :len(held_out)] * sy + my
    return preds


def scores(y, pred):
    sse = ((y - pred) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    r2 = 1 - sse / sst
    hue_error = np.degrees(np.arctan2(pred[:, 1], pred[:, 0]) - np.arctan2(y[:, 1], y[:, 0]))
    return dict(r2_a=r2[0], r2_b=r2[1], r2_chrominance=1 - sse[:2].sum() / sst[:2].sum(), r2_luminosite=r2[2],
                erreur_teinte_deg=float(np.median(np.abs((hue_error + 180) % 360 - 180))))


def residual_log_ratios(logits, shapes):
    """log(p_j / p_t) et log(p_k / p_t) (j < k, les deux formes non cibles) = différences de logits."""
    n = len(shapes)
    others = np.array([[j for j in range(3) if j != s] for s in range(3)])[shapes]
    return logits[np.arange(n)[:, None], others] - logits[np.arange(n), shapes][:, None]


def feature_groups(d, img):
    residual = residual_log_ratios(d["logits_forme"].astype(np.float64), img["forme"])
    deviations = d["attributs_decodes"].astype(np.float64) - img["attributs_vrais"]
    context = img["attributs_vrais"].astype(np.float64)
    groups = {"residu": residual, "continus": deviations, "les_deux": np.hstack([residual, deviations]),
              "contexte": context, "residu_contexte": np.hstack([residual, context]),
              "continus_contexte": np.hstack([deviations, context]),
              "les_deux_contexte": np.hstack([residual, deviations, context]),
              "latent_gw": d["latent_gw"].astype(np.float64)}
    if d["couleur_decodee"].shape[1]:
        groups["couleur_decodee"] = d["couleur_decodee"].astype(np.float64)
    return groups


def color_targets(img, source):
    """(a, b, I) de la couleur d'entrée : palette du jeu de test ou couleur mesurée sur l'image."""
    rgb = img["rgb_palette" if source == "palette" else "rgb_mesuree"].astype(np.float64)
    a, b = opponent(rgb)
    return np.column_stack([a, b, rgb.mean(1)])


def variance_explained_by_hue(x, hue_index, shapes):
    """η² : part de la variance de chaque variable expliquée par la teinte d'entrée, forme par forme
    (moyenne sur les formes et sur les variables ; environ 0,004 sans aucun lien)."""
    values = []
    for s in range(3):
        m = shapes == s
        for v in x[m].T:
            means = np.array([v[hue_index[m] == c].mean() for c in range(N_COLORS)])
            values.append(np.var(means[hue_index[m]]) / v.var())
    return float(np.mean(values))


def descriptive(d, img):
    """Ampleur du résidu et des écarts, et part de leur variance due à la teinte : de quoi interpréter les R²."""
    logits = d["logits_forme"].astype(np.float64)
    p = softmax(logits)
    t = img["forme"]
    mass = 1 - p[np.arange(len(t)), t]
    deviations = d["attributs_decodes"].astype(np.float64) - img["attributs_vrais"]
    return dict(forme_reconnue=float(np.mean(logits.argmax(1) == t)),
                masse_residuelle_mediane=float(np.median(mass)),
                masse_residuelle_p90=float(np.percentile(mass, 90)),
                ecart_continus_median=float(np.median(np.abs(deviations))),
                eta2_residu=variance_explained_by_hue(residual_log_ratios(logits, t), img["color_index"], t),
                eta2_continus=variance_explained_by_hue(deviations, img["color_index"], t))


def fit_model(d, img, args, regressions, device):
    """R² de chaque groupe de variables, pour les régressions demandées, vraies couleurs et couleurs mélangées."""
    y = color_targets(img, args.target)
    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(len(y)), N_FOLDS)
    y_shuffled = y[rng.permutation(len(y))]
    labelled = (("vraies", y), ("mélangées", y_shuffled))
    rows = []
    for group, x in feature_groups(d, img).items():
        for regression in regressions:
            if regression == "MLP":  # entraînements séparés : les réseaux à couleurs mélangées s'arrêtent tôt
                preds = [mlp_cross_val_predict(x, [yy], img["forme"], folds, args.seed, device)[0]
                         for _, yy in labelled]
            else:
                preds = [cross_val_predict(x, yy, img["forme"], folds, PREDICTORS[regression]) for _, yy in labelled]
            for (labels, yy), pred in zip(labelled, preds):
                rows.append(dict(groupe=group, regression=regression, etiquettes=labels, **scores(yy, pred)))
    return rows


def model_key(exp_name, instance):
    order = list(CONFIG_LABELS)
    return order.index(exp_name) if exp_name in order else len(order), instance == "control", instance


def model_label(exp_name, instance):
    label = CONFIG_LABELS.get(exp_name, exp_name)
    if instance == "control":
        return f"{label}, contrôle"
    return f"{label}, graine {instance.replace('seed', '')}"


def fit_all(data_dir, img, args, selected, device):
    """R² de tous les modèles calculés (mis en cache dans donnees/<modèle>_r2_<cible>.csv). Seules les
    régressions absentes du cache sont calculées, et seulement pour les modèles sélectionnés ; les autres
    figurent dans les tableaux avec ce que contient leur cache."""
    rows = []
    for f in sorted(data_dir.glob("*.npz"), key=lambda p: p.name):
        if f.name == "images.npz":
            continue
        d = np.load(f)
        exp_name, instance = str(d["config"]), str(d["instance"])
        cache = data_dir / f"{exp_name}_{instance}_r2_{args.target}.csv"
        refit = args.refit and (exp_name, instance) in selected
        model_rows = pd.read_csv(cache).to_dict("records") if cache.exists() and not refit else []
        missing = [r for r in args.regressions if r not in {row["regression"] for row in model_rows}]
        if missing and (exp_name, instance) in selected:
            t0 = time.time()
            model_rows += fit_model(d, img, args, missing, device)
            tmp = cache.with_suffix(".tmp")
            pd.DataFrame(model_rows).to_csv(tmp, index=False)
            tmp.replace(cache)  # écriture atomique : un autre processus peut lire ce cache
            print(f"Régressions ({', '.join(missing)}) : {exp_name}/{instance} ({time.time() - t0:.0f} s)")
        info = dict(config=exp_name, instance=instance, controle=bool(d["colored"]), **descriptive(d, img))
        rows += [{**info, **r} for r in model_rows]
    if not rows:
        raise SystemExit(f"Aucune sortie de modèle dans {data_dir}")
    df = pd.DataFrame(rows)
    df["configuration"] = df["config"].map(lambda c: CONFIG_LABELS.get(c, c))
    df["groupe_libelle"] = df["groupe"].map(GROUPS)
    df["_ordre"] = [model_key(c, i) for c, i in zip(df["config"], df["instance"])]
    return df.sort_values("_ordre", kind="stable").drop(columns="_ordre").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Tableaux, synthèse, figure
# ----------------------------------------------------------------------------------------------
def read_transmission(f=TRANSMISSION_CSV):
    """R̄ de transmission de la teinte en cycle complet (synesthesia_figure.py), moyenne des 3 formes."""
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    df = df[df["condition"] == "sans_bruit"]
    return {(c, i): float(v) for (c, i), v in df.groupby(["config", "instance"])["transmission_cycle_complet"].mean()
            .items()}


def fr(x, digits=2):
    if x is None or not np.isfinite(x):
        return "–"
    if round(x, digits) == 0:
        x = 0.0  # pas de « −0,00 »
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def value(df, exp_name, instance, group, regression, labels="vraies", key="r2_chrominance"):
    sel = df[(df["config"] == exp_name) & (df["instance"] == instance) & (df["groupe"] == group)
             & (df["regression"] == regression) & (df["etiquettes"] == labels)]
    return float(sel[key].iloc[0]) if len(sel) else np.nan


def models_in(df):
    return list(dict.fromkeys(zip(df["config"], df["instance"])))


def best_of(df, exp_name, instance, group, key="r2_chrominance"):
    return np.nanmax([value(df, exp_name, instance, group, r, key=key) for r in REGRESSIONS])


def table_by_model(df, key, groups):
    rows = []
    for exp_name, instance in models_in(df):
        row = [model_label(exp_name, instance)]
        for g in groups:
            vals = [value(df, exp_name, instance, g, r, key=key) for r in REGRESSIONS]
            row.append("–" if all(np.isnan(vals)) else " / ".join(fr(v) for v in vals))
        shuffled = df[(df["config"] == exp_name) & (df["instance"] == instance) & (df["etiquettes"] == "mélangées")]
        row.append(fr(shuffled[key].max()))
        rows.append(row)
    return md_table(["Modèle"] + [GROUPS[g] for g in groups] + ["mélangées (max)"], rows)


def write_reports(df, img, out_dir, args):
    transmission = read_transmission()
    main_groups = ["residu", "continus", "les_deux", "les_deux_contexte", "latent_gw"]
    context_groups = ["contexte", "residu_contexte", "continus_contexte", "les_deux_contexte"]
    n = len(img["forme"])
    rgb = img["rgb_palette"].astype(np.float64)
    gap = np.abs(img["rgb_mesuree"] - rgb)
    intro = (f"{n} images du jeu de test ({args.per_cell} par case teinte × forme × taille, orientation et "
             f"position tirées au hasard, graine {args.seed}). Image → VAE → GW → décodeur d'attributs. "
             "Cibles : couleur d'entrée (palette, `color_index`) dans le plan de chrominance "
             "(a = R − (G+B)/2, b = √3/2 (G − B)) et luminosité I = (R+G+B)/3. R² en validation croisée à "
             f"{N_FOLDS} plis (prédictions hors pli, un modèle par forme cible) ; "
             "« ridge / kNN / MLP » : régression ridge / k plus proches voisins / perceptron multicouche ; "
             "« mélangées (max) » : plus grande valeur obtenue, tous groupes et régressions confondus, quand les "
             "couleurs sont mélangées entre images.")
    md = ["# Lecture des résidus par régression : tableaux\n", intro + "\n",
          "## R² de la chrominance (a, b)\n", table_by_model(df, "r2_chrominance", main_groups),
          "\n## R² de la luminosité I\n", table_by_model(df, "r2_luminosite", main_groups),
          "\n## Avec le contexte (vraies valeurs des attributs continus), R² de la chrominance\n",
          table_by_model(df, "r2_chrominance", context_groups),
          "\n## Erreur médiane sur la teinte (degrés ; 90° = hasard, écart entre teintes voisines : 14,4°)\n",
          table_by_model(df, "erreur_teinte_deg", main_groups)]
    controls = [(c, i) for c, i in models_in(df) if i == "control"]
    if controls:
        rows = [[model_label(c, i)] + [" / ".join(fr(value(df, c, i, "couleur_decodee", r, key=k)) for r in REGRESSIONS)
                                       for k in ("r2_chrominance", "r2_luminosite")] for c, i in controls]
        md += ["\n## Modèles Complets : couleur décodée (repère)\n",
               md_table(["Modèle", "R² chrominance", "R² luminosité"], rows)]
    rows = []
    for exp_name, instance in models_in(df):
        sel = df[(df["config"] == exp_name) & (df["instance"] == instance)].iloc[0]
        rows.append([model_label(exp_name, instance), fr(100 * sel["forme_reconnue"], 1),
                     f"{sel['masse_residuelle_mediane']:.1e}".replace(".", ","),
                     f"{sel['masse_residuelle_p90']:.1e}".replace(".", ","), fr(sel["ecart_continus_median"], 3),
                     fr(sel["eta2_residu"]), fr(sel["eta2_continus"], 3),
                     fr(transmission.get((exp_name, instance), np.nan))])
    md += ["\n## Ampleur du résidu et des écarts\n",
           "Masse résiduelle : 1 − p(forme cible). Écart des continus : |décodé − vrai|, médiane sur les 5 "
           "attributs (échelle −1 à 1). η² : part de la variance de chaque variable (log-rapports du résidu, écarts "
           "des continus) expliquée par la teinte d'entrée, forme par forme, moyenne sur les formes et les variables "
           "(environ 0,004 sans aucun lien). R̄ cycle complet : transmission de la teinte d'entrée en cycle complet "
           "(`synesthesie/associations.csv`, moyenne des 3 formes).\n",
           md_table(["Modèle", "forme reconnue (%)", "masse résiduelle médiane", "masse résiduelle p90",
                     "écart des continus médian", "η² résidu", "η² continus", "R̄ cycle complet"], rows)]
    (out_dir / "r2.md").write_text("\n".join(md) + "\n")
    df.to_csv(out_dir / "r2.csv", index=False, float_format="%.5g", quoting=csv.QUOTE_MINIMAL)
    write_synthesis(df, img, out_dir, transmission, args, gap)
    figure(df, transmission, out_dir / "r2_par_modele.png")


def config_summary(df, group, regression=None, key="r2_chrominance"):
    """{config: (moyenne, min, max)} sur les 3 graines (meilleure des trois régressions si regression=None)."""
    out = {}
    for exp_name in CONFIG_LABELS:
        vals = [best_of(df, c, i, group, key) if regression is None else value(df, c, i, group, regression, key=key)
                for c, i in models_in(df) if c == exp_name and i != "control"]
        if vals:
            out[exp_name] = (np.mean(vals), np.min(vals), np.max(vals))
    return out


def write_synthesis(df, img, out_dir, transmission, args, gap):
    groups = ["residu", "continus", "continus_contexte", "les_deux", "les_deux_contexte", "latent_gw"]
    summaries = {g: config_summary(df, g) for g in groups}
    rows = []
    for exp_name in CONFIG_LABELS:
        if exp_name not in summaries["residu"]:
            continue
        row = [CONFIG_LABELS[exp_name]]
        for g in groups:
            m, lo, hi = summaries[g][exp_name]
            row.append(f"{fr(m)} ({fr(lo)} à {fr(hi)})")
        trans = [transmission[(c, i)] for c, i in models_in(df) if c == exp_name and i != "control"
                 and (c, i) in transmission]
        row.append(fr(np.mean(trans)) if trans else "–")
        rows.append(row)
    for exp_name, instance in [(c, i) for c, i in models_in(df) if i == "control"]:
        rows.append([model_label(exp_name, instance)] + [fr(best_of(df, exp_name, instance, g)) for g in groups]
                    + ["–"])
    shuffled = df[df["etiquettes"] == "mélangées"]
    lum = {g: config_summary(df, g, key="r2_luminosite") for g in ("residu", "les_deux", "les_deux_contexte")}
    lum_rows = [[CONFIG_LABELS[c]] + [f"{fr(lum[g][c][0])} ({fr(lum[g][c][1])} à {fr(lum[g][c][2])})"
                                      for g in lum] for c in CONFIG_LABELS if c in lum["residu"]]
    md = ["# Lecture des résidus par régression : synthèse\n",
          "Question : la couleur de l'image d'entrée peut-elle être relue dans les attributs que le GW en décode "
          "(image → VAE → GW → décodeur d'attributs), alors que ces attributs ne contiennent pas de couleur ?\n",
          f"Méthode : {len(img['forme'])} images du jeu de test (stratifiées par teinte × forme × taille), "
          "sorties du décodeur d'attributs de chaque modèle ; régression de la couleur d'entrée (chrominance a, b "
          "et luminosité I = (R+G+B)/3) sur le résidu des probabilités de forme (log-rapports des deux formes non "
          "cibles à la forme cible), sur les écarts des 5 attributs continus à leurs vraies valeurs, ou sur les "
          "deux, avec ou sans les vraies valeurs des attributs (contexte) ; ridge, k plus proches voisins et "
          f"perceptron multicouche, un modèle par forme cible, R² en validation croisée à {N_FOLDS} plis. Détail "
          "par modèle : `r2.md`, `r2.csv` ; figure : `r2_par_modele.png`.\n",
          "## R² de la chrominance par configuration\n",
          "Meilleure des trois régressions, moyenne des 3 graines (min à max). « + contexte » : les vraies valeurs "
          "des attributs continus sont ajoutées aux variables (elles ne disent rien de la couleur à elles seules). "
          "Latent GW : la couleur lue directement dans l'espace de travail, avant le décodeur d'attributs.\n",
          md_table(["Configuration"] + [GROUPS[g] for g in groups] + ["R̄ cycle complet"], rows),
          "\n## R² de la luminosité I par configuration\n",
          md_table(["Configuration"] + [GROUPS[g] for g in lum], lum_rows),
          "\n## Contrôles de la méthode\n",
          f"- Couleurs mélangées entre images : R² entre {fr(shuffled['r2_chrominance'].min(), 3)} et "
          f"{fr(shuffled['r2_chrominance'].max(), 3)} (chrominance), entre {fr(shuffled['r2_luminosite'].min(), 3)} "
          f"et {fr(shuffled['r2_luminosite'].max(), 3)} (luminosité), tous modèles, groupes et régressions.",
          f"- Contexte seul (vraies valeurs des attributs continus) : R² de la chrominance au plus "
          f"{fr(df[(df['groupe'] == 'contexte') & (df['etiquettes'] == 'vraies')]['r2_chrominance'].max(), 3)} : "
          "la teinte est bien indépendante de la géométrie dans le sous-ensemble.",
          f"- Couleur d'entrée mesurée sur l'image (barycentre du masque d'Otsu) contre palette : écart médian "
          f"{fr(np.median(gap), 1)}, 95e centile {fr(np.percentile(gap, 95), 1)} (unités RGB) ; la cible est la "
          "palette.",
          "- Luminosité : L et S (HLS) sont fixés à la génération du jeu de test ; I = (R+G+B)/3 ne varie qu'avec la "
          "teinte (157 pour rouge, vert, bleu ; 201 pour jaune, cyan, magenta), comme une harmonique d'ordre 3 de "
          "la teinte."]
    md += knn_dilution_notes(df)
    md[3:3] = reading_lines(df, transmission) + answer_lines(df) + hypothesis_lines(df, transmission)
    (out_dir / "synthese.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# Hypothèse de la session précédente (R̄ de transmission de la teinte en cycle complet)
EXPECTED = {"high_cycles": "l'essentiel", "base_params": "une partie", "high_alpha": "une partie",
            "high_translation": "une partie", "high_contrastive": "rien"}
CLASSES = ["rien", "une partie", "l'essentiel"]


def r2_class(r2):
    """Classe de l'hypothèse : l'essentiel (R² >= 0,7), une partie (0,1 à 0,7), rien (< 0,1)."""
    return "l'essentiel" if r2 >= 0.7 else ("une partie" if r2 >= 0.1 else "rien")


def best_regression(df, exp_name, instance, group):
    vals = {r: value(df, exp_name, instance, group, r) for r in REGRESSIONS}
    return max(vals, key=lambda r: -np.inf if np.isnan(vals[r]) else vals[r])


def describe_models(keys, semantic):
    """« Référence, Emphase Cycle-Complet (graine 0) » : configurations entières ou graines."""
    parts = []
    for c in CONFIG_LABELS:
        seeds = [k[1].replace("seed", "") for k in keys if k[0] == c]
        if seeds:
            whole = len(seeds) == sum(1 for k in semantic if k[0] == c)
            parts.append(CONFIG_LABELS[c] + ("" if whole else f" (graine{'s' if len(seeds) > 1 else ''} "
                                                              f"{', '.join(seeds)})"))
    return ", ".join(parts) if parts else "aucun modèle"


def reading_lines(df, transmission):
    """Lecture des résultats, rédigée d'après les chiffres (modèles « vide sémantique » et contrôles)."""
    semantic = [k for k in models_in(df) if k[1] != "control"]
    controls = [k for k in models_in(df) if k[1] == "control"]
    if not semantic:
        return []
    configs = [c for c in CONFIG_LABELS if any(k[0] == c for k in semantic)]
    r2 = {g: {k: best_of(df, *k, g) for k in semantic}
          for g in ("residu", "continus", "continus_contexte", "les_deux_contexte", "latent_gw")}
    ctrl = {g: max([best_of(df, *k, g) for k in controls], default=np.nan)
            for g in ("residu", "continus_contexte", "les_deux_contexte")}
    tout = r2["les_deux_contexte"]
    out = ["## Lecture\n"]
    if controls:
        above = [c for c in configs if min(tout[k] for k in semantic if k[0] == c) > ctrl["les_deux_contexte"] + 0.1]
        sel = [k for k in semantic if k[0] in above]
        if sel:
            rest = [k for k in semantic if k[0] not in above]
            out.append(f"- **La couleur est lisible dans les attributs décodés**, qui n'en contiennent pas, pour "
                       f"{len(above)} configurations sur {len(configs)} ({describe_models(sel, semantic)}) : R² de "
                       f"{fr(min(tout[k] for k in sel))} à {fr(max(tout[k] for k in sel))} selon la graine (résidu, "
                       f"écarts des continus et géométrie ensemble), contre au plus {fr(ctrl['les_deux_contexte'])} "
                       "chez les Modèles Complets, qui n'ont aucune raison de la cacher"
                       + (f" ; {describe_models(rest, semantic)} : {fr(min(tout[k] for k in rest))} à "
                          f"{fr(max(tout[k] for k in rest))}, à peine au-dessus." if rest else "."))
    classes = {name: [k for k in semantic if r2_class(r2["residu"][k]) == name] for name in reversed(CLASSES)}
    out.append(f"- **Résidu des probabilités de forme, seul** : l'essentiel de la couleur (R² ≥ 0,7) pour "
               f"{describe_models(classes[CLASSES[2]], semantic)} ; une partie (0,1 à 0,7) pour "
               f"{describe_models(classes[CLASSES[1]], semantic)} ; presque rien (< 0,1) pour "
               f"{describe_models(classes[CLASSES[0]], semantic)}. Chez les Modèles Complets : au plus "
               f"{fr(ctrl['residu'])}.")
    cc = r2["continus_contexte"]
    text = (f"- **Écarts des attributs continus** : seuls, au plus {fr(max(r2['continus'].values()))} ; avec la "
            f"géométrie (vraies valeurs), {fr(min(cc.values()))} à {fr(max(cc.values()))}")
    if controls:
        text += f", contre au plus {fr(ctrl['continus_contexte'])} chez les Modèles Complets"
    out.append(text + ". La couleur y est donc aussi inscrite, mais d'une façon qui dépend de la forme, de la taille, "
               "de la position ou de l'orientation : illisible sans elles.")
    gain = [tout[k] - max(r2["residu"][k], cc[k]) for k in semantic]
    out.append(f"- **Les deux canaux se complètent** : « tout » dépasse le meilleur des deux pris séparément de "
               f"{fr(np.mean(gain))} en moyenne (de {fr(min(gain))} à {fr(max(gain))}).")
    keys = [k for k in semantic if k in transmission]
    if len(keys) > 2:
        rho = {g: spearmanr([transmission[k] for k in keys], [r2[g][k] for k in keys])[0]
               for g in ("les_deux_contexte", "residu")}
        out.append(f"- **Ce qui est lisible prédit ce que transmet le cycle complet** (image → GW → attributs → GW → "
                   f"image) : corrélation de Spearman avec R̄ de {fr(rho['les_deux_contexte'])} pour « tout », "
                   f"{fr(rho['residu'])} pour le résidu seul ({len(keys)} modèles).")
    gw = r2["latent_gw"]
    low = [c for c in configs if np.mean([gw[k] for k in semantic if k[0] == c]) < 0.5]
    high = [k for k in semantic if k[0] not in low]
    text = f"- **Latent GW** : R² de {fr(min(gw[k] for k in high))} à {fr(max(gw[k] for k in high))}" if high else \
        "- **Latent GW** :"
    if low:
        low_keys = [k for k in semantic if k[0] in low]
        text += (f" ; seulement {fr(min(gw[k] for k in low_keys))} à {fr(max(gw[k] for k in low_keys))} pour "
                 f"{describe_models(low_keys, semantic)}, où la couleur est déjà en grande partie perdue dans "
                 "l'espace de travail")
    if high:
        text += (f". Ailleurs, c'est le décodeur d'attributs qui en laisse passer plus ou moins (« tout » de "
                 f"{fr(min(tout[k] for k in high))} à {fr(max(tout[k] for k in high))})")
    out.append(text + ".")
    lum = {k: best_of(df, *k, "les_deux_contexte", key="r2_luminosite") for k in semantic}
    ridge_lum = max(value(df, *k, g, "ridge", key="r2_luminosite") for k in semantic
                    for g in ("residu", "continus", "les_deux", "les_deux_contexte"))
    best_cfg = max(configs, key=lambda c: np.mean([lum[k] for k in semantic if k[0] == c]))
    out.append(f"- **Luminosité I** (ici fonction de la teinte seule, harmonique d'ordre 3) : lue seulement de façon "
               f"non linéaire (ridge : au plus {fr(ridge_lum)}) et là où la teinte est relue précisément (« tout » : "
               f"{fr(np.mean([lum[k] for k in semantic if k[0] == best_cfg]))} en moyenne pour "
               f"{CONFIG_LABELS[best_cfg]}, de {fr(min(lum.values()))} à {fr(max(lum.values()))} sur l'ensemble).")
    return out + [""]


def answer_lines(df):
    """Chiffres par configuration (moyenne des graines, meilleure des trois régressions)."""
    out = ["## En chiffres\n",
           "R² de la chrominance (moyenne des graines, meilleure des trois régressions) et, entre parenthèses, erreur "
           "médiane sur la teinte relue (90° = hasard ; deux teintes voisines du jeu de test sont à 14,4°). « Tout » : "
           "résidu et écarts des continus, avec les vraies valeurs des attributs (contexte). Latent GW : couleur "
           "présente dans l'espace de travail avant le décodeur d'attributs.\n"]

    def summary(keys, group):
        r2 = np.mean([best_of(df, *k, group) for k in keys])
        err = np.median([value(df, *k, group, best_regression(df, *k, group), key="erreur_teinte_deg") for k in keys])
        return f"{fr(r2)} ({err:.0f}°)"

    for exp_name in CONFIG_LABELS:
        keys = [(c, i) for c, i in models_in(df) if c == exp_name and i != "control"]
        if keys:
            out.append(f"- **{CONFIG_LABELS[exp_name]}** : résidu {summary(keys, 'residu')} ; continus "
                       f"{summary(keys, 'continus')}, avec le contexte {summary(keys, 'continus_contexte')} ; tout "
                       f"{summary(keys, 'les_deux_contexte')} ; latent GW {summary(keys, 'latent_gw')}.")
    controls = [(c, i) for c, i in models_in(df) if i == "control"]
    if controls:
        vals = {g: [best_of(df, *k, g) for k in controls]
                for g in ("residu", "continus_contexte", "les_deux_contexte", "couleur_decodee", "latent_gw")}
        span = {g: f"{fr(min(v))} à {fr(max(v))}" for g, v in vals.items()}
        out.append(f"- **Modèles Complets** ({len(controls)}, couleur dans les attributs) : couleur décodée "
                   f"{span['couleur_decodee']} ; résidu {span['residu']} ; continus avec le contexte "
                   f"{span['continus_contexte']} ; tout {span['les_deux_contexte']} ; latent GW {span['latent_gw']}. "
                   "Ce « tout » sert de niveau de référence : la couleur modifie un peu la géométrie perçue (taille, "
                   "position), sans aucun besoin de la cacher.")
    return out + [""]


def hypothesis_lines(df, transmission):
    """Confrontation à l'hypothèse : le résidu porte l'essentiel de la couleur (Cycle-Complet), une partie
    (Référence, Attributs Continus, Traduction), rien (Contrastive)."""
    rows = []
    for exp_name in CONFIG_LABELS:
        keys = [(c, i) for c, i in models_in(df) if c == exp_name and i != "control"]
        if not keys:
            continue
        r2 = [best_of(df, *k, "residu") for k in keys]
        observed = [r2_class(v) for v in r2]
        expected = EXPECTED[exp_name]
        n_ok = sum(o == expected for o in observed)
        gaps = [CLASSES.index(o) - CLASSES.index(expected) for o in observed]
        verdict = "oui" if n_ok == len(keys) else ("non" if n_ok == 0 else f"{n_ok} graine(s) sur {len(keys)}")
        if n_ok < len(keys):
            verdict += " (plus que prévu)" if all(g >= 0 for g in gaps) else (
                " (moins que prévu)" if all(g <= 0 for g in gaps) else "")
        trans = [transmission[k] for k in keys if k in transmission]
        rows.append([CONFIG_LABELS[exp_name], f"{fr(min(trans))} à {fr(max(trans))}" if trans else "–", expected,
                     " ; ".join(fr(v) for v in r2), ", ".join(dict.fromkeys(observed)), verdict,
                     " ; ".join(fr(best_of(df, *k, "les_deux_contexte")) for k in keys)])
    out = ["## Hypothèse de départ\n",
           "D'après la transmission de la teinte en cycle complet (R̄), le résidu devait porter l'essentiel de la "
           "couleur pour Emphase Cycle-Complet, une partie pour Référence, Emphase Attributs Continus et Emphase "
           "Traduction, rien pour Emphase Contrastive. Classes : l'essentiel (R² ≥ 0,7), une partie (0,1 à 0,7), "
           "rien (< 0,1) ; R² du résidu seul, meilleure des trois régressions, par graine. Dernière colonne : R² de "
           "tous les attributs décodés avec le contexte.\n",
           md_table(["Configuration", "R̄ cycle complet", "attendu", "R² du résidu (graines 0 ; 1 ; 2)", "observé",
                     "conforme", "R² de tout (graines 0 ; 1 ; 2)"], rows)]
    keys = [(c, i) for c, i in models_in(df) if i != "control" and (c, i) in transmission]
    if len(keys) > 2:
        out.append("")
        for g in ("residu", "continus", "les_deux_contexte"):
            rho, p = spearmanr([transmission[k] for k in keys], [best_of(df, *k, g) for k in keys])
            p_text = f"{p:.1e}".replace(".", ",").replace("e-0", "e-")
            out.append(f"- Corrélation de Spearman entre R̄ du cycle complet et R² de la chrominance "
                       f"({GROUPS[g]}), {len(keys)} modèles « vide sémantique » : ρ = {fr(rho)} (p = {p_text}).")
    return out + [""]


def knn_dilution_notes(df):
    """Le kNN se dégrade quand on lui ajoute des variables peu informatives ; le perceptron, non."""
    keys = models_in(df)

    def n_worse(regression):
        pairs = [(value(df, *k, "les_deux_contexte", regression), value(df, *k, "residu", regression)) for k in keys]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        return sum(a < b - 0.02 for a, b in pairs), len(pairs)

    gw = {r: [value(df, *k, "latent_gw", r) for k in keys if i_semantic(k)] for r in REGRESSIONS}
    out = []
    for r in ("kNN", "MLP"):
        worse, n = n_worse(r)
        if n:
            out.append(f"{r} : « les deux + contexte » en dessous du résidu seul (de plus de 0,02) pour {worse} "
                       f"modèle{'s' if worse > 1 else ''} sur {n}")
    wide = ["les_deux", "residu_contexte", "continus_contexte", "les_deux_contexte", "latent_gw"]
    winners = [best_regression(df, *k, g) for k in keys for g in wide if np.isfinite(best_of(df, *k, g))]
    share = {r: winners.count(r) for r in REGRESSIONS}
    out = [f"- Variables peu informatives : le kNN se dégrade quand on lui en ajoute (distances diluées), le "
           f"perceptron beaucoup moins ({' ; '.join(out)}). Sur le latent GW des modèles « vide sémantique » "
           "(12 variables) : "
           + ", ".join(f"{r} {fr(np.nanmin(v))} à {fr(np.nanmax(v))}" for r, v in gw.items() if np.isfinite(v).any())
           + ". Les résumés retiennent la meilleure des trois régressions ; pour les groupes de 7 variables ou plus, "
           f"c'est {share['MLP']} fois sur {len(winners)} le perceptron, {share['kNN']} fois le kNN et "
           f"{share['ridge']} fois la ridge."]
    return out


def i_semantic(key):
    """Modèle « vide sémantique » (pas un contrôle)."""
    return key[1] != "control"


def figure(df, transmission, out_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    groups = ["residu", "continus", "continus_contexte", "les_deux_contexte"]
    # Palette catégorielle de référence (guide de visualisation), dans l'ordre ; validée (daltonismes compris)
    colors = dict(zip(groups, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    labels = dict(GROUPS, les_deux_contexte="tout (résidu + continus + contexte)")
    models = models_in(df)
    plt.rcParams.update({"text.color": ink, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.edgecolor": muted})
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [3.2, 1]}, facecolor=surface)
    ax = axes[0]
    width = 0.2
    x = np.arange(len(models))
    for j, g in enumerate(groups):
        vals = [best_of(df, c, i, g) for c, i in models]
        ax.bar(x + (j - 1.5) * width, np.clip(vals, -0.05, None), width, color=colors[g], label=labels[g],
               edgecolor=surface, linewidth=0.8)
    ax.plot(x, [best_of(df, c, i, "latent_gw") for c, i in models], "_", color=ink, ms=16, mew=2.2,
            label=GROUPS["latent_gw"])
    decoded = [best_of(df, c, i, "couleur_decodee") if i == "control" else np.nan for c, i in models]
    ax.plot(x, decoded, "*", color=ink, ms=8, label=GROUPS["couleur_decodee"])
    shuffled = [df[(df["config"] == c) & (df["instance"] == i) & (df["etiquettes"] == "mélangées")]["r2_chrominance"]
                .max() for c, i in models]
    ax.plot(x, shuffled, "x", color=muted, ms=6, label="couleurs mélangées (plus grand R²)")
    ax.set_xticks(x)
    ax.set_xticklabels([("contrôle" if i == "control" else i.replace("seed", "graine ")) for c, i in models],
                       rotation=60, ha="right", fontsize=8)
    bounds = [k for k in range(1, len(models)) if models[k][0] != models[k - 1][0]]
    for k in bounds:
        ax.axvline(k - 0.5, color=muted, lw=0.6)
    for s, e in zip([0] + bounds, bounds + [len(models)]):
        ax.text((s + e - 1) / 2, 1.07, CONFIG_LABELS.get(models[s][0], models[s][0]).replace("Emphase ", "Emph. "),
                ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(-0.05, 1.13)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.axhline(0, color=muted, lw=0.6)
    ax.set_ylabel("R² de la chrominance (a, b), validation croisée")
    ax.set_title("Couleur d'entrée relue dans les attributs décodés (meilleure de ridge, kNN et MLP)", pad=24)
    ax.grid(axis="y", color="#d9d8d4", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, names = ax.get_legend_handles_labels()
    order = [names.index(labels[g]) for g in groups] + [k for k, n in enumerate(names)
                                                        if n not in [labels[g] for g in groups]]
    fig.legend([handles[k] for k in order], [names[k] for k in order], loc="lower left", bbox_to_anchor=(0.01, 0.0),
               ncol=4, fontsize=8.5, frameon=False)

    ax = axes[1]
    keys = [(c, i) for c, i in models if i != "control" and (c, i) in transmission]
    if keys:
        markers = dict(zip(CONFIG_LABELS, ["o", "s", "^", "D", "v"]))
        for exp_name, marker in markers.items():
            ks = [k for k in keys if k[0] == exp_name]
            for g in ("residu", "les_deux_contexte"):
                ax.scatter([transmission[k] for k in ks], [best_of(df, *k, g) for k in ks], s=52, marker=marker,
                           color=colors[g], edgecolors=ink, linewidths=0.5, zorder=3)
        ax.set_xlabel(r"$\bar{R}$ de transmission de la teinte en cycle complet")
        ax.set_ylabel("R² de la chrominance")
        ax.set_title("Modèles « vide sémantique » (une marque par graine)", fontsize=10)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(color="#d9d8d4", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        legend = [Line2D([], [], ls="", marker="o", ms=7, color=colors[g], mec=ink, mew=0.5, label=labels[g])
                  for g in ("residu", "les_deux_contexte")]
        legend += [Line2D([], [], ls="", marker=m, ms=6.5, color="white", mec=ink, mew=0.8, label=CONFIG_LABELS[c])
                   for c, m in markers.items() if any(k[0] == c for k in keys)]
        ax.legend(handles=legend, fontsize=7.5, loc="upper left", frameon=False)
    else:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_file, dpi=120, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append", help="Configuration(s) (défaut : toutes)")
    parser.add_argument("--no-controls", action="store_true", help="Sans les Modèles Complets")
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--per-cell", type=int, default=32,
                        help="Images par case teinte × forme × taille (600 cases ; 32 -> 19 200 images)")
    parser.add_argument("--seed", type=int, default=0, help="Graine du sous-ensemble, des plis et du mélange")
    parser.add_argument("--target", choices=("palette", "mesuree"), default="palette",
                        help="Couleur d'entrée : palette du jeu de test (défaut) ou mesurée sur l'image")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "saturation" / "regression")
    parser.add_argument("--analyze-only", action="store_true", help="Sans passage dans les modèles")
    parser.add_argument("--refit", action="store_true", help="Recalcule les régressions déjà en cache")
    parser.add_argument("--regressions", nargs="+", choices=REGRESSIONS, default=list(REGRESSIONS),
                        help="Régressions à calculer (défaut : les trois ; ridge et kNN seuls vont bien plus vite)")
    parser.add_argument("--threads", type=int, default=None, help="Threads PyTorch sur CPU")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Passage dans les modèles et perceptron multicouche (auto : GPU s'il y en a un)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    data_dir = args.out / "donnees"
    img = load_image_set(data_dir, args.per_cell, args.seed)
    models = thesis_models(args.experiment, include_controls=not args.no_controls)[:args.max_models]
    if not args.analyze_only:
        compute(models, img, data_dir, device, args.batch_size)
    df = fit_all(data_dir, img, args, {(m[0], m[1]) for m in models}, device)
    write_reports(df, img, args.out, args)


if __name__ == "__main__":
    main()
