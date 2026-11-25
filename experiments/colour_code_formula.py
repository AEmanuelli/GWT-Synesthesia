"""Formule du code couleur : deux contributions compressées (logarithmiques), additionnées dans l'espace de travail.

Constat (étages du réseau). En traduction, le vecteur de forme p = (1 - m)·e_t + m·w·e_j + m·(1 - w)·e_k et
les attributs continus entrent dans l'encodeur du GW (perceptron ReLU 8 → 256 → 256 → 256 → 12), dont la sortie
e passe par tanh pour donner l'état z de l'espace de travail, décodé ensuite en image. La réponse de e au
résidu n'est pas linéaire : très forte pour les masses infimes (jusqu'à environ 0,1 %), elle se compresse
ensuite. La non-linéarité est donc dans l'encodeur, pas dans la sigmoïde de sortie.

Modèle. Chaque forme absente contribue par une fonction compressive de sa probabilité, le long d'une
direction fixe de l'espace de travail, et les deux contributions s'additionnent :

    e(m, w) = e0 + φ(m·w)·g_j + φ(m·(1 - w))·g_k,     φ(x) = x0 · log(1 + x / x0)

(φ(x) ≈ x pour x ≪ x0, ≈ x0·log(x / x0) au-delà). Conséquences :
  - à faible quantité, la teinte est celle du mélange g_j : g_k dans la proportion w : (1 - w) ;
  - quand la quantité grandit, la compression rapproche les deux contributions (φ(0,8 m)/φ(0,2 m) → 1) : la
    teinte glisse vers celle du mélange moitié-moitié, d'où une teinte qui tourne avec la quantité ;
  - si g_j et g_k sont presque parallèles, la teinte ne dépend ni de w ni de m : une couleur par forme.

Test. Grille (m, w) sur 6 géométries par forme. On ajuste x0, g_j, g_k (communs aux géométries) sur 3
géométries, puis on prédit e sur les 3 autres et on fait passer la prédiction dans le reste du réseau (tanh,
décodeurs) pour obtenir la couleur prédite, comparée à la couleur observée. Comparaisons : le même modèle sans
compression (φ(x) = x) et un mélange linéaire fait directement en RGB.

    python experiments/colour_code_formula.py            # Référence et Emphase Cycle-Complet
    python experiments/colour_code_formula.py --analyze-only
Sorties : results/publication/formule/ (donnees/, formule.csv, etages.csv, synthese.md, figures).
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
from shape_noise_sweep import CONFIG_LABELS, SHAPE_LABELS, hls_arrays, load_model, md_table, thesis_models  # noqa: E402
from SSD_utils import extract_shape_color, normalize_position, normalize_rotation, normalize_size, segment_shape  # noqa: E402

M_GRID = np.r_[0.0, np.logspace(-5, np.log10(0.05), 22)]
W_GRID = np.linspace(0, 1, 11)
X0_GRID = np.logspace(-6, -1, 41)
S_COLOURED = 0.3
CONFIG_ORDER = ["base_params", "high_cycles"]
STAGES = ("e", "z", "v", "logit", "rgb")


def geometries(per_shape, seed):
    rows = [r for r in csv.DictReader(open(paths.EVALSET_DIR / "attributes.csv")) if float(r["size"]) >= 10]
    rng = np.random.default_rng(seed)
    out = []
    for t in range(3):
        cand = [r for r in rows if int(r["class"]) == t]
        for g, i in enumerate(rng.choice(len(cand), per_shape, replace=False)):
            r = cand[i]
            px, py = normalize_position(list(ast.literal_eval(r["location"])))
            rx, ry = normalize_rotation(float(r["rotation"]))
            out.append((t, g, [px, py, normalize_size(float(r["size"])), rx, ry]))
    return out


def inputs(geos):
    oh, at, meta = [], [], []
    for t, g, cont in geos:
        j, k = [c for c in range(3) if c != t]
        for m in M_GRID:
            for w in ([0.5] if m == 0 else W_GRID):
                p = np.zeros(3)
                p[t], p[j], p[k] = 1 - m, m * w, m * (1 - w)
                oh.append(p)
                at.append(cont)
                meta.append((t, g, m, w))
    return np.array(oh, np.float32), np.array(at, np.float32), np.array(meta)


def barycentre_patch(mask):
    """Mêmes pixels que SSD_utils.extract_shape_color(..., barycentre=True)."""
    m00 = mask.sum()
    cx = int(np.sum(mask * np.arange(mask.shape[1]).reshape(1, -1)) / m00)
    cy = int(np.sum(mask * np.arange(mask.shape[0]).reshape(-1, 1)) / m00)
    h, w = mask.shape
    cy, cx = max(0, min(cy, h - 1)), max(0, min(cx, w - 1))
    return slice(max(0, cy - 1), min(h, cy + 2)), slice(max(0, cx - 1), min(w, cx + 2))


class Pipeline:
    """Traduction attributs -> GW -> image, avec accès à chaque étage (e, z, v, logit avant sigmoïde, RGB)."""

    def __init__(self, gw, device):
        from shimmer.modules.selection import SingleDomainSelection
        self.gw, self.device = gw, device
        self.selection = SingleDomainSelection().to(device)
        self.sigmoid = next(m for m in gw.domain_mods["v_latents"].modules() if isinstance(m, torch.nn.Sigmoid))

    def colours(self, images, logits):
        ims = images.permute(0, 2, 3, 1).cpu().numpy()
        lgs = logits.permute(0, 2, 3, 1).cpu().numpy()
        rgb, pre = [], []
        for im, lg in zip(ims, lgs):
            mask = segment_shape(im)
            ys, xs = barycentre_patch(mask) if mask.sum() >= 10 else (slice(None), slice(None))
            rgb.append(im[ys, xs].reshape(-1, 3).mean(0))
            pre.append(lg[ys, xs].reshape(-1, 3).mean(0))
        return np.array(rgb), np.array(pre)

    def from_e(self, e):
        """Reste du réseau à partir de la sortie e de l'encodeur (z = tanh(e), sélection d'un seul domaine)."""
        captured = {}
        hook = self.sigmoid.register_forward_hook(lambda mod, inp, out: captured.__setitem__("pre", inp[0].detach()))
        try:
            with torch.no_grad():
                z = torch.tanh(e)
                v = self.gw.gw_mod.decode(z)["v_latents"]
                images = self.gw.domain_mods["v_latents"].decode_images(v)
        finally:
            hook.remove()
        rgb, pre = self.colours(images, captured["pre"])
        return z.cpu().numpy(), v.cpu().numpy(), pre, rgb

    def run(self, oh, at, batch_size):
        out = {s: [] for s in STAGES}
        gap = 0.0
        for start in range(0, len(oh), batch_size):
            o = torch.from_numpy(oh[start:start + batch_size]).to(self.device)
            a = torch.from_numpy(at[start:start + batch_size]).to(self.device)
            with torch.no_grad():
                e = self.gw.gw_mod.gw_encoders["attr"](self.gw.encode_domain([o, a], "attr"))
                z_ref = self.gw.gw_mod.encode_and_fuse({"attr": self.gw.encode_domain([o, a], "attr")},
                                                      selection_module=self.selection)
            z, v, pre, rgb = self.from_e(e)
            gap = max(gap, float((torch.tanh(e) - z_ref).abs().max()))   # contrôle : z = tanh(e)
            for s, x in zip(STAGES, (e.cpu().numpy(), z, v, pre, rgb)):
                out[s].append(x)
        return {s: np.concatenate(x) for s, x in out.items()}, gap


# ----------------------------------------------------------------------------------------------
def phi(x, x0):
    return x if x0 is None else x0 * np.log1p(x / x0)


def fit_e(meta, E, t, train_g, x0):
    """Ajuste g_j, g_k communs aux géométries d'apprentissage (moindres carrés) pour une compression x0."""
    sel = (meta[:, 0] == t) & np.isin(meta[:, 1], train_g) & (meta[:, 2] > 0)
    e0 = {g: E[(meta[:, 0] == t) & (meta[:, 1] == g) & (meta[:, 2] == 0)][0] for g in np.unique(meta[:, 1])}
    base = np.array([e0[g] for g in meta[:, 1]])
    X = np.column_stack([phi(meta[:, 2] * meta[:, 3], x0), phi(meta[:, 2] * (1 - meta[:, 3]), x0)])
    G = np.linalg.lstsq(X[sel], (E - base)[sel], rcond=None)[0]
    return G, base, X


def r2(obs, pred):
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean(0)) ** 2))


def dist(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def compute(models, geos, data_dir, device, batch_size, n_geo):
    oh, at, meta = inputs(geos)
    train_g, test_g = np.arange(n_geo // 2), np.arange(n_geo // 2, n_geo)
    for exp_name, instance, ckpt, colored in models:
        f = data_dir / f"{exp_name}_{instance}.npz"
        if f.exists():
            print(f"Déjà calculé : {f.name}")
            continue
        print(f"Formule : {exp_name}/{instance} ({len(oh)} traductions)")
        t0 = time.time()
        gw = load_model(ckpt, colored, device)
        pipe = Pipeline(gw, device)
        obs, gap = pipe.run(oh, at, batch_size)
        preds = {}
        for t in range(3):
            # compression : x0 choisi sur les géométries d'apprentissage
            train = (meta[:, 0] == t) & np.isin(meta[:, 1], train_g) & (meta[:, 2] > 0)

            def train_r2(x0):
                G, base, X = fit_e(meta, obs["e"], t, train_g, x0)
                return r2((obs["e"] - base)[train], (X @ G)[train])
            best = max(X0_GRID, key=train_r2)
            for name, x0 in (("log", best), ("lin", None)):
                G, base, X = fit_e(meta, obs["e"], t, train_g, x0)
                sel = meta[:, 0] == t
                e_pred = base + X @ G
                rgb_pred = np.zeros((len(meta), 3))
                e_t = torch.from_numpy(e_pred[sel].astype(np.float32)).to(device)
                parts = [pipe.from_e(e_t[i:i + batch_size])[3] for i in range(0, len(e_t), batch_size)]
                rgb_pred[sel] = np.concatenate(parts)
                preds[(name, t)] = dict(G=G, x0=x0 if x0 is not None else np.nan, e_pred=e_pred, rgb_pred=rgb_pred)
        np.savez_compressed(f, meta=meta, config=exp_name, instance=instance, gap=gap,
                            **{f"obs_{s}": x for s, x in obs.items()},
                            **{f"{n}_{t}_{k}": v for (n, t), d in preds.items() for k, v in d.items()})
        print(f"  {time.time() - t0:.0f} s ; écart tanh(e) - z : {gap:.1e}")
        del gw, pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
def chroma_angle(rgb):
    a = rgb[..., 0] - (rgb[..., 1] + rgb[..., 2]) / 2
    b = np.sqrt(3) / 2 * (rgb[..., 1] - rgb[..., 2])
    return np.degrees(np.arctan2(b, a)) % 360


def fr(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")


def analyze(data_dir, out_dir, n_geo):
    test_g = np.arange(n_geo // 2, n_geo)
    rows, stage_rows, store = [], [], {}
    files = sorted(data_dir.glob("*.npz"), key=lambda f: (str(np.load(f)["config"]) != "high_cycles", f.name))
    for f in files:
        d = dict(np.load(f))
        key = (str(d["config"]), str(d["instance"]))
        meta = d["meta"]
        h_obs, _, s_obs = hls_arrays(255 * d["obs_rgb"])
        # Étages : gain |Δ|/m et rotation de la direction (cos avec la direction à m = 0,001 %)
        for s in STAGES:
            X = d[f"obs_{s}"]
            gains, coss = {mv: [] for mv in (1e-4, 1e-3, 1e-2, 5e-2)}, {mv: [] for mv in (1e-3, 1e-2, 5e-2)}
            for t in range(3):
                for g in range(n_geo):
                    for w in (0.0, 0.5, 1.0):
                        ray = (meta[:, 0] == t) & (meta[:, 1] == g) & np.isclose(meta[:, 3], w)
                        base = X[(meta[:, 0] == t) & (meta[:, 1] == g) & (meta[:, 2] == 0)][0]
                        ref = X[ray & np.isclose(meta[:, 2], M_GRID[1])][0] - base
                        for mv in gains:
                            mm = M_GRID[np.argmin(np.abs(M_GRID - mv))]
                            dl = X[ray & np.isclose(meta[:, 2], mm)][0] - base
                            gains[mv].append(np.linalg.norm(dl) / mm)
                            if mv in coss:
                                coss[mv].append(dl @ ref / (np.linalg.norm(dl) * np.linalg.norm(ref) + 1e-12))
            stage_rows.append(dict(config=key[0], instance=key[1], etage=s,
                                   **{f"gain_{mv:g}": float(np.median(v)) for mv, v in gains.items()},
                                   **{f"cos_{mv:g}": float(np.median(v)) for mv, v in coss.items()}))
        for t in range(3):
            te = (meta[:, 0] == t) & np.isin(meta[:, 1], test_g) & (meta[:, 2] > 0)
            col = te & (s_obs >= S_COLOURED)
            r = dict(config=key[0], instance=key[1], forme=t, n_test_colorees=int(col.sum()))
            for name in ("log", "lin"):
                G, x0 = d[f"{name}_{t}_G"], float(d[f"{name}_{t}_x0"])
                e_pred, rgb_pred = d[f"{name}_{t}_e_pred"], d[f"{name}_{t}_rgb_pred"]
                base = np.array([d["obs_e"][(meta[:, 0] == t) & (meta[:, 1] == g) & (meta[:, 2] == 0)][0]
                                 for g in meta[:, 1]])
                h_p, _, s_p = hls_arrays(255 * np.clip(rgb_pred, 0, 1))
                r[f"r2_e_{name}"] = r2((d["obs_e"] - base)[te], (e_pred - base)[te])
                r[f"teinte_err_{name}"] = float(np.median(dist(h_p[col], h_obs[col]))) if col.any() else np.nan
                r[f"teinte_15_{name}"] = float(np.mean(dist(h_p[col], h_obs[col]) <= 15)) if col.any() else np.nan
                r[f"S_r2_{name}"] = r2(s_obs[te], s_p[te])
                if name == "log":
                    r["x0"] = x0
                    gj, gk = G[0], G[1]
                    r["cos_g"] = float(gj @ gk / (np.linalg.norm(gj) * np.linalg.norm(gk)))
                    store[key + (t,)] = dict(meta=meta, obs=d["obs_rgb"], pred=rgb_pred, lin=d[f"lin_{t}_rgb_pred"])
            # Mélange linéaire en RGB (même structure, sans réseau) : comparaison
            tr = (meta[:, 0] == t) & ~np.isin(meta[:, 1], test_g) & (meta[:, 2] > 0)
            r0 = np.array([d["obs_rgb"][(meta[:, 0] == t) & (meta[:, 1] == g) & (meta[:, 2] == 0)][0] for g in meta[:, 1]])
            X = np.column_stack([meta[:, 2] * meta[:, 3], meta[:, 2] * (1 - meta[:, 3])])
            A = np.linalg.lstsq(X[tr], (d["obs_rgb"] - r0)[tr], rcond=None)[0]
            h_l, _, s_l = hls_arrays(255 * np.clip(r0 + X @ A, 0, 1))
            r["teinte_err_rgb"] = float(np.median(dist(h_l[col], h_obs[col]))) if col.any() else np.nan
            r["S_r2_rgb"] = r2(s_obs[te], s_l[te])
            rows.append(r)
    write(rows, stage_rows, out_dir)
    figure(store, out_dir, n_geo)
    return rows


def write(rows, stage_rows, out_dir):
    for name, data in (("formule.csv", rows), ("etages.csv", stage_rows)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    st = [[f"{CONFIG_LABELS[r['config']]} g{r['instance'][-1]}", r["etage"],
           " / ".join(f"{r[f'gain_{mv:g}']:.0f}" for mv in (1e-4, 1e-3, 1e-2, 5e-2)),
           " / ".join(fr(r[f"cos_{mv:g}"]) for mv in (1e-3, 1e-2, 5e-2))] for r in stage_rows]
    tb = [[f"{CONFIG_LABELS[r['config']]} g{r['instance'][-1]}", SHAPE_LABELS[r["forme"]], str(r["n_test_colorees"]),
           f"{100 * r['x0']:.2g} %".replace(".", ","), fr(r["cos_g"]),
           f"{fr(r['r2_e_log'])} / {fr(r['r2_e_lin'])}",
           f"{fr(r['teinte_err_log'], 0)}° / {fr(r['teinte_err_lin'], 0)}° / {fr(r['teinte_err_rgb'], 0)}°",
           f"{fr(100 * r['teinte_15_log'], 0)} %", f"{fr(r['S_r2_log'])} / {fr(r['S_r2_lin'])} / {fr(r['S_r2_rgb'])}"]
          for r in rows]
    L = ["# Formule du code couleur", "",
         "## Où la réponse au résidu devient-elle non linéaire ?", "",
         "Gain |Δ|/m à m = 0,01 % / 0,1 % / 1 % / 5 % (constant si la réponse est proportionnelle à m) ; cos entre la "
         "direction de la réponse à m et à 0,001 % (1 : même direction), à m = 0,1 % / 1 % / 5 %. Étages : e (sortie "
         "de l'encodeur du GW), z = tanh(e) (espace de travail), v (latent du VAE), logit (avant la sigmoïde de "
         "sortie, au barycentre), RGB.", "",
         md_table(["Modèle", "Étage", "gain |Δ|/m", "cos (direction)"], st), "",
         "## Formule e = e0 + φ(m·w)·g_j + φ(m·(1 − w))·g_k, φ(x) = x0·log(1 + x/x0)", "",
         "Ajustée (x0, g_j, g_k communs aux géométries) sur 3 géométries par forme, testée sur les 3 autres, toutes "
         "quantités de 0,001 % à 5 % ; couleur prédite en faisant passer la prédiction dans le reste du réseau. "
         "Comparaisons : sans compression (φ(x) = x), et mélange linéaire en RGB.", "",
         md_table(["Modèle", "Forme", "n", "x0", "cos(g_j, g_k)", "R² de e : log / linéaire",
                   "erreur de teinte : log / linéaire / RGB", "teinte à ±15° (log)", "R² de S : log / linéaire / RGB"], tb)]
    (out_dir / "synthese.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figure(store, out_dir, n_geo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in store if (k[0], k[1]) in {("high_cycles", "seed0"), ("base_params", "seed1")}]
    keys.sort(key=lambda k: (k[0] != "high_cycles", k[2]))
    ms = M_GRID[1:]
    g = n_geo // 2
    fig, axes = plt.subplots(len(keys), 3, figsize=(8.6, 1.75 * len(keys) + 0.7), squeeze=False)
    for r, key in enumerate(keys):
        st = store[key]
        meta = st["meta"]
        for c, (name, title) in enumerate((("obs", "observé"), ("pred", "formule (log)"), ("lin", "sans compression"))):
            grid = np.zeros((len(ms), len(W_GRID), 3))
            for a, m in enumerate(ms):
                for b, w in enumerate(W_GRID):
                    sel = (meta[:, 0] == key[2]) & (meta[:, 1] == g) & np.isclose(meta[:, 2], m) & np.isclose(meta[:, 3], w)
                    grid[a, b] = st[name][sel][0]
            ax = axes[r][c]
            ax.imshow(np.clip(grid, 0, 1), origin="lower", aspect="auto", interpolation="nearest",
                      extent=(0, 1, 0, len(ms)))
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0.5, 7.5, 14.5, 21.5])
            ax.set_yticklabels(["0,001 %", f"{100 * ms[7]:.1g} %".replace(".", ","),
                                f"{100 * ms[14]:.1g} %".replace(".", ","), "5 %"], fontsize=6.5)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{CONFIG_LABELS[key[0]]} g{key[1][-1]}\n{SHAPE_LABELS[key[2]]}", fontsize=7)
            if r == len(keys) - 1:
                ax.set_xlabel("répartition w", fontsize=8)
    fig.suptitle("Couleur selon la quantité m (ordonnée) et la répartition w du résidu, géométrie non vue à "
                 "l'ajustement :\nobservée, prédite par la formule, prédite sans compression", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "formule_cartes.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", action="append")
    ap.add_argument("--per-shape", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "publication" / "formule")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    data_dir = args.out / "donnees"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = thesis_models(args.experiment or CONFIG_ORDER, include_controls=False)
        compute(models, geometries(args.per_shape, args.seed), data_dir, device, args.batch_size, args.per_shape)
    analyze(data_dir, args.out, args.per_shape)
    print(f"Résultats dans {args.out}")


if __name__ == "__main__":
    main()
