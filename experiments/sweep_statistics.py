"""Statistiques du balayage du bruit sur la forme (`shape_noise_sweep.py`).

Établit, avec intervalles de confiance et tests corrigés pour comparaisons multiples :
  A. effet de la masse résiduelle sur la saturation (relation dose-réponse, appariée par exemple) ;
  B. spécificité : même amplitude sur les attributs continus (témoin) ;
  C. plage naturelle de la masse résiduelle produite par le décodeur d'attributs sur de vraies images ;
  D. dépendance de la teinte à la répartition du résidu entre les deux formes non cibles.

Ne relance aucun modèle : lit `results/saturation/balayage/donnees/` (balayage) et, pour C,
`results/saturation/regression/donnees/` (`residual_regression.py`). Unité statistique : l'exemple
(48 par instance : 3 formes × 4 orientations × 2 tailles × 2 positions) ; une valeur par exemple
(médiane sur les 7 directions admissibles), pour ne pas compter plusieurs fois le même exemple.

Usage : python experiments/sweep_statistics.py [--results-dir results/saturation]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_noise_sweep import (CONFIG_LABELS, EPSILONS, KIND_BASE, KIND_CONT, KIND_SHAPE,  # noqa: E402
                               load_sweeps)

ADMISSIBLE = np.arange(240, 301, 10)
S_COLOURED = 0.3        # seuil « colorée » (entrée : 0,91 ; traduction sans bruit : ≈ 0,05)
N_BOOT = 5000
CONFIG_ORDER = ["base_params", "high_translation", "high_contrastive", "high_cycles", "high_alpha"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci(x, stat=np.median, n=N_BOOT, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = stat(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return np.percentile(vals, [2.5, 97.5])


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def fr(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def fp(p):
    if p < 1e-4:
        return "< 10⁻⁴"
    return fr(p, 4) if p < 0.01 else fr(p, 3)


def label(key):
    cfg, inst = key
    return f"{CONFIG_LABELS.get(cfg, cfg)} — {'contrôle' if inst == 'control' else 'graine ' + inst[-1]}"


def per_example(d, kind, eps, value, dirs=None):
    """Médiane par exemple de `value` sur les directions retenues, à l'amplitude eps."""
    sel = (d["kind"] == kind) & np.isclose(d["eps"], eps)
    if dirs is not None:
        deg = np.round(np.degrees(np.nan_to_num(d["theta"]))).astype(int)
        sel &= np.isin(deg, dirs)
    out = np.full(int(d["base"].max()) + 1, np.nan)
    for b in np.unique(d["base"][sel]):
        out[b] = np.median(d[value][sel & (d["base"] == b)])
    return out


def base_value(d, value):
    sel = d["kind"] == KIND_BASE
    out = np.full(int(d["base"].max()) + 1, np.nan)
    out[d["base"][sel]] = d[value][sel]
    return out


def hue_at_direction(d, eps, deg):
    th = np.round(np.degrees(np.nan_to_num(d["theta"]))).astype(int)
    sel = (d["kind"] == KIND_SHAPE) & np.isclose(d["eps"], eps) & (th == deg)
    n = int(d["base"].max()) + 1
    h, s = np.full(n, np.nan), np.full(n, np.nan)
    h[d["base"][sel]] = d["H"][sel]
    s[d["base"][sel]] = d["S"][sel]
    return h, s


def circ_diff(a, b):
    return (a - b + 180) % 360 - 180


def analyse_dose(sweeps, keys):
    rows, pvals = [], []
    for key in keys:
        d = sweeps[key]
        s0 = base_value(d, "S")
        res = {"key": key, "S0": np.median(s0)}
        for e in EPSILONS:
            se = per_example(d, KIND_SHAPE, e, "S", ADMISSIBLE)
            res[e] = se
        # Tests appariés à ε = 0,001 et ε = 0,01 (H1 : S augmente)
        for tag, e in (("1e-3", EPSILONS[0]), ("1e-2", EPSILONS[4])):
            diff = res[e] - s0
            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = stats.wilcoxon(res[e], s0, alternative="greater", zero_method="zsplit").pvalue
            res["p" + tag] = p
            k = int(np.sum(res[e] >= S_COLOURED))
            res["frac" + tag] = (k, len(diff), *wilson(k, len(diff)))
            res["ci" + tag] = boot_ci(res[e])
        # Dose-réponse : ρ de Spearman (ε, S) par exemple, puis test des signes sur les 48 exemples
        mat = np.vstack([s0] + [res[e] for e in EPSILONS]).T
        eps_axis = np.r_[0, EPSILONS]
        rhos = np.array([stats.spearmanr(eps_axis, row).statistic if np.ptp(row) > 0 else 0.0 for row in mat])
        res["rho_med"] = np.median(rhos)
        npos, nneg = int(np.sum(rhos > 0)), int(np.sum(rhos < 0))
        res["rho_sign"] = (npos, nneg)
        res["p_rho"] = stats.binomtest(npos, npos + nneg, 0.5, alternative="greater").pvalue if npos + nneg else 1.0
        # Plus petite amplitude où la médiane des exemples dépasse le seuil
        eps_star = [e for e in EPSILONS if np.median(res[e]) >= S_COLOURED]
        res["eps_star"] = eps_star[0] if eps_star else np.nan
        rows.append(res)
        pvals += [res["p1e-3"], res["p1e-2"], res["p_rho"]]
    adj = holm(pvals)
    for i, res in enumerate(rows):
        res["p1e-3_holm"], res["p1e-2_holm"], res["p_rho_holm"] = adj[3 * i: 3 * i + 3]
    return rows


def analyse_specificity(sweeps, keys):
    rows, pvals = [], []
    for key in keys:
        d = sweeps[key]
        res = {"key": key}
        for tag, e in (("1e-3", EPSILONS[0]), ("1e-2", EPSILONS[4])):
            cs = per_example(d, KIND_SHAPE, e, "dC", ADMISSIBLE)
            cc = per_example(d, KIND_CONT, e, "dC")
            res["shape" + tag], res["cont" + tag] = np.median(cs), np.median(cc)
            ratio = np.log10((cs + 1e-3) / (cc + 1e-3))
            lo, hi = boot_ci(ratio)
            res["ratio" + tag] = (10 ** np.median(ratio), 10 ** lo, 10 ** hi)
            p = stats.wilcoxon(cs, cc, alternative="greater", zero_method="zsplit").pvalue \
                if not np.allclose(cs, cc) else 1.0
            res["p" + tag] = p
            pvals.append(p)
        rows.append(res)
    adj = holm(pvals)
    for i, res in enumerate(rows):
        res["p1e-3_holm"], res["p1e-2_holm"] = adj[2 * i: 2 * i + 2]
    return rows


def analyse_natural(reg_dir, keys):
    """Masse résiduelle 1 - p(forme vraie) en sortie du décodeur d'attributs (image → GW → attributs)."""
    img_file = reg_dir / "images.npz"
    if not img_file.exists():
        return None
    img = np.load(img_file)
    shape_key = next(k for k in ("forme", "shape", "formes") if k in img.files)
    rows = []
    for key in keys:
        f = reg_dir / f"{key[0]}_{key[1]}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        probs, numero = z["probas"], z["numero"]
        shapes = img[shape_key][numero] if len(img[shape_key]) != len(probs) else img[shape_key]
        resid = 1 - probs[np.arange(len(probs)), shapes.astype(int)]
        rows.append({"key": key, "q": np.percentile(resid, [5, 25, 50, 75, 95]),
                     "frac_in_range": np.mean((resid >= 7e-4) & (resid <= 8e-3)),
                     "frac_above": np.mean(resid >= 7e-4)})
    return rows


def analyse_direction(sweeps, keys, eps, s_min=S_COLOURED):
    rows = []
    floor = []
    for key in keys:
        d = sweeps[key]
        h1, s1 = hue_at_direction(d, eps, 240)
        h2, s2 = hue_at_direction(d, eps, 300)
        ok = (s1 >= s_min) & (s2 >= s_min)
        dh = np.abs(circ_diff(h1, h2))[ok]
        # pas entre directions voisines et monotonie de la trajectoire (teinte dépliée)
        traj = np.vstack([hue_at_direction(d, eps, g)[0] for g in ADMISSIBLE]).T[ok]
        if len(traj):
            unwrapped = np.degrees(np.unwrap(np.radians(traj), axis=1))
            steps = np.abs(np.diff(unwrapped, axis=1))
            sign = np.sign(np.diff(unwrapped, axis=1))
            mono = np.mean(np.all(sign >= 0, axis=1) | np.all(sign <= 0, axis=1))
            step_med = np.median(steps)
            step_max = np.median(steps.max(1))
        else:
            mono = step_med = step_max = np.nan
        res = {"key": key, "n": int(ok.sum()), "dh": dh, "mono": mono, "step_med": step_med,
               "step_max": step_max}
        if key[1] == "control":
            floor.append(dh)
        rows.append(res)
    floor = np.concatenate(floor) if floor else np.array([0.0])
    thr = max(np.percentile(floor, 99), 5.0)
    pvals = []
    for res in rows:
        n = res["n"]
        if n:
            k = int(np.sum(res["dh"] > thr))
            res["above"] = (k, n, *wilson(k, n))
            res["ci"] = boot_ci(res["dh"])
            # H0 : l'écart ne dépasse le plancher des contrôles pas plus souvent que dans 1 % des cas
            res["p"] = stats.binomtest(k, n, 0.01, alternative="greater").pvalue
        else:
            res["above"], res["ci"], res["p"] = (0, 0, np.nan, np.nan), (np.nan, np.nan), np.nan
        pvals.append(res["p"] if np.isfinite(res["p"]) else 1.0)
    adj = holm(pvals)
    for res, a in zip(rows, adj):
        res["p_holm"] = a
    return rows, thr, floor


def write_report(out, dose, spec, natural, direction, thr, floor, direction_alt=None, s_alt=None):
    L = ["# Balayage du bruit sur la forme : statistiques", "",
         "Généré par `experiments/sweep_statistics.py` à partir de `balayage/donnees/`. Unité statistique : "
         "l'exemple (48 par instance), avec une valeur par exemple (médiane sur les 7 directions admissibles, "
         "240° à 300°). Intervalles : bootstrap des exemples (5 000 tirages, 95 %) ou Wilson pour une "
         "proportion. Corrections de Holm sur l'ensemble des tests de chaque tableau. Les instances d'une même "
         "configuration ne sont que 3 : la généralisation à la configuration repose sur ce nombre.", "",
         "Masse déplacée : une perturbation de norme ε dans une direction admissible retire de 0,71 ε (240°, 300°) "
         "à 0,82 ε (270°) à la forme cible. ε = 0,001 correspond donc à 0,07–0,08 % de masse, ε = 0,01 à "
         "0,7–0,8 %.", ""]

    L += ["## A. La masse résiduelle colore-t-elle la traduction ?", "",
          f"S par exemple (médiane sur les directions admissibles). « colorés » : exemples avec S ≥ {fr(S_COLOURED, 1)}. "
          "p : Wilcoxon apparié unilatéral (S(ε) > S(0)), corrigé (Holm, 57 tests). ρ : Spearman entre ε et S "
          "par exemple (10 niveaux, 0 compris) ; « + / − » : exemples à ρ positif / négatif, test des signes corrigé.", "",
          "| Modèle | S(0) | S(ε=0,001) [IC 95 %] | colorés | p | S(ε=0,01) [IC 95 %] | colorés | p | ρ médian (+ / −) | p (signes) | ε* |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in dose:
        e1, e2 = EPSILONS[0], EPSILONS[4]
        f1, f2 = r["frac1e-3"], r["frac1e-2"]
        L.append(f"| {label(r['key'])} | {fr(r['S0'])} | {fr(np.median(r[e1]))} [{fr(r['ci1e-3'][0])}–{fr(r['ci1e-3'][1])}] "
                 f"| {f1[0]}/{f1[1]} | {fp(r['p1e-3_holm'])} | {fr(np.median(r[e2]))} [{fr(r['ci1e-2'][0])}–{fr(r['ci1e-2'][1])}] "
                 f"| {f2[0]}/{f2[1]} | {fp(r['p1e-2_holm'])} | {fr(r['rho_med'])} ({r['rho_sign'][0]} / {r['rho_sign'][1]}) "
                 f"| {fp(r['p_rho_holm'])} | {('%.3g' % r['eps_star']).replace('.', ',') if np.isfinite(r['eps_star']) else '–'} |")
    L += ["", "ε* : plus petite amplitude où la médiane des exemples atteint S ≥ 0,3.", ""]

    L += ["## B. Spécificité : la même amplitude sur les attributs continus", "",
          "ΔC : déplacement de la couleur dans le plan de chrominance (unités RGB), médiane par exemple ; "
          "forme : directions admissibles ; continus : 36 directions aléatoires de même norme ε. Rapport : "
          "médiane des rapports par exemple [IC 95 %]. p : Wilcoxon apparié unilatéral, corrigé (Holm, 38 tests).", "",
          "| Modèle | ΔC forme / continus (ε=0,001) | rapport [IC] | p | ΔC forme / continus (ε=0,01) | rapport [IC] | p |",
          "|---|---|---|---|---|---|---|"]
    for r in spec:
        a, b = r["ratio1e-3"], r["ratio1e-2"]
        L.append(f"| {label(r['key'])} | {fr(r['shape1e-3'], 1)} / {fr(r['cont1e-3'], 2)} | {fr(a[0], 1)} [{fr(a[1], 1)}–{fr(a[2], 1)}] "
                 f"| {fp(r['p1e-3_holm'])} | {fr(r['shape1e-2'], 1)} / {fr(r['cont1e-2'], 2)} | {fr(b[0], 1)} [{fr(b[1], 1)}–{fr(b[2], 1)}] "
                 f"| {fp(r['p1e-2_holm'])} |")
    L.append("")

    if natural:
        L += ["## C. La plage balayée est-elle réaliste ?", "",
              "Masse résiduelle 1 − p(forme vraie) produite par le décodeur d'attributs sur 19 200 vraies images "
              "(image → VAE → GW → attributs ; données de `residual_regression.py`). « dans la plage » : part des "
              "images dont la masse résiduelle est entre 0,07 % et 0,8 % (ε de 0,001 à 0,01).", "",
              "| Modèle | 5 % | 25 % | médiane | 75 % | 95 % | dans la plage | ≥ 0,07 % |", "|---|---|---|---|---|---|---|---|"]
        for r in natural:
            q = r["q"]
            L.append(f"| {label(r['key'])} | " + " | ".join(f"{v:.1e}".replace(".", ",") for v in q)
                     + f" | {fr(100 * r['frac_in_range'], 0)} % | {fr(100 * r['frac_above'], 0)} % |")
        L.append("")

    L += ["## D. La teinte dépend-elle de la répartition du résidu ? (ε = 0,01)", "",
          f"Exemples colorés aux deux extrêmes (S ≥ {fr(S_COLOURED, 1)} à 240° et à 300° : tout le résidu sur l'une ou "
          f"l'autre forme non cible). |Δh| : écart de teinte entre ces deux extrêmes. Plancher de mesure : 99e centile de "
          f"|Δh| chez les 4 contrôles, qui ont une couleur dans leurs attributs ({fr(thr, 1)}° ; maximum "
          f"{fr(float(np.max(floor)), 1)}°). p : test binomial (part des écarts au-dessus du plancher > 1 %), corrigé "
          "(Holm, 19 tests). Monotonie : part des exemples dont la teinte (dépliée) varie dans un seul sens de 240° à 300°.", "",
          "| Modèle | n | |Δh| médian [IC 95 %] | au-dessus du plancher [IC] | p | pas médian | plus grand pas (médiane) | monotonie |",
          "|---|---|---|---|---|---|---|---|"]
    for r in direction:
        if r["n"] == 0:
            L.append(f"| {label(r['key'])} | 0 | – | – | – | – | – | – |")
            continue
        k, n, lo, hi = r["above"]
        L.append(f"| {label(r['key'])} | {n} | {fr(float(np.median(r['dh'])), 1)}° [{fr(r['ci'][0], 1)}–{fr(r['ci'][1], 1)}] "
                 f"| {k}/{n} [{fr(100 * lo, 0)}–{fr(100 * hi, 0)} %] | {fp(r['p_holm'])} | {fr(r['step_med'], 1)}° "
                 f"| {fr(r['step_max'], 1)}° | {fr(100 * r['mono'], 0)} % |")
    L.append("")
    if direction_alt is not None:
        L += [f"Sensibilité au seuil : mêmes mesures avec S ≥ {fr(s_alt, 2)} aux deux extrêmes (même plancher).", "",
              "| Modèle | n | |Δh| médian [IC 95 %] | au-dessus du plancher | p |", "|---|---|---|---|---|"]
        for r in direction_alt:
            if r["n"] == 0:
                L.append(f"| {label(r['key'])} | 0 | – | – | – |")
                continue
            k, n, lo, hi = r["above"]
            L.append(f"| {label(r['key'])} | {n} | {fr(float(np.median(r['dh'])), 1)}° [{fr(r['ci'][0], 1)}–{fr(r['ci'][1], 1)}] "
                     f"| {k}/{n} | {fp(r['p_holm'])} |")
        L.append("")
    out.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = Path(os.environ.get("GWT_RESULTS_DIR", "results")) / "saturation"
    ap.add_argument("--results-dir", type=Path, default=default)
    args = ap.parse_args()
    sweep_dir = args.results_dir / "balayage"
    sweeps = load_sweeps(sweep_dir)
    keys = sorted(sweeps, key=lambda k: (CONFIG_ORDER.index(k[0]) if k[0] in CONFIG_ORDER else 9,
                                         k[1] == "control", k[1]))
    dose = analyse_dose(sweeps, keys)
    spec = analyse_specificity(sweeps, keys)
    natural = analyse_natural(args.results_dir / "regression" / "donnees", keys)
    direction, thr, floor = analyse_direction(sweeps, keys, EPSILONS[4])
    direction_alt, _, _ = analyse_direction(sweeps, keys, EPSILONS[4], s_min=0.15)
    out = sweep_dir / "statistiques.md"
    write_report(out, dose, spec, natural, direction, thr, floor, direction_alt, 0.15)
    print(f"Écrit : {out}")


if __name__ == "__main__":
    main()
