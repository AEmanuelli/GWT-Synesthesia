"""Statistiques de robustesse (étape 4 du protocole « saturation ») : annexe B du compte rendu.

Ne relance aucun modèle. Lit :
  - results/saturation/robustesse/extraction_bruit/donnees/ (saturation_robustness.py) : volets 1 et 2 ;
  - results/saturation/robustesse/tests/tests_*.csv (multiple_testing.py) : volet 3 ;
  - results/saturation/robustesse/regression/sensibilite.csv (regression_sensitivity.py) : volet 4.

Unités statistiques :
  - volets 1 et 2 : l'image traduite (un jeu d'attributs tiré du jeu d'évaluation ; 1 536 par forme,
    mêmes attributs pour tous les modèles et toutes les conditions, donc comparaisons appariées) ;
  - volet 3 : la paire de catégories (régularité) ou la paire d'instances (arbitraire) ;
  - volet 4 : le modèle (15 modèles « vide sémantique », 4 contrôles).
Intervalles : bootstrap des images (2 000 tirages, 95 %) ou Wilson pour une proportion. Corrections
de Holm sur l'ensemble des tests de chaque tableau.

    python experiments/robustness_statistics.py

Sortie : results/saturation/robustesse/annexe_B.md (à copier dans docs/annexes/B_robustesse.md).
"""

import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import paths  # noqa: E402
from saturation_robustness import (AMPLITUDES, NOISE_SEEDS, THESIS_AMP, THESIS_SEED, conditions,  # noqa: E402
                                   load_all, model_order, noise_vector, recipient, select_attributes)
from shape_noise_sweep import CONFIG_LABELS, SHAPES  # noqa: E402
from synesthesia_figure import C_MIN, HUE_TOL, MIN_CONSTANCY, SHAPE_NAMES, opponent  # noqa: E402

N_BOOT = 2000
SMALL_EFFECT = 0.147


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def fr(x, nd=2):
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", "−")


def fp(p):
    return "< 10⁻⁴" if p < 1e-4 else fr(p, 4 if p < 0.01 else 3)


def label(key):
    return f"{CONFIG_LABELS[key[0]]} — {'contrôle' if key[1] == 'control' else 'graine ' + key[1][-1]}"


def sat(rgb):
    x = np.clip(rgb / 255, 0, 1)
    return np.array([colorsys.rgb_to_hls(*p)[2] for p in x])


def hue_deg(rgb):
    a, b = opponent(rgb)
    return np.degrees(np.arctan2(b, a)) % 360, np.hypot(a, b)


def cdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def constancy(rgb):
    """Dominante (direction du vecteur chrominance moyen), constance (part des images colorées à ±30°)."""
    a, b = opponent(rgb)
    dom = np.degrees(np.arctan2(b.mean(), a.mean())) % 360
    h = np.degrees(np.arctan2(b, a)) % 360
    ok = (np.hypot(a, b) >= C_MIN) & (np.abs(cdiff(h, dom)) <= HUE_TOL)
    return dom, ok


def boot_dominant(rgb, idx):
    a, b = opponent(rgb)
    return np.degrees(np.arctan2(b[idx].mean(axis=1), a[idx].mean(axis=1))) % 360


def cliff(x, y):
    u = stats.mannwhitneyu(x, y, method="asymptotic")
    return 2 * u.statistic / (len(x) * len(y)) - 1, u.pvalue


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ----------------------------------------------------------------------------------------------
def volet1(models, inputs, rng):
    conds = [c[0] for c in conditions()]
    ci_noisy = conds.index(f"g{THESIS_SEED}_a{THESIS_AMP:g}")
    keys = sorted(models, key=model_order)
    L = ["## B.1 Autre extraction : pixel le plus intérieur du masque\n",
         "Unité : l'image traduite (n = 4 608 par modèle et condition, appariées entre extractions). "
         "ΔS = S(intérieur) − S(barycentre), médiane [IC 95 %, bootstrap] ; p : Wilcoxon apparié bilatéral, "
         f"Holm sur les {2 * len(keys)} tests. δ(forme) : |δ de Cliff| maximal entre deux formes (S), pour "
         "chaque extraction ; « contrôles » : effet de mesure de référence.\n"]
    rows, ps = [], []
    for k in keys:
        for c, cname in ((0, "sans bruit"), (ci_noisy, "annexe E")):
            sb, si = sat(models[k][c, :, 0]), sat(models[k][c, :, 1])
            d = si - sb
            boot = np.median(d[rng.integers(0, len(d), (N_BOOT, len(d)))], axis=1)
            p = stats.wilcoxon(si, sb).pvalue if np.any(d != 0) else 1.0
            deltas = []
            for s in (sb, si):
                deltas.append(max(abs(cliff(s[inputs["forme"] == i], s[inputs["forme"] == j])[0])
                                  for i, j in ((0, 1), (0, 2), (1, 2))))
            rows.append([label(k), cname, fr(np.median(sb)), fr(np.median(si)),
                         f"{fr(np.median(d), 3)} [{fr(np.percentile(boot, 2.5), 3)} ; {fr(np.percentile(boot, 97.5), 3)}]",
                         None, fr(deltas[0]), fr(deltas[1])])
            ps.append(p)
    for r, p in zip(rows, holm(ps)):
        r[5] = fp(p)
    L.append(md(["Modèle", "condition", "S barycentre", "S intérieur", "ΔS [IC 95 %]", "p (Holm)",
                 "δ(forme) barycentre", "δ(forme) intérieur"], rows) + "\n")

    # associations
    L.append("### Associations forme → couleur (bruit de l'annexe E, graine 42, 0,01)\n")
    L.append("Constance = part des images de la forme dont la couleur est colorée (chroma ≥ 40) et à ±30° de la "
             "teinte dominante [IC de Wilson] ; n = 1 536 images par forme. Δteinte : écart des teintes dominantes "
             "entre extractions [IC 95 %, bootstrap apparié des images]. Association : constance ≥ 60 %, et "
             "« sûre » si la borne basse de l'IC l'est aussi.\n")
    rows, n_b, n_i, n_both = [], 0, 0, 0
    for k in keys:
        if k[1] == "control":
            continue
        for t, shape in enumerate(SHAPES):
            sel = inputs["forme"] == t
            rb, ri = models[k][ci_noisy, sel, 0], models[k][ci_noisy, sel, 1]
            (db, okb), (di, oki) = constancy(rb), constancy(ri)
            cb, ci_ = okb.mean(), oki.mean()
            if max(cb, ci_) < MIN_CONSTANCY:
                continue
            idx = rng.integers(0, sel.sum(), (N_BOOT, sel.sum()))
            dh = cdiff(boot_dominant(ri, idx), boot_dominant(rb, idx))
            lb, ub = wilson(okb.sum(), len(okb)), wilson(oki.sum(), len(oki))
            n_b += lb[0] >= MIN_CONSTANCY
            n_i += ub[0] >= MIN_CONSTANCY
            n_both += lb[0] >= MIN_CONSTANCY and ub[0] >= MIN_CONSTANCY
            rows.append([label(k), SHAPE_NAMES[shape], f"{cb:.0%} [{lb[0]:.0%}–{lb[1]:.0%}]",
                         f"{ci_:.0%} [{ub[0]:.0%}–{ub[1]:.0%}]",
                         f"{fr(cdiff(di, db), 0)}° [{fr(np.percentile(dh, 2.5), 0)} ; {fr(np.percentile(dh, 97.5), 0)}]"])
    L.append(md(["Modèle", "forme", "constance barycentre", "constance intérieur", "Δteinte dominante"], rows) + "\n")
    L.append(f"Associations sûres : {n_b} avec le barycentre, {n_i} avec le pixel intérieur, {n_both} avec les deux.\n")
    return L


def volet2(models, inputs, rng):
    conds = conditions()
    keys = [k for k in sorted(models, key=model_order) if k[1] != "control"]
    L = ["## B.2 Graines et amplitudes du bruit de l'annexe E\n",
         "Unité : l'image traduite (1 536 par forme, mêmes attributs dans toutes les conditions). Le bruit est "
         "un seul vecteur pour toutes les images (comme dans l'évaluation) : la graine ne tire donc pas un bruit "
         "par image mais **une direction du résidu** ; 6 graines = 6 directions.\n"]
    vec = [[str(s), " ; ".join(fr(v, 4) for v in noise_vector(s, 0.01)),
            ", ".join(f"{SHAPE_NAMES[sh]} → {SHAPE_NAMES[recipient(s, 0.01, t)]}" for t, sh in enumerate(SHAPES))]
           for s in NOISE_SEEDS]
    L.append(md(["graine", "vecteur (× 0,01 : losange ; œuf ; triangle)", "forme qui reçoit le plus de résidu"], vec) + "\n")

    # S médiane par configuration, amplitude et graine
    L.append("### Saturation médiane (images), moyenne des 3 instances\n")
    rows = []
    for exp in CONFIG_LABELS:
        ks = [k for k in keys if k[0] == exp]
        cells = []
        for amp in AMPLITUDES:
            vals = [np.mean([np.median(sat(models[k][[c[0] for c in conds].index(f"g{s}_a{amp:g}"), :, 0]))
                             for k in ks]) for s in NOISE_SEEDS]
            cells.append(f"{fr(min(vals))}–{fr(max(vals))} (g42 : {fr(vals[0])})")
        s0 = np.mean([np.median(sat(models[k][0, :, 0])) for k in ks])
        rows.append([CONFIG_LABELS[exp], fr(s0)] + cells)
    L.append(md(["Configuration", "sans bruit", *[f"amplitude {fr(a, 3)} : min–max sur 6 graines" for a in AMPLITUDES]],
                rows) + "\n")

    # associations par condition, et survie de celles du mémoire
    recs, ref = [], {}
    for k in keys:
        for t, shape in enumerate(SHAPES):
            sel = inputs["forme"] == t
            idx = rng.integers(0, sel.sum(), (N_BOOT, sel.sum()))
            ref_c = [c[0] for c in conds].index(f"g{THESIS_SEED}_a{THESIS_AMP:g}")
            rb = models[k][ref_c, sel, 0]
            dref, okref = constancy(rb)
            bref = boot_dominant(rb, idx)
            for ci, (name, seed, amp) in enumerate(conds):
                if seed is None:
                    continue
                rgb = models[k][ci, sel, 0]
                dom, ok = constancy(rgb)
                lo, hi = wilson(ok.sum(), len(ok))
                dh = cdiff(boot_dominant(rgb, idx), bref)
                recs.append(dict(key=k, forme=shape, graine=seed, amplitude=amp, constance=ok.mean(), lo=lo, hi=hi,
                                 dom=dom, dh=cdiff(dom, dref), dh_lo=np.percentile(dh, 2.5),
                                 dh_hi=np.percentile(dh, 97.5), recv=recipient(seed, amp, t)))
    df = pd.DataFrame(recs)
    df["sure"] = df.lo >= MIN_CONSTANCY
    L.append("### Associations sûres (borne basse de Wilson de la constance ≥ 60 %), 15 modèles × 3 formes\n")
    rows = [[fr(a, 3)] + [int(df[(df.amplitude == a) & (df.graine == s)].sure.sum()) for s in NOISE_SEEDS]
            for a in AMPLITUDES]
    L.append(md(["amplitude", *[f"graine {s}" for s in NOISE_SEEDS]], rows) + "\n")

    L.append("### Associations du mémoire (graine 42, 0,01) face aux autres graines et amplitudes\n")
    L.append("Pour chaque association sûre à la graine 42 : dans combien d'autres conditions reste-t-elle sûre "
             "**avec la même couleur** (IC 95 % bootstrap apparié de l'écart de teinte dominante inclus dans ±30°) ; "
             "entre parenthèses, conditions où le résidu principal part vers la même forme qu'à la graine 42.\n")
    base = df[(df.graine == THESIS_SEED) & (df.amplitude == THESIS_AMP) & df.sure]
    rows, n_all_seeds = [], 0
    for _, r in base.iterrows():
        g = df[(df.key == r.key) & (df.forme == r.forme)]
        same = g.sure & (g.dh_lo >= -HUE_TOL) & (g.dh_hi <= HUE_TOL)
        o_seed = g[(g.amplitude == THESIS_AMP) & (g.graine != THESIS_SEED)]
        o_amp = g[(g.graine == THESIS_SEED) & (g.amplitude != THESIS_AMP)]
        same_recv = o_seed.recv == r.recv
        n_seed = int(same[o_seed.index].sum())
        n_all_seeds += n_seed == 5
        rows.append([label(r.key), SHAPE_NAMES[r.forme], f"{r.constance:.0%} [{r.lo:.0%}–{r.hi:.0%}]",
                     f"{n_seed}/5 ({int(same[o_seed.index[same_recv.to_numpy()]].sum())}/{int(same_recv.sum())})",
                     f"{int(same[o_amp.index].sum())}/2",
                     ", ".join(f"g{int(x.graine)} {fr(x.dh, 0)}°" for _, x in o_seed.iterrows() if x.sure)])
    L.append(md(["Modèle", "forme", "constance g42 [IC]", "même couleur, autres graines (même receveuse)",
                 "même couleur, amplitudes 0,003 et 0,03", "écart de teinte des graines sûres"], rows) + "\n")
    L.append(f"{len(base)} associations sûres à la graine 42 ; {n_all_seeds} gardent la même couleur avec les "
             "5 autres graines.\n")

    # dépendance à la graine : test de la receveuse
    L.append("### La couleur dépend-elle de la direction du résidu ?\n")
    L.append("Pour chaque modèle × forme, paires de graines (amplitude 0,01) où la forme est associée (sûre) dans "
             "les deux : |écart de teinte dominante| selon que le résidu principal part vers la même forme ou "
             "non. Test : Mann-Whitney unilatéral (même receveuse < autre receveuse) ; unité : la paire de graines "
             "(non indépendantes : indication seulement).\n")
    same, other = [], []
    x = df[(df.amplitude == THESIS_AMP) & df.sure]
    for (k, f), g in x.groupby(["key", "forme"]):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = abs(cdiff(g.dom[i], g.dom[j]))
                (same if g.recv[i] == g.recv[j] else other).append(d)
    if same and other:
        p = stats.mannwhitneyu(same, other, alternative="less").pvalue
        L.append(f"Même receveuse : médiane {fr(np.median(same), 0)}° (n = {len(same)} paires) ; autre receveuse : "
                 f"{fr(np.median(other), 0)}° (n = {len(other)}) ; p = {fp(p)}.\n")
    else:
        L.append(f"Paires insuffisantes (même receveuse : {len(same)}, autre : {len(other)}).\n")
    return L, df


def volet3(tests_dir):
    L = ["## B.3 Tests : test du mémoire, KS, critère d'effet, corrections\n",
         "Unité : la paire de catégories (régularité, au sein d'une instance) ou la paire (instances, catégorie) "
         "(arbitraire). Famille de correction : toutes les paires d'un même tableau, chemin et canal. Tableaux "
         "complets : `results/saturation/robustesse/tests/tests_*.md`. Ici : chemin de traduction (évaluation "
         "exacte), sans bruit et avec le bruit de l'annexe E ; % de paires qui diffèrent.\n"]
    d = pd.read_csv(tests_dir / "tests_traduction_seule.csv")
    shown = ["memoire", "memoire_holm", "memoire_bh", "brut", "brut_holm", "effet", "effet_holm"]
    head = ["mémoire", "mémoire + Holm", "mémoire + BH", "KS", "KS + Holm", "effet", "effet + Holm"]
    for tab, col in (("régularité", "paires_"), ("arbitraire", "paires_")):
        for path, pl in (("translated", "Traduction"), ("translated_noisy", "Traduction bruitée (annexe E)")):
            for ch in ("H", "S"):
                rows = []
                for attr, al in (("shape", "Forme"), ("rotation", "Orientation")):
                    x = d[(d.tableau == tab) & (d.chemin == path) & (d.canal == ch) & (d.attribut == attr)]
                    for _, r in x.iterrows():
                        rows.append([al, r.groupe, int(r.n_paires)] + [f"{100 * r[col + c]:.0f}" for c in shown])
                L.append(f"**{tab.capitalize()}, {pl}, canal {ch}**\n")
                L.append(md(["attribut", "groupe", "n paires", *head], rows) + "\n")
    return L


def volet4(reg_csv):
    L = ["## B.4 Régression : sensibilité au sous-ensemble d'images\n"]
    if not reg_csv.exists():
        return L + ["(non calculé)\n"]
    df = pd.read_csv(reg_csv)
    ref_name = [v for v in df.variante.unique() if "mémoire" in v][0]
    L.append("Unité : le modèle. R² de la chrominance (validation croisée à 5 plis, meilleure de ridge et kNN), "
             f"référence « {ref_name} ». Écart : R² variante − R² référence, médiane [min ; max] sur les 19 modèles. "
             "ρ : Spearman du classement des 15 modèles « vide sémantique », p corrigé (Holm, sur tous les ρ du tableau).\n")
    rows, ps = [], []
    for g, gl in (("residu", "résidu"), ("continus_contexte", "continus + géométrie"),
                  ("les_deux_contexte", "tout"), ("latent_gw", "latent GW")):
        ref = df[(df.variante == ref_name) & (df.groupe == g)].set_index(["config", "instance"]).r2
        for v in df.variante.unique():
            if v == ref_name:
                continue
            x = df[(df.variante == v) & (df.groupe == g)].set_index(["config", "instance"]).r2
            common = ref.index.intersection(x.index)
            diff = x[common] - ref[common]
            sem = [k for k in common if k[1] != "control"]
            rho = stats.spearmanr(ref[sem], x[sem])
            ctrl = [k for k in common if k[1] == "control"]
            rows.append([gl, v, len(common), f"{fr(diff.median())} [{fr(diff.min())} ; {fr(diff.max())}]",
                         fr(rho.statistic), None, fr(x[ctrl].max()) if ctrl else "–"])
            ps.append(rho.pvalue)
    for r, p in zip(rows, holm(ps)):
        r[5] = fp(p)
    L.append(md(["variables", "variante", "n modèles", "écart de R² médian [min ; max]", "ρ", "p (Holm)",
                 "R² max des contrôles"], rows) + "\n")
    return L


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = paths.RESULTS_DIR / "saturation" / "robustesse"
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--per-shape", type=int, default=1536)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    inputs = select_attributes(args.per_shape)
    models = load_all(args.root / "extraction_bruit" / "donnees")
    L = ["# Robustesse des résultats sur la saturation : statistiques\n",
         f"Généré par `experiments/robustness_statistics.py`. {len(models)} modèles. Intervalles : bootstrap "
         f"({N_BOOT} tirages, 95 %) ou Wilson ; corrections de Holm par tableau. Hors de portée sur ce GPU : "
         "autres instances ou réentraînements (3 instances par configuration seulement).\n"]
    L += volet1(models, inputs, rng)
    l2, df2 = volet2(models, inputs, rng)
    L += l2
    df2.drop(columns="key").assign(config=[k[0] for k in df2.key], instance=[k[1] for k in df2.key]).to_csv(
        args.root / "bruit_conditions.csv", index=False)
    L += volet3(args.root / "tests")
    L += volet4(args.root / "regression" / "sensibilite.csv")
    (args.root / "annexe_B.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
