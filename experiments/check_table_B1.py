"""Compare les losses de validation finales (export CSV de W&B) au tableau B.1 du mémoire.

    python experiments/check_table_B1.py wandb_export.csv

L'export doit contenir les colonnes Name, val/loss, val/translations, val/contrastives,
val/cycles, val/demi_cycles (tableau "Runs" de W&B -> Export CSV).
Pour chaque configuration, moyenne des runs du manifeste (vide sémantique / contrôle)
et comparaison aux valeurs du mémoire, arrondies au millième.
"""
import csv
import sys
from pathlib import Path

import yaml

# Tableau B.1 du mémoire : (L_total, L_traduction, L_contrastive, L_cycle-complet, L_demi-cycle)
TABLE_B1 = {
    ("base_params", "control"): (0.017, 0.005, 0.545, 0.005, 0.003),
    ("base_params", "runs"): (0.055, 0.094, 0.791, 0.039, 0.007),
    ("high_translation", "control"): (0.026, 0.003, 0.644, 0.004, 0.006),
    ("high_translation", "runs"): (0.268, 0.089, 0.982, 0.064, 0.012),
    ("high_contrastive", "control"): (0.128, 0.007, 0.490, 0.011, 0.005),
    ("high_contrastive", "runs"): (0.192, 0.097, 0.557, 0.081, 0.050),
    ("high_cycles", "control"): (0.057, 0.087, 0.805, 0.005, 0.008),
    ("high_cycles", "runs"): (0.110, 0.157, 1.286, 0.014, 0.011),
    ("high_alpha", "control"): (0.017, 0.006, 0.555, 0.005, 0.003),
    ("high_alpha", "runs"): (0.055, 0.095, 0.792, 0.038, 0.006),
}
KEYS = ["val/loss", "val/translations", "val/contrastives", "val/cycles", "val/demi_cycles"]


def main(csv_path):
    rows = {r["Name"].strip(): r for r in csv.DictReader(open(csv_path))}
    manifest = yaml.safe_load(open(Path(__file__).with_name("thesis_runs.yaml")))["experiments"]
    print(f"{'':38s}{'total':>8s}{'trad.':>8s}{'contr.':>8s}{'cycle':>8s}{'demi':>8s}")
    for exp_name, exp in manifest.items():
        for group in ("runs", "control"):
            runs = exp[group] if group == "runs" else [exp["control"]]
            vals = []
            for run in runs:
                name = run.get("wandb_name", run["original_run_name"])
                if name in rows:
                    vals.append([float(rows[name][k]) for k in KEYS])
                else:
                    print(f"  (absent de l'export : {name})")
            if not vals:
                continue
            mean = [sum(v[i] for v in vals) / len(vals) for i in range(len(KEYS))]
            ref = TABLE_B1[(exp_name, group)]
            n_exact = sum(round(m, 3) == r for m, r in zip(mean, ref))
            label = f"{exp_name} / {'contrôle' if group == 'control' else f'{len(vals)} instances'}"
            print(f"{label:38s}" + "".join(f"{m:8.3f}" for m in mean) + f"   ({n_exact}/5 identiques)")
            print(f"{'  mémoire':38s}" + "".join(f"{r:8.3f}" for r in ref))


if __name__ == "__main__":
    main(sys.argv[1])
