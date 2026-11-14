"""Recalcule le tableau B.1 du mémoire à partir des checkpoints (losses de validation).

    python experiments/compute_table_B1.py --use-original-names     # modèles publiés
    python experiments/compute_table_B1.py                          # modèles réentraînés

Chaque modèle du manifeste (experiments/thesis_runs.yaml) est évalué sur le jeu de
validation du Simple Shapes Dataset avec la même fonction de loss qu'à l'entraînement.
Nécessite le dataset d'entraînement et ses latents visuels (scripts/download_data.sh).

Colonnes : L_total, L_traduction, L_contrastive, L_cycle-complet, L_demi-cycle.
L_total = moyenne des quatre termes pondérés, (αL_tr + βL_cont + γL_dcy + δL_cy) / 4.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401
from lightning.pytorch import Trainer  # noqa: E402
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config  # noqa: E402
from SSD_eval_regularity import load_global_workspace  # noqa: E402
from SSD_Train import setup_data_module  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from check_table_B1 import TABLE_B1  # noqa: E402

KEYS = ["val/loss", "val/translations", "val/contrastives", "val/cycles", "val/demi_cycles"]
# Instances effectivement moyennées dans le mémoire quand ce ne sont pas toutes (README §5)
THESIS_SUBSETS = {("base_params", "runs"): ["seed1", "seed2"]}


def validation_losses(exp, run, full_attr, use_original_names):
    config = load_config(paths.CONFIG_DIR, use_cli=False, load_files=[exp["config"]])
    config.seed = run["seed"]
    config.dataset.path = paths.DATASET_DIR
    run_dir = run["original_run_name"] if use_original_names else run["run_dir"]
    ckpt_name = run.get("original_ckpt", run["ckpt"]) if use_original_names else run["ckpt"]
    ckpt = paths.gw_checkpoint(run_dir, ckpt_name)
    if not ckpt.exists():
        print(f"  (checkpoint absent : {ckpt})")
        return None
    attr_variant = DomainModuleVariant.attr_legacy if full_attr else DomainModuleVariant.attr_legacy_no_color
    gw_config = config.global_workspace
    global_workspace = load_global_workspace(ckpt, {
        "domains": [
            LoadedDomainConfig(domain_type=DomainModuleVariant.v_latents, checkpoint_path=paths.DOMAIN_V_CKPT),
            LoadedDomainConfig(domain_type=attr_variant, checkpoint_path=paths.DOMAIN_ATTR_CKPT,
                               args={"temperature": 1, "alpha": exp["attr_alpha"]}),
        ],
        "global_workspace": {
            "latent_dim": gw_config.latent_dim,
            "encoders": {"hidden_dim": gw_config.encoders.hidden_dim, "n_layers": gw_config.encoders.n_layers},
            "decoders": {"hidden_dim": gw_config.decoders.hidden_dim, "n_layers": gw_config.decoders.n_layers},
        },
    })
    data_module = setup_data_module(config, exclude_colors=not full_attr)
    trainer = Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    metrics = trainer.validate(global_workspace, data_module, verbose=False)[0]
    return [float(metrics[k]) for k in KEYS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-original-names", action="store_true",
                        help="Checkpoints sous leurs noms de dossiers d'origine (poids publiés)")
    parser.add_argument("--experiment", action="append", help="Configuration(s) à traiter (défaut : toutes)")
    args = parser.parse_args()
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    manifest = yaml.safe_load(open(paths.THESIS_RUNS_FILE))["experiments"]
    lines = [f"{'':44s}{'total':>8s}{'trad.':>8s}{'contr.':>8s}{'cycle':>8s}{'demi':>8s}"]
    for exp_name in args.experiment or manifest:
        exp = manifest[exp_name]
        for group in ("control", "runs"):
            runs = [exp["control"]] if group == "control" else exp["runs"]
            values = {}
            for run in runs:
                losses = validation_losses(exp, run, group == "control", args.use_original_names)
                if losses is not None:
                    values[run["run_id"]] = losses
                    print(f"{exp_name}/{run['run_id']}: " + " ".join(f"{v:.4f}" for v in losses), flush=True)
            if not values:
                continue
            subsets = [("toutes les instances", list(values))] if group == "runs" else [("contrôle", list(values))]
            if (exp_name, group) in THESIS_SUBSETS and all(r in values for r in THESIS_SUBSETS[(exp_name, group)]):
                subsets.append(("instances du mémoire", THESIS_SUBSETS[(exp_name, group)]))
            reference = TABLE_B1[(exp_name, group)]
            for label, run_ids in subsets:
                mean = [sum(values[r][i] for r in run_ids) / len(run_ids) for i in range(len(KEYS))]
                n_exact = sum(round(m, 3) == r for m, r in zip(mean, reference))
                lines.append(f"{exp_name + ' / ' + label:44s}" + "".join(f"{m:8.3f}" for m in mean)
                             + f"   ({n_exact}/5 identiques)")
            lines.append(f"{'  mémoire (tableau B.1)':44s}" + "".join(f"{r:8.3f}" for r in reference))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
