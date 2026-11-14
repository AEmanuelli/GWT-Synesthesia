
"""Critère de Régularité pour UN modèle (mémoire §2.3.2).

Compare, avec des tests de Kolmogorov-Smirnov, les distributions de couleurs
reconstruites entre catégories d'un même attribut, pour les chemins
traduction / demi-cycle / cycle complet.

    python SSD_eval/SSD_eval_regularity.py --checkpoint checkpoints/runs/high_cycles_seed0/checkpoints/last.ckpt
    python SSD_eval/SSD_eval_regularity.py --checkpoint ... --with-colors   # Modèle Complet
"""
import argparse
import sys
from pathlib import Path
import torch
from shimmer.modules.global_workspace import GlobalWorkspace2Domains

from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401  (checkpoints GPU -> CPU si pas de GPU)
from SSD_utils import binning_config_6144, default_binning_config
from category_probs import use_category_probabilities

from shimmer_ssd.modules.domains import load_pretrained_domains

# Additional imports for shape generation and evaluation
import os

# from SSD_bin_consistency import test_color_consistency_across_bins, test_all_attributes_color_consistency
from SSD_H_evaluation_functions import HueShapeAnalyzer


import multiprocessing
import time


def load_global_workspace(gw_checkpoint: Path, config: dict) -> GlobalWorkspace2Domains:
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config["domains"],
        config["global_workspace"]["latent_dim"],
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
    )
    # Catégories de forme en probabilités, comme à l'entraînement (voir category_probs.py)
    use_category_probabilities(domain_modules, gw_decoders)
    global_workspace = GlobalWorkspace2Domains.load_from_checkpoint(
        gw_checkpoint,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )
    return global_workspace


# ====================
# MAIN : ARGUMENTS & EXECUTION
# ====================


def eval_regularity(
    dataset_csv=str(paths.EVALSET_DIR / "attributes.csv"), 
    full_attr=True, 
    run_id=None,  
    checkpoint_base_path=None, 
    n_encoders=2, 
    n_decoders=1, 
    encoders_size=256, 
    decoders_size=256, 
    name="àtrier",
    latent_dim=12,
    debug=False,
    num_bins=50,
    reverb_n=1,
    binning_config=None,
    rgb=False,
    tilting=1,
    parrallel=False,
    barycentre = False,
    conditioning = False,
    my_hparams={"temperature": 1, "alpha": 1}, # hyperparameters for the model
):
    """
    Evaluate model regularity on shape representations.

    Args:
        dataset_csv: Path to CSV with dataset attributes
        full_attr: Boolean indicating if the attribute domain uses color info
        run_id: Optional identifier for the run (e.g., seed value)
        checkpoint_base_path: Path to the model checkpoint
        n_encoders: Number of encoder layers
        n_decoders: Number of decoder layers
        encoders_size: Hidden dimension size for encoders
        decoders_size: Hidden dimension size for decoders
        name: Name for the output directory
        latent_dim: Dimension of the latent space
        debug: Whether to run in debug mode
        num_bins: Number of bins for distribution analysis
    """
   
    
    # Set seed if run_id is provided
    seed = int(run_id[-1]) if run_id else 0

    # Determine configuration based on full_attr
    if full_attr:
        attr_domain_type = DomainModuleVariant.attr_legacy
        config_suffix = "avec_couleurs"
    else:
        attr_domain_type = DomainModuleVariant.attr_legacy_no_color
        config_suffix = "sans_couleurs"

    print("_______________________________________________________________________________")
    print(f"Evaluation du modèle {version} {'avec' if full_attr else 'sans'} couleurs dans les attributs")
    if run_id:
        print(f"Run ID: {run_id}")
    print("_______________________________________________________________________________")

    # Define output directory
    output_base = str(paths.RESULTS_DIR / "regularity" / name)
    run_specific_part = f"results_{config_suffix}"
    if run_id:
        run_specific_part += f"_{run_id}"

    output_dir = os.path.join(output_base, run_specific_part)
    output_dir = Path(output_dir).resolve()

    if debug:
        import time
        debug_timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results_debug/{version}_{config_suffix}{'_' + run_id if run_id else ''}_{debug_timestamp}"

    # Check if output exists
    if os.path.exists(output_dir) and not debug:
        print(f"WARNING: Le répertoire {output_dir} existe déjà. Les résultats pourraient être écrasés ou incomplets.")

    os.makedirs(output_dir, exist_ok=True)

    # Setup config and load model
    config = {
        "domains": [
            LoadedDomainConfig(
                domain_type=DomainModuleVariant.v_latents,
                checkpoint_path=paths.DOMAIN_V_CKPT,
            ),
            LoadedDomainConfig(
                domain_type=attr_domain_type,
                checkpoint_path=paths.DOMAIN_ATTR_CKPT,
                args=my_hparams,
            ),
        ],
        "global_workspace": {
            "latent_dim": latent_dim,
            "encoders": {"hidden_dim": encoders_size, "n_layers": n_encoders},
            "decoders": {"hidden_dim": decoders_size, "n_layers": n_decoders},
        },
    }
    
    gw_checkpoint = Path(checkpoint_base_path)
    if not gw_checkpoint.exists():
        print(f"ERROR: Checkpoint file not found at {gw_checkpoint}")
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEBUG_MODE:
            # Dont load the model if in debug mode
            print("DEBUG_MODE is ON. Skipping model loading.")
            global_workspace = None
        else:
            global_workspace = load_global_workspace(gw_checkpoint, config)
            global_workspace.to(device) 
        print(f"Model loaded successfully from {gw_checkpoint} onto {device}.")
    except Exception as e:
        print(f"ERROR: Failed to load model from checkpoint {gw_checkpoint}. Error: {e}")
        return
    
    analyzer = HueShapeAnalyzer(
        global_workspace=global_workspace,
        device=device,
        color=full_attr,
        output_dir=output_dir,
        debug=debug,
        seed=seed, 
        num_bins=num_bins,
        reverb_n = reverb_n,
        rgb=rgb,
        barycentre=barycentre,
    )

    # Run analysis
    print(f"Starting analysis. Results will be saved in: {output_dir}")
    analysis_results = analyzer.analyze_dataset(
        csv_path=dataset_csv,
        display_examples=True, 
        binning_config=binning_config,
        im_dir=dataset_csv.replace("/attributes.csv", ''),
        tilting=tilting,
        parrallel=parrallel,
        conditioning=conditioning,
    )
    print("Analysis completed.")


DEBUG_MODE = False
version = "regularity"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .ckpt du Global Workspace")
    parser.add_argument("--with-colors", action="store_true", help="Le modèle est un Modèle Complet (contrôle)")
    parser.add_argument("--name", default=None, help="Sous-dossier de sortie dans results/regularity/")
    parser.add_argument("--run-id", default="seed0")
    parser.add_argument("--dataset-csv", default=str(paths.EVALSET_DIR / "attributes.csv"))
    parser.add_argument("--num-bins", type=int, default=15, help="Bins des histogrammes (15 dans le mémoire)")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--shape-noise", type=float, default=0.0,
                        help="Bruit sur les attributs de forme en traduction (0.01 = Annexe E)")
    parser.add_argument("--shape-noise-seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    import SSD_utils
    SSD_utils.SHAPE_NOISE_SCALE = args.shape_noise
    SSD_utils.SHAPE_NOISE_SEED = args.shape_noise_seed
    if args.shape_noise > 0 and args.name is None:
        args.name = f"{Path(args.checkpoint).parents[1].name}_shape_noise_{args.shape_noise:g}"
    eval_regularity(
        full_attr=args.with_colors,
        run_id=args.run_id,
        checkpoint_base_path=args.checkpoint,
        encoders_size=256,
        n_encoders=2,
        n_decoders=2,
        name=args.name or Path(args.checkpoint).parents[1].name,
        num_bins=args.num_bins,
        dataset_csv=args.dataset_csv,
        reverb_n=1,
        rgb=True,
        binning_config=binning_config_6144,
        tilting=1,
        parrallel=args.parallel,
        conditioning=True,
        barycentre=True,
    )
