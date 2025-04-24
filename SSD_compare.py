import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import default_collate

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from shimmer.modules.global_workspace import GlobalWorkspace2Domains
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
from shimmer_ssd.logging import LogGWImagesCallback
from shimmer_ssd.modules.domains import load_pretrained_domains

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
from SSD_H_evaluation_functions import HueShapeAnalyzer  # Evaluation functions


def custom_collate(batch):
    """Custom collate function that removes the last 3 values from a specific tensor in the batch."""
    result = default_collate(batch)
    if (
        isinstance(result, dict)
        and "attr" in result
        and isinstance(result["attr"], list)
        and len(result["attr"]) >= 2
        and isinstance(result["attr"][1], torch.Tensor)
        and result["attr"][1].size(-1) >= 4
    ):
        result["attr"][1] = result["attr"][1][..., :-3]
    return result


def get_scheduler(optimizer: Optimizer, config) -> OneCycleLR:
    """Returns a OneCycleLR scheduler based on the configuration."""
    return OneCycleLR(optimizer, config.training.optim.max_lr, config.training.max_steps)


def override_config_with_sweep(config: Any, sweep_overrides: Dict[str, Any], hparams: dict) -> None:
    """
    Overwrite configuration parameters with values from the wandb sweep.
    This function updates both the training config and hparams.
    """
    # Update top-level configurations
    if "global_workspace_latent_dim" in sweep_overrides:
        config.global_workspace.latent_dim = sweep_overrides["global_workspace_latent_dim"]
    if "global_workspace.encoders.n_layers" in sweep_overrides:
        config.global_workspace.encoders.n_layers = sweep_overrides["global_workspace.encoders.n_layers"]
    if "global_workspace.decoders.n_layers" in sweep_overrides:
        config.global_workspace.decoders.n_layers = sweep_overrides["global_workspace.decoders.n_layers"]
    if "global_workspace.encoders.hidden_dim" in sweep_overrides:
        config.global_workspace.encoders.hidden_dim = sweep_overrides["global_workspace.encoders.hidden_dim"]
    if "global_workspace.decoders.hidden_dim" in sweep_overrides:
        config.global_workspace.decoders.hidden_dim = sweep_overrides["global_workspace.decoders.hidden_dim"]
    if "training.optim.lr" in sweep_overrides:
        config.training.optim.lr = sweep_overrides["training.optim.lr"]
    if "training.optim.max_lr" in sweep_overrides:
        config.training.optim.max_lr = sweep_overrides["training.optim.max_lr"]
    if "training.optim.weight_decay" in sweep_overrides:
        config.training.optim.weight_decay = sweep_overrides["training.optim.weight_decay"]
    if "training.batch_size" in sweep_overrides:
        config.training.batch_size = sweep_overrides["training.batch_size"]
    if "training.max_steps" in sweep_overrides:
        config.training.max_steps = sweep_overrides["training.max_steps"]
    else : 
        # Set default max_steps based on batch size
        config.training.max_steps = 500000 // config.training.batch_size
    


    # Update hyperparameters (my_hparams)
    if "my_hparams.alpha" in sweep_overrides:
        hparams["alpha"] = sweep_overrides["my_hparams.alpha"]
    if "my_hparams.temperature" in sweep_overrides:
        hparams["temperature"] = sweep_overrides["my_hparams.temperature"]

    # Update loss coefficients in the global workspace config (assuming it's a dict)
    loss_coeffs = config.global_workspace.loss_coefficients
    if isinstance(loss_coeffs, dict):
        if "global_workspace.loss_coefficients.cycles" in sweep_overrides:
            loss_coeffs["cycles"] = sweep_overrides["global_workspace.loss_coefficients.cycles"]
        if "global_workspace.loss_coefficients.contrastives" in sweep_overrides:
            loss_coeffs["contrastives"] = sweep_overrides["global_workspace.loss_coefficients.contrastives"]
        if "global_workspace.loss_coefficients.demi_cycles" in sweep_overrides:
            loss_coeffs["demi_cycles"] = sweep_overrides["global_workspace.loss_coefficients.demi_cycles"]
        if "global_workspace.loss_coefficients.translations" in sweep_overrides:
            loss_coeffs["translations"] = sweep_overrides["global_workspace.loss_coefficients.translations"]
    else:
        # If loss_coefficients is not a dict, fallback to attribute assignment
        if "global_workspace.loss_coefficients.cycles" in sweep_overrides:
            setattr(config.global_workspace.loss_coefficients, "cycles", sweep_overrides["global_workspace.loss_coefficients.cycles"])
        if "global_workspace.loss_coefficients.contrastives" in sweep_overrides:
            setattr(config.global_workspace.loss_coefficients, "contrastives", sweep_overrides["global_workspace.loss_coefficients.contrastives"])
        if "global_workspace.loss_coefficients.demi_cycles" in sweep_overrides:
            setattr(config.global_workspace.loss_coefficients, "demi_cycles", sweep_overrides["global_workspace.loss_coefficients.demi_cycles"])
        if "global_workspace.loss_coefficients.translations" in sweep_overrides:
            setattr(config.global_workspace.loss_coefficients, "translations", sweep_overrides["global_workspace.loss_coefficients.translations"])

    
    # Custom hyperparameters
    if "my_hparams.temperature" in sweep_overrides:
        hparams["temperature"] = sweep_overrides["my_hparams.temperature"]
    if "my_hparams.alpha" in sweep_overrides:
        hparams["alpha"] = sweep_overrides["my_hparams.alpha"]

def run_experiment(exclude_colors: bool, run_name: str, sweep_overrides: Dict[str, Any] = None) -> Tuple[GlobalWorkspace2Domains, float]:
    """
    Run training experiment with or without color attributes.
    Returns the trained global workspace and the validation loss.
    """
    # Load and update configuration
    config = load_config("./config", use_cli=False, load_files=["train_gw.yaml"])
    config.domain_data_args["v_latents"]["presaved_path"] = "domain_v.npy"
    config.global_workspace.latent_dim = 12  # default; may be overridden via sweep
    config.domain_proportions = {
        frozenset(["v"]): 1.0,
        frozenset(["attr"]): 1.0,
        frozenset(["v", "attr"]): 1.0,
    }
    hparams = {"temperature": 0.5, "alpha": 2}  # default; may be overridden via sweep
    checkpoint_path = Path("./checkpoints")

    if sweep_overrides:
        # print("Sweep overrides:", sweep_overrides)
        override_config_with_sweep(config, sweep_overrides, hparams)

    print("training max steps", config.training.max_steps)

    # Choose domain variant based on color inclusion
    attr_variant = DomainModuleVariant.attr_legacy_no_color if exclude_colors else DomainModuleVariant.attr_legacy
    config.domains = [
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.v_latents,
            checkpoint_path=checkpoint_path / "domain_v.ckpt",
        ),
        LoadedDomainConfig(
            domain_type=attr_variant,
            checkpoint_path=checkpoint_path / "domain_attr.ckpt",
            args=hparams,
        ),
    ]

    # Create data module (apply custom collate if excluding colors)
    domain_classes = get_default_domains(["v_latents", "attr"])
    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.domain_data_args,
        **({"collate_fn": custom_collate} if exclude_colors else {})
    )

    # Load pretrained domains
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )

    # Initialize the global workspace with its scheduler
    global_workspace = GlobalWorkspace2Domains(
        domain_modules,
        gw_encoders,
        gw_decoders,
        config.global_workspace.latent_dim,
        config.global_workspace.loss_coefficients,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler=lambda opt: get_scheduler(opt, config),
    )

    # Set up Wandb logger with a specific run name
    logger = WandbLogger(name=run_name, project="shimmer-ssd-compare")
    logger.log_hyperparams(hparams)

    # Prepare sample images for logging callbacks
    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    # Convert sample domains to dictionary format (only take the first set)
    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
        break

    # Create checkpoint directory
    gw_root = config.default_root_dir / "gw"
    gw_root.mkdir(exist_ok=True)

    # Define callbacks for image logging and checkpointing
    callbacks: list[Callback] = [
        LogGWImagesCallback(
            val_samples,
            log_key="images/val",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
            filter=config.logging.filter_images,
            exclude_colors=exclude_colors,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            filter=config.logging.filter_images,
            exclude_colors=exclude_colors,
        ),
        ModelCheckpoint(
            dirpath=config.default_root_dir / "gw" / f"version_color{exclude_colors}_{logger.version}",
            filename="{epoch}",
            monitor="val/loss",  # Ensure your LightningModule logs this key
            mode="min",
            save_last="link",
            save_top_k=1,
        ),
    ]

    # Create trainer and run training/validation
    trainer = Trainer(
        logger=logger,
        max_steps=config.training.max_steps,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
    )

    trainer.fit(global_workspace, data_module)
    results = trainer.validate(global_workspace, data_module, ckpt_path="last")
    val_loss = results[0].get("val/loss", float("inf"))
    return global_workspace, val_loss


def further_processing(global_workspace: GlobalWorkspace2Domains, full_attr: bool, eval_csv: str) -> float:
    """
    Evaluate the SSD dataset using HueShapeAnalyzer.
    Returns the mean KL divergence from the hue distribution consistency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_workspace.to(device)

    analyzer = HueShapeAnalyzer(
        global_workspace=global_workspace,
        device=device,
        color=full_attr,
        output_dir=f"./results_{'avec' if full_attr else 'sans'}_couleurs"
    )

    print("Comparing hue distributions across shapes...")
    results_consistency = analyzer.compare_hue_distributions_across_shapes(csv_path=eval_csv)

    # Collect all numerical KL divergence values
    all_kl_values = []
    for shape_pair, channel_results in results_consistency["kl_divergence_results"].items():
        for channel_name, kl_metric_results in channel_results.items():
            for kl_metric_name, kl_value in kl_metric_results.items():
                if isinstance(kl_value, (int, float)):
                    all_kl_values.append(kl_value)
                else:
                    print(f"Warning: Non-numerical KL value for {shape_pair}, channel '{channel_name}', metric '{kl_metric_name}': {kl_value}. Skipping.")

    if all_kl_values:
        return np.mean(all_kl_values)
    else:
        print("Warning: No numerical KL divergence values found. Returning NaN.")
        return float('nan')


def main():
    parser = argparse.ArgumentParser(
        description="Train model with/without color and perform evaluation with KL divergence sweep."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="evaluation_set/attributes.csv", help="CSV file for evaluation")
    
    # Add arguments for all configurable parameters
    parser.add_argument("--global_workspace.decoders.hidden_dim", type=int, default=256)
    parser.add_argument("--global_workspace.decoders.n_layers", type=int, default=2)
    parser.add_argument("--global_workspace.encoders.hidden_dim", type=int, default=256)
    parser.add_argument("--global_workspace.encoders.n_layers", type=int, default=0)
    parser.add_argument("--global_workspace.latent_dim", type=int, default=12)
    
    # Loss coefficients
    parser.add_argument("--global_workspace.loss_coefficients.contrastives", type=float, default=0.5)
    parser.add_argument("--global_workspace.loss_coefficients.cycles", type=float, default=1.0)
    parser.add_argument("--global_workspace.loss_coefficients.demi_cycles", type=float, default=2.0)
    parser.add_argument("--global_workspace.loss_coefficients.translations", type=float, default=1.5)
    
    # My hparams
    parser.add_argument("--my_hparams.alpha", type=float, default=2.0)
    parser.add_argument("--my_hparams.temperature", type=float, default=0.5)
    

    # Training parameters
    parser.add_argument("--training.max_steps", type=int, default=2599)
    parser.add_argument("--training.batch_size", type=int, default=64)
    parser.add_argument("--training.optim.lr", type=float, default=1e-4)
    parser.add_argument("--training.optim.max_lr", type=float, default=1e-3)
    parser.add_argument("--training.optim.weight_decay", type=float, default=1e-4)

    args = parser.parse_args()
    
    # Convert parsed args to a dictionary for sweep overrides
    sweep_config = {
        key.replace('.', '_'): value 
        for key, value in vars(args).items() 
        if key not in ['seed', 'eval_csv']
    }
    # print("DEBUG : Sweep config:", sweep_config)
    # Initialize wandb sweep
    import wandb
    wandb.init(project="SSD-compare", config=sweep_config)
    
    # Use wandb config to override any defaults
    sweep_config = wandb.config

    seed_everything(args.seed)

    # Run experiments with both configurations (with and without color)
    print("Training with color included...")
    gw_with_color, loss_with_color = run_experiment(
        exclude_colors=False, run_name="gw_with_color", sweep_overrides=sweep_config
    )
    print("Training with color excluded...")
    gw_without_color, loss_without_color = run_experiment(
        exclude_colors=True, run_name="gw_without_color", sweep_overrides=sweep_config
    )

    # Report validation loss differences
    loss_diff = loss_without_color - loss_with_color
    print(f"Validation Loss (with color): {loss_with_color}")
    print(f"Validation Loss (without color): {loss_without_color}")
    print(f"Difference in Validation Loss: {loss_diff}")

    # Evaluate using HueShapeAnalyzer
    print("Evaluating model with color attributes...")
    kl_with_color = further_processing(gw_with_color, full_attr=True, eval_csv=args.eval_csv)
    print("Evaluating model without color attributes...")
    kl_without_color = further_processing(gw_without_color, full_attr=False, eval_csv=args.eval_csv)
    kl_diff = kl_without_color - kl_with_color
    kl_div = kl_without_color/kl_with_color

    print(f"Mean KL Divergence (with color): {kl_with_color}")
    print(f"Mean KL Divergence (without color): {kl_without_color}")
    print(f"Difference in Mean KL Divergence: {kl_diff}")
    print(f"Division between ")
    # Log metrics to wandb
    wandb.log({
        "val_loss_difference": loss_diff,
        "kl_diff": kl_diff,
        "kl_with_color": kl_with_color,
        "kl_without_color": kl_without_color,
        "kl_division_pondéréé":kl_div/loss_without_color,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
