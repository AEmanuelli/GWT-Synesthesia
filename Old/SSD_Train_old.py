# from collections.abc import Mapping, Sequence
# from pathlib import Path
# from typing import Any, cast

# import matplotlib
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from lightning.pytorch import Callback, Trainer, seed_everything
# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.loggers import TensorBoardLogger
# from shimmer import DomainModule, LossOutput
# from shimmer.modules.domain import DomainModule
# from shimmer.modules.global_workspace import GlobalWorkspace2Domains, SchedulerArgs
# from shimmer.modules.vae import (
#     VAE,
#     VAEDecoder,
#     VAEEncoder,
#     gaussian_nll,
#     kl_divergence_loss,
# )
# from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
# from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
# from shimmer_ssd.dataset.pre_process import TokenizeCaptions
# from shimmer_ssd.logging import (
#     LogAttributesCallback,
#     LogGWImagesCallback,
#     LogVisualCallback,
#     batch_to_device,
# )
# from shimmer_ssd.modules.domains import load_pretrained_domains
# from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
# from shimmer_ssd.modules.vae import RAEDecoder, RAEEncoder
# from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
# from torch import nn
# from torch.nn.functional import mse_loss
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.optimizer import Optimizer
# from torchvision.utils import make_grid

# from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
# import torch
# from torch.utils.data import default_collate
# import sys



# ########################
# exclude_colors = True
# logger_name = "Reprise des hyperparamètres"
# # f"gw_{'without' if exclude_colors else 'with'} colors"
# project_name = "shimmer-ssd" 
# ###########################



# #################################config######################################
# config = load_config("./config", use_cli=False)


# config = load_config("./config", use_cli=False, load_files=["train_gw.yaml"])
# config.domain_data_args["v_latents"]["presaved_path"] = "domain_v.npy"
# config.global_workspace.latent_dim = 12


# config.domain_proportions = {
#     frozenset(["v"]): 1.0,
#     frozenset(["attr"]): 1.0,
#     frozenset(["v", "attr"]): 1.0,
# }


# my_hparams = {"temperature":0.5, "alpha": 2}
# checkpoint_path = Path("./checkpoints")


# # On entraine selon les hparams de la version 


# # Set up the global workspace configuration
# config.global_workspace.encoders.hidden_dim = 256
# config.global_workspace.decoders.hidden_dim = 256
# config.global_workspace.decoders.n_layers = 1
# config.global_workspace.encoders.n_layers = 2

# ## Set up the loss coefficients
# config.global_workspace.loss_coefficients = {
#     "cycles" : 5,
#     "contrastives" : 0.08547863808211277,
#     "demi_cycles" : 4,
#     "translations" : 1,
# }

# ## Set up the training configuration
# config.training.optim.lr = 0.0002855219529128854
# config.training.optim.max_lr = 0.002589508359966693
# config.training.optim.weight_decay = 0.00000723146540361476
# config.training.max_steps = 200000
# config.training.batch_size = 2056
# config.training.accelerator = "gpu"
# ##################################################################################





# domain_classes = get_default_domains(["v_latents", "attr"])

# # Set up domain configurations based on exclude_colors
# attr_variant = DomainModuleVariant.attr_legacy_no_color if exclude_colors else DomainModuleVariant.attr_legacy

# config.domains = [
#     LoadedDomainConfig(
#         domain_type=DomainModuleVariant.v_latents,
#         checkpoint_path=checkpoint_path / "domain_v.ckpt",
#     ),
#     LoadedDomainConfig(
#         domain_type=attr_variant,
#         checkpoint_path=checkpoint_path / "domain_attr.ckpt",
#         args=my_hparams,
#     ),
# ]





# def custom_collate(batch):
#     result = default_collate(batch)
    
#     # Check if we need to modify the second tensor in attr list
#     if (isinstance(result, dict) and "attr" in result and 
#         isinstance(result["attr"], list) and len(result["attr"]) >= 2 and
#         isinstance(result["attr"][1], torch.Tensor) and result["attr"][1].size(-1) >= 4):
        
#         # Remove the last 3 values from the tensor
#         result["attr"][1] = result["attr"][1][..., :-3]
    
#     return result








# # Create data module with optional collate_fn
# data_module = SimpleShapesDataModule(
#     config.dataset.path,
#     domain_classes,
#     config.domain_proportions,
#     batch_size=config.training.batch_size,
#     num_workers=config.training.num_workers,
#     seed=config.seed,
#     domain_args=config.domain_data_args,
#     **({"collate_fn": custom_collate} if exclude_colors else {})
# )




# domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
#     config.domains,
#     config.global_workspace.latent_dim,
#     config.global_workspace.encoders.hidden_dim,
#     config.global_workspace.encoders.n_layers,
#     config.global_workspace.decoders.hidden_dim,
#     config.global_workspace.decoders.n_layers,
# )


# def get_scheduler(optimizer: Optimizer) -> OneCycleLR:
#     return OneCycleLR(optimizer, config.training.optim.max_lr, config.training.max_steps)


# global_workspace = GlobalWorkspace2Domains(
#     domain_modules,
#     gw_encoders,
#     gw_decoders,
#     config.global_workspace.latent_dim,
#     config.global_workspace.loss_coefficients,
#     config.training.optim.lr,
#     config.training.optim.weight_decay,
#     scheduler=get_scheduler,
# )






# from lightning.pytorch.loggers.wandb import WandbLogger


# logger_wandb = WandbLogger(name=logger_name, project=project_name)


# logger = logger_wandb

# logger_wandb.log_hyperparams(my_hparams)


# train_samples = data_module.get_samples("train", 32)
# val_samples = data_module.get_samples("val", 32)


# for domains in val_samples:
#     for domain in domains:
#         val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
#     break

# (config.default_root_dir / "gw").mkdir(exist_ok=True)

# callbacks: list[Callback] = [

#     LogGWImagesCallback(
#         val_samples,
#         log_key="images/val",
#         mode="val",
#         every_n_epochs=config.logging.log_val_medias_every_n_epochs,
#         filter=config.logging.filter_images,
#         exclude_colors=exclude_colors
#     ),

#     LogGWImagesCallback(
#         train_samples,
#         log_key="images/train",
#         mode="train",
#         every_n_epochs=config.logging.log_train_medias_every_n_epochs,
#         filter=config.logging.filter_images,
#         exclude_colors=exclude_colors
#     ),

#     ModelCheckpoint(
#         dirpath=config.default_root_dir / "gw" / f"version_color{exclude_colors}_{logger.version}",
#         filename="{epoch}",
#         monitor="val/loss",
#         mode="min",
#         save_last="link",
#         save_top_k=1,
#     ),
# ]


# gw_checkpoint = config.default_root_dir / "gw" / f"version_{logger.version}"
# print(gw_checkpoint)


# trainer = Trainer(
#     logger=logger,
#     max_steps=config.training.max_steps,
#     default_root_dir=config.default_root_dir,
#     callbacks=callbacks,
#     precision=config.training.precision,
#     accelerator=config.training.accelerator,
#     devices=config.training.devices,
# )

# trainer.fit(global_workspace, data_module)
# trainer.validate(global_workspace, data_module, "best")


import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import DomainModule, LossOutput
from shimmer.modules.domain import DomainModule
from shimmer.modules.global_workspace import GlobalWorkspace2Domains, SchedulerArgs
from shimmer.modules.vae import (
    VAE,
    VAEDecoder,
    VAEEncoder,
    gaussian_nll,
    kl_divergence_loss,
)
from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR # Assuming LOGGER is configured
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config, Config
from shimmer_ssd.dataset.pre_process import TokenizeCaptions
from shimmer_ssd.logging import (
    LogAttributesCallback,
    LogGWImagesCallback,
    LogVisualCallback,
    batch_to_device,
)
from shimmer_ssd.modules.domains import load_pretrained_domains
from shimmer_ssd.modules.domains.visual import VisualLatentDomainModule
from shimmer_ssd.modules.vae import RAEDecoder, RAEEncoder
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import default_collate
from torchvision.utils import make_grid

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
import wandb
from wandb.sdk.wandb_run import Run

# Configure base logger if shimmer_ssd.LOGGER is not already set up
if not LOGGER.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

# --- Helper Functions ---

def custom_collate_factory(exclude_colors: bool):
    """Returns a collate function that optionally removes color info."""
    if not exclude_colors:
        return default_collate

    def custom_collate(batch):
        """Collate function that removes the last 3 attrs (assumed colors)."""
        result = default_collate(batch)
        # Check if we need to modify the second tensor in attr list
        if (isinstance(result, dict) and "attr" in result and
            isinstance(result["attr"], list) and len(result["attr"]) >= 2 and
            isinstance(result["attr"][1], torch.Tensor) and result["attr"][1].size(-1) >= 4):
            # Remove the last 3 values from the tensor
            result["attr"][1] = result["attr"][1][..., :-3]
        return result
    return custom_collate

def init_weights(m: nn.Module):
    """Applies Kaiming Normal initialization to Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_scheduler(optimizer: Optimizer, config: Config) -> OneCycleLR:
    """Creates the OneCycleLR scheduler based on config."""
    # Ensure necessary config values exist
    max_lr = getattr(config.training.optim, 'max_lr', 0.001) # Default if not set
    max_steps = getattr(config.training, 'max_steps', 100000) # Default if not set
    LOGGER.info(f"Creating OneCycleLR with max_lr={max_lr}, max_steps={max_steps}")
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=max_steps
    )

# --- Main Training Function ---

def train_global_workspace(
    seed: int,
    config_base_path: str | Path = "./config",
    domain_checkpoint_path: str | Path = "./checkpoints",
    output_dir: str | Path = "./lightning_logs",
    logger_name: str = "gw_training",
    project_name: str = "shimmer-ssd",
    exclude_colors: bool = True,
    custom_hparams: Optional[Dict[str, Any]] = None,
    use_wandb: bool = True,
    apply_custom_init: bool = True, # Flag to control custom initialization
) -> Tuple[Optional[str], Trainer]:
    """
    Sets up and trains the Global Workspace model based on configuration.

    Args:
        seed: Random seed for reproducibility.
        config_base_path: Path to the configuration directory.
        domain_checkpoint_path: Path to the directory containing pretrained domain checkpoints.
        output_dir: Base directory for logs and model checkpoints.
        logger_name: Name for the specific run/logger instance.
        project_name: Project name (for WandB).
        exclude_colors: Whether to exclude color attributes from the 'attr' domain.
        custom_hparams: Dictionary of hyperparameters to potentially override config
                        or pass to specific modules (like attr domain).
        use_wandb: Whether to use WandbLogger. If False, uses TensorBoardLogger.
        apply_custom_init: Whether to apply the custom `init_weights` function.

    Returns:
        A tuple containing:
        - The path to the best model checkpoint (str or None).
        - The PyTorch Lightning Trainer instance.
    """
    LOGGER.info(f"Starting Global Workspace training run with seed: {seed}")
    seed_everything(seed, workers=True)

    # --- Paths ---
    config_base_path = Path(config_base_path)
    domain_checkpoint_path = Path(domain_checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # --- Configuration ---
    LOGGER.info(f"Loading configuration from: {config_base_path}")
    # Load base config first, then the specific training config
    config = load_config(config_base_path, use_cli=False) # Load defaults if needed
    config = load_config(config_base_path, use_cli=False, load_files=["train_gw.yaml"]) # Override with train_gw.yaml

    # Apply overrides from the original script (Consider moving these fully into YAML)
    config.domain_data_args["v_latents"]["presaved_path"] = "domain_v.npy" # Hardcoded?
    config.global_workspace.latent_dim = 12
    config.domain_proportions = {
        frozenset(["v"]): 1.0,
        frozenset(["attr"]): 1.0,
        frozenset(["v", "attr"]): 1.0,
    }
    config.global_workspace.encoders.hidden_dim = 256
    config.global_workspace.decoders.hidden_dim = 256
    config.global_workspace.decoders.n_layers = 1
    config.global_workspace.encoders.n_layers = 2
    config.global_workspace.loss_coefficients = {
        "cycles" : 1, "contrastives" : 0.1,
        "demi_cycles" : 1, "translations" : 1,
    }
    # lr: 
    #     weight_decay: 
    #     max_lr: 3e-3
    #     start_lr: 8e-4
    #     end_lr: 8e-4
    #     pct_start: 0.2
    config.training.optim.lr = 4e-3
    config.training.optim.max_lr = 3e-3
    config.training.optim.weight_decay = 1e-6
    config.training.max_steps = 20000
    config.training.batch_size = 2056
    config.training.accelerator = "gpu" # Make parameterizable?
    config.training.devices = 1         # Make parameterizable?

    # Set seed and output dir in config
    config.seed = seed
    config.default_root_dir = output_dir

    # --- Domain Setup ---
    domain_classes = get_default_domains(["v_latents", "attr"])
    attr_variant = DomainModuleVariant.attr_legacy_no_color if exclude_colors else DomainModuleVariant.attr_legacy

    domain_configs = [
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.v_latents,
            checkpoint_path=domain_checkpoint_path / "domain_v.ckpt",
        ),
        LoadedDomainConfig(
            domain_type=attr_variant,
            checkpoint_path=domain_checkpoint_path / "domain_attr.ckpt",
            # Pass custom hparams specifically to the attr domain if provided
            args=custom_hparams if custom_hparams is not None else {},
        ),
    ]
    config.domains = domain_configs # Add to config object

    # --- Data Module ---
    LOGGER.info("Setting up DataModule...")
    collate_fn = custom_collate_factory(exclude_colors)
    data_module = SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=collate_fn
    )

    # --- Model Components ---
    LOGGER.info("Loading pretrained domains and setting up GW components...")
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config.domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )

    # --- Global Workspace Module ---
    LOGGER.info("Instantiating Global Workspace module...")
     # Need to define the scheduler factory function that uses the helper
    def scheduler_factory(optimizer: Optimizer) -> OneCycleLR:
        return get_scheduler(optimizer, config) # Pass config to the helper

    global_workspace = GlobalWorkspace2Domains(
        domain_modules,
        gw_encoders,
        gw_decoders,
        config.global_workspace.latent_dim,
        config.global_workspace.loss_coefficients,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler=scheduler_factory, # Pass the factory function
    )

    # --- Improved Weight Initialization (Optional) ---
    if apply_custom_init:
        LOGGER.info("Applying custom weight initialization (Kaiming Normal for Linear)...")
        global_workspace.apply(init_weights)
    else:
        LOGGER.info("Skipping custom weight initialization, using defaults.")

    # --- Logger ---
    LOGGER.info(f"Setting up logger: {'WandB' if use_wandb else 'TensorBoard'}")
    if use_wandb:
        logger = WandbLogger(
            name=logger_name,
            project=project_name,
            save_dir=output_dir,
            log_model=False, # Usually handled by ModelCheckpoint
        )
        # # Log combined hyperparameters (config + custom overrides)
        # combined_hparams = vars(config)  # Convert config object to dictionary
        # if custom_hparams:
        #     combined_hparams.update(custom_hparams)
        # logger.log_hyperparams(combined_hparams)

    # --- Callbacks ---
    LOGGER.info("Setting up callbacks...")
    # Prepare samples for logging
    try:
        train_samples = data_module.get_samples("train", 32)
        val_samples = data_module.get_samples("val", 32)
        # Add single domain samples for validation logging
        first_val_key = next(iter(val_samples.keys()))
        first_val_sample_dict = val_samples[first_val_key]
        for domain_name, domain_data in first_val_sample_dict.items():
             val_samples[frozenset([domain_name])] = {domain_name: domain_data}
    except Exception as e:
        LOGGER.error(f"Could not get samples for logging: {e}")
        train_samples = {}
        val_samples = {}


    # Define checkpoint directory using logger version for uniqueness
    run_version = logger.version if logger and hasattr(logger, 'version') else 'unknown_version'
    checkpoint_dir = output_dir / project_name / logger_name / f"version_color{exclude_colors}_{run_version}" / "checkpoints"
    LOGGER.info(f"Model checkpoints will be saved to: {checkpoint_dir}")

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{step}-{val/loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_last="link",
            save_top_k=1,
            auto_insert_metric_name=False, # Avoids clutter like 'val/loss=0.00.ckpt'
        ),
    ]
    # Add image logging callbacks only if samples were loaded
    if val_samples:
        callbacks.append(
            LogGWImagesCallback(
                val_samples,
                log_key="images/val",
                mode="val",
                every_n_epochs=config.logging.log_val_medias_every_n_epochs,
                filter=config.logging.filter_images,
                exclude_colors=exclude_colors
            )
        )
    if train_samples:
         callbacks.append(
            LogGWImagesCallback(
                train_samples,
                log_key="images/train",
                mode="train",
                every_n_epochs=config.logging.log_train_medias_every_n_epochs,
                filter=config.logging.filter_images,
                exclude_colors=exclude_colors
            )
        )

    # --- Trainer ---
    LOGGER.info("Setting up Trainer...")
    trainer = Trainer(
        logger=logger,
        max_steps=config.training.max_steps,
        # default_root_dir=config.default_root_dir, # default_root_dir is mainly for logs/checkpoints if logger/checkpoint paths aren't specified
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        # gradient_clip_val=1.0
    )

    # --- Training & Validation ---
    LOGGER.info("Starting training...")
    trainer.fit(global_workspace, datamodule=data_module)

    LOGGER.info("Starting final validation on best checkpoint...")
    # Find the ModelCheckpoint callback to get the best model path
    best_model_path = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_model_path = cb.best_model_path
            break

    if best_model_path and Path(best_model_path).exists():
         LOGGER.info(f"Loading best model from: {best_model_path}")
         trainer.validate(global_workspace, datamodule=data_module, ckpt_path="best") # Use "best" keyword
    elif trainer.checkpoint_callback and trainer.checkpoint_callback.last_model_path and Path(trainer.checkpoint_callback.last_model_path).exists():
        LOGGER.warning("Best checkpoint not found or invalid. Validating last checkpoint instead.")
        trainer.validate(global_workspace, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.last_model_path)
    else:
        LOGGER.warning("No valid checkpoint found for final validation.")


    LOGGER.info(f"Training run {logger_name} completed.")
    # Update best_model_path after validation call with ckpt_path='best' might be needed if path changes
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_model_path = cb.best_model_path
            break

    return best_model_path, trainer



# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration for the Example Run ---
    SEED = 1
    CONFIG_PATH = "./config"
    CHECKPOINT_PATH = Path("./checkpoints") # Make sure domain checkpoints exist here
    OUTPUT_DIR = Path("./gw_training_output")
    LOGGER_NAME = "GW_Run_Example"
    PROJECT_NAME = "shimmer-ssd-gw-highcycle_alphawann"
    EXCLUDE_COLORS = True # Set to False to include colors
    USE_WANDB = True # Set to True to log to WandB (requires wandb login)
    APPLY_CUSTOM_INIT = True

    # Custom hyperparameters (e.g., for the attr domain or overrides)
    # Matches the hparams from the original script for the attr domain
    CUSTOM_HPARAMS = {"temperature": 1, "alpha": 1}
    
    # Ensure domain checkpoints exist (or download/train them first)
    # Example check:
    if not (CHECKPOINT_PATH / "domain_v.ckpt").exists() or \
       not (CHECKPOINT_PATH / "domain_attr.ckpt").exists():
        LOGGER.error(f"Domain checkpoints not found in {CHECKPOINT_PATH}. Please ensure 'domain_v.ckpt' and 'domain_attr.ckpt' exist.")
        # sys.exit(1) # Optional: exit if checkpoints are missing

    LOGGER.info("--- Starting Example Global Workspace Training ---")

    # best_ckpt_path, final_trainer = train_global_workspace(
    #     seed=SEED,
    #     config_base_path=CONFIG_PATH,
    #     domain_checkpoint_path=CHECKPOINT_PATH,
    #     output_dir=OUTPUT_DIR,
    #     logger_name=LOGGER_NAME,
    #     project_name=PROJECT_NAME,
    #     exclude_colors=EXCLUDE_COLORS,
    #     custom_hparams=CUSTOM_HPARAMS,
    #     use_wandb=USE_WANDB,
    #     apply_custom_init=APPLY_CUSTOM_INIT,
    # )

    # LOGGER.info("--- Example Run Finished ---")
    # if best_ckpt_path:
    #     LOGGER.info(f"Best model checkpoint saved at: {best_ckpt_path}")
    # else:
    #     LOGGER.warning("No best model checkpoint path was reported.")


if __name__ == "__main__":
    # Example usage of the training function
    train_global_workspace(seed = 0, project_name="shimmer-ssd", logger_name="Base_params", custom_hparams=CUSTOM_HPARAMS)