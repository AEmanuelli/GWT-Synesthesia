"""Entraînement d'un Global Workspace (mémoire §2.2).

Exemples :
    # Vide sémantique (couleur retirée des attributs), configuration 4, seed 0
    python SSD_eval/SSD_Train.py --experiment high_cycles --seed 0

    # Modèle Complet de contrôle (couleur présente dans les attributs)
    python SSD_eval/SSD_Train.py --experiment high_cycles --seed 0 --with-colors

    # Configuration 5 (poids x5 sur les attributs continus)
    python SSD_eval/SSD_Train.py --experiment high_alpha --attr-alpha 5 --seed 0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import default_collate

from shimmer_ssd.config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

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

def init_weights(m: nn.Module, seed: int):
    """Applies Kaiming Normal initialization to Linear layers."""
    if isinstance(m, nn.Linear):
        # Infer device from the weight tensor itself
        device = m.weight.device
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu',
                                generator=torch.Generator(device=device).manual_seed(seed))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def setup_data_module(config, exclude_colors=True):
    """
    Set up the data module for training.
    
    Args:
        config: Configuration with dataset parameters
        
    Returns:
        SimpleShapesDataModule: Configured data module
    """
    from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
    
    
    domain_classes = get_default_domains(["v_latents", "attr"])
    
    return SimpleShapesDataModule(
        config.dataset.path,
        domain_classes,
        config.domain_proportions,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.seed,
        domain_args=config.domain_data_args,
        collate_fn=custom_collate_factory(exclude_colors)
    )


def setup_global_workspace(config, hparams, exclude_colors=True, apply_custom_init=True, load_from_checkpoint=None):
    """
    Set up the global workspace model.
    
    Args:
        config: Configuration with model parameters
        hparams: Hyperparameters dictionary
        
    Returns:
        tuple: (global_workspace, domain_modules)
    """
    from pathlib import Path
    from shimmer_ssd.config import LoadedDomainConfig, DomainModuleVariant
    from shimmer_ssd.modules.domains import load_pretrained_domains
    from shimmer.modules.global_workspace import GlobalWorkspace2Domains
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.optim.optimizer import Optimizer
    
    # Set up domain configurations
    domains = [
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.v_latents,
            checkpoint_path=paths.DOMAIN_V_CKPT,
        ),
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.attr_legacy_no_color if exclude_colors else DomainModuleVariant.attr_legacy,
            checkpoint_path=paths.DOMAIN_ATTR_CKPT,
            args=hparams,
        ),
    ]
    
    # Create scheduler function
    def get_scheduler(optimizer: Optimizer, scheduler_type: str = "onecycle"):
        if scheduler_type == "onecycle":
            return OneCycleLR(optimizer, config.training.optim.max_lr, 
                          int(config.training.max_steps), 
                          pct_start=config.training.optim.pct_start, div_factor=.38, final_div_factor=5)
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.training.max_steps
                )
        elif scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=config.training.max_steps,
                eta_min=config.training.optim.lr / 100
            )
        else:
            print(f"Scheduler type {scheduler_type} not supported")
            return None
    # Load domains and create GW
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        domains,
        config.global_workspace.latent_dim,
        config.global_workspace.encoders.hidden_dim,
        config.global_workspace.encoders.n_layers,
        config.global_workspace.decoders.hidden_dim,
        config.global_workspace.decoders.n_layers,
    )
    
    # Adjust learning rate based on loss coefficients
    # lr = config.training.optim.lr * 3.1 / sum(config.global_workspace.loss_coefficients.values()) #### TODO: check if this is correct
    lr = config.training.optim.lr
    # Create global workspace
    

    if apply_custom_init:
        # Only apply initialization to GW encoders and decoders, not to the pretrained visual module
        for modality in gw_encoders:
            encoder = gw_encoders[modality]
            encoder.apply(lambda m: init_weights(m, config.seed))
        for modality in gw_decoders:
            decoder = gw_decoders[modality]
            decoder.apply(lambda m: init_weights(m, config.seed))

    global_workspace = GlobalWorkspace2Domains(
        domain_modules,
        gw_encoders,
        gw_decoders,
        config.global_workspace.latent_dim,
        config.global_workspace.loss_coefficients,
        lr,
        config.training.optim.weight_decay,
        scheduler=get_scheduler,
    )
    
    # Optionnel : reprendre les poids d'un GW existant (chemin vers un .ckpt)
    if load_from_checkpoint:
        print(f"Loading weights from checkpoint: {load_from_checkpoint}")
        global_workspace = GlobalWorkspace2Domains.load_from_checkpoint(
            load_from_checkpoint,
            domain_mods=domain_modules,
            gw_encoders=gw_encoders,
            gw_decoders=gw_decoders,
        )
        global_workspace.optimizer = None
        global_workspace.scheduler = None
        global_workspace.callbacks = None

    return global_workspace, domain_modules


def setup_logger_and_callbacks(config, 
                               data_module, 
                               experiment_name="gw_no_color", 
                               project_name="shimmer-ssd", 
                               exclude_colors=True):
    """
    Set up logging and callbacks for training.
    
    Args:
        config: Configuration with logging parameters
        data_module: Data module to get samples from
        experiment_name: Name for the wandb experiment
        
    Returns:
        tuple: (logger, callbacks, checkpoint_dir)
    """
    from lightning.pytorch.loggers.wandb import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from shimmer_ssd.logging import LogGWImagesCallback
    
    output_dir = Path(config.default_root_dir) / experiment_name

    # Set up logger
    logger = WandbLogger(
            name=experiment_name,
            project=project_name,
            save_dir=output_dir,
            log_model=False, # Usually handled by ModelCheckpoint
            # Mode hors-ligne si aucune clé W&B n'est configurée
            offline=not os.environ.get("WANDB_API_KEY"),

        )
    

    # Get samples for visualization
    train_samples = data_module.get_samples("train", 32)
    val_samples = data_module.get_samples("val", 32)
    
    
    # Split validation samples
    for domains in val_samples:
        for domain in domains:
            val_samples[frozenset([domain])] = {domain: val_samples[domains][domain]}
        break
    
    # Create checkpoint directory
    # run_version = logger.version if logger and hasattr(logger, 'version') else 'unknown_version'
    version_dir = output_dir / "checkpoints"
    print(f"Model checkpoints will be saved to: {version_dir}")
    
    # Set up callbacks
    callbacks = [
        LogGWImagesCallback(
            val_samples,
            log_key="images/val",
            mode="val",
            every_n_epochs=config.logging.log_val_medias_every_n_epochs,
            filter=config.logging.filter_images,
            exclude_colors=exclude_colors
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=config.logging.log_train_medias_every_n_epochs,
            filter=config.logging.filter_images,
            exclude_colors=exclude_colors
        ),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=version_dir,
            filename="{epoch}",
            monitor="val/loss",
            mode="min",
            save_last="link",
            save_top_k=1,
        ),
    ]
    
    return logger, callbacks, version_dir


def train_global_workspace(config, custom_hparams=None, experiment_name="debugging", project_name="gwt-synesthesia", apply_custom_init=True, exclude_colors=True, load_from_checkpoint=None):
    """
    Train a global workspace model with the given configuration.
    
    Args:
        config: Configuration object with model and training parameters
        custom_hparams: Optional dictionary of hyperparameters to override defaults
        experiment_name: Name for the wandb experiment
        
    Returns:
        tuple: (trained_model, checkpoint_path)
    """
    from lightning.pytorch import Trainer
    
    # Initialize default hyperparameters
    hparams = {"temperature": 1, "alpha": 1}
    if custom_hparams:
        hparams.update(custom_hparams)
    
    # 1. Set up data module
    data_module = setup_data_module(config, exclude_colors=exclude_colors)
    
    # 2. Set up global workspace
    global_workspace, _ = setup_global_workspace(config, hparams, exclude_colors=exclude_colors, apply_custom_init=apply_custom_init, load_from_checkpoint=load_from_checkpoint)
    
    # 3. Set up logger and callbacks
    logger, callbacks, checkpoint_dir = setup_logger_and_callbacks(
        config, data_module, experiment_name, project_name, exclude_colors=exclude_colors
    )
    
    # Log hyperparameters
    hparams_to_log = {
        # Domain HParams
        **hparams, # Logs temperature and alpha
        # Model Architecture
        "encoder_size": config.global_workspace.encoders.hidden_dim,
        "decoder_size": config.global_workspace.decoders.hidden_dim,
        "encoder_layers": config.global_workspace.encoders.n_layers,
        "decoder_layers": config.global_workspace.decoders.n_layers,
        "latent_dim": config.global_workspace.latent_dim,
        # Optimizer / Training
        "lr_base": config.training.optim.lr, # Log the base LR from config
        "max_lr": config.training.optim.max_lr,
        "weight_decay": config.training.optim.weight_decay,
        "max_steps": config.training.max_steps,
        "batch_size": config.training.batch_size,
        # Other settings
        "seed": config.seed, # Log seed if set in config
        "exclude_colors": exclude_colors # Logging the setting used for this run
    }
    logger.log_hyperparams(hparams_to_log)
    
    # 4. Create trainer
    trainer = Trainer(
        logger=logger,
        max_steps=config.training.max_steps,
        default_root_dir=config.default_root_dir,
        callbacks=callbacks,
        precision=config.training.precision,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        gradient_clip_val=1.0,  # Set your desired clipping value here
        gradient_clip_algorithm="value",
    )
    
    # 5. Train and validate
    trainer.fit(global_workspace, data_module)
    trainer.validate(global_workspace, data_module, "best")

    wandb.finish()

    
    return global_workspace, checkpoint_dir

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True,
                        help="Nom du fichier de config (sans .yaml) dans config/, ex. base_params, high_cycles")
    parser.add_argument("--seed", type=int, default=0, help="Graine (initialisation Kaiming + data)")
    parser.add_argument("--with-colors", action="store_true",
                        help="Modèle Complet (contrôle) : garder R,G,B dans le vecteur d'attributs")
    parser.add_argument("--attr-alpha", type=float, default=1.0,
                        help="Poids relatif des attributs continus dans la loss attributs (5 pour la configuration 5)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None, help="Surcharge training.max_steps")
    parser.add_argument("--run-name", default=None,
                        help="Nom du dossier de sortie (défaut : <experiment>_seed<seed> ou <experiment>_control)")
    parser.add_argument("--project", default="gwt-synesthesia", help="Projet W&B")
    parser.add_argument("--resume-weights", default=None, help="(optionnel) .ckpt de GW dont repartir")
    parser.add_argument("--detect-anomaly", action="store_true",
                        help="Active torch.autograd.set_detect_anomaly (activé lors des runs d'origine, très lent)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    config = load_config(paths.CONFIG_DIR, use_cli=False, load_files=[f"{args.experiment}.yaml"])
    config.seed = args.seed
    config.dataset.path = paths.DATASET_DIR
    config.default_root_dir = paths.RUNS_DIR
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps

    exclude_colors = not args.with_colors
    run_name = args.run_name or (
        f"{args.experiment}_control" if args.with_colors else f"{args.experiment}_seed{args.seed}"
    )

    model, checkpoint_dir = train_global_workspace(
        config,
        custom_hparams={"temperature": args.temperature, "alpha": args.attr_alpha},
        project_name=args.project,
        experiment_name=run_name,
        apply_custom_init=True,
        exclude_colors=exclude_colors,
        load_from_checkpoint=args.resume_weights,
    )
    print(f"Done. Checkpoints in {checkpoint_dir}")
