# ======================================================
#                 IMPORTS & SETUP
# ======================================================
import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import itertools
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional, Any, Set
import sys
import gc # Garbage collection
import wandb
import statistics
import json

# --- PyTorch Lightning & Related ---
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import default_collate

# --- Shimmer & SSD Specific ---
try:
    from shimmer import DomainModule, LossOutput
    from shimmer.modules.domain import DomainModule
    from shimmer.modules.global_workspace import GlobalWorkspace2Domains, SchedulerArgs
    from shimmer_ssd import DEBUG_MODE as SSD_DEBUG_MODE, LOGGER as SSD_LOGGER, PROJECT_DIR
    from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config, Config
    from shimmer_ssd.logging import LogGWImagesCallback
    from shimmer_ssd.modules.domains import load_pretrained_domains
    from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
    # Evaluation/Analysis specific imports
    from SSD_utils import (
        generate_fixed_colors, kl_divergence,
    )
    from SSD_H_evaluation_functions import (
         HueShapeAnalyzer,
    )
    # Helper functions
    from SSD_arbitrary import (run_evaluation, find_significant_bin_comparisons, load_hue_data_for_bin,
     visualize_multiple_distributions_overlay, compare_hue_distribution_for_bin_across_models)
    from SSD_eval.SSD_Train_old import custom_collate_factory, get_scheduler, init_weights

except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the correct directory or paths are configured.")
    exit(1)

# --- Setup Logging ---
LOGGER = logging.getLogger("SweepKLTrainAndCompare")
if not LOGGER.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

# --- Constants ---
MODEL_VERSION_NAME = "kl_optimized_sweep"
BASE_CONFIG_PATH = Path("./config")
BASE_DOMAIN_CHECKPOINT_PATH = Path("./checkpoints")
BASE_OUTPUT_DIR = Path("./gw_sweep_kl_optimization")
PROJECT_NAME_WANDB = "shimmer-ssd-gw-kl-sweep"
EXCLUDE_COLORS_TRAINING = True
DEBUG_MODE = False
MAX_TRAINING_STEPS = 243 if DEBUG_MODE else 86751

# KL Divergence threshold for model selection
KL_THRESHOLD = 2.0
# Number of additional models to train per promising configuration
ADDITIONAL_MODELS_PER_CONFIG = 2

# Evaluation & Comparison Constants
EVALUATION_PARENT_DIR = BASE_OUTPUT_DIR / "evaluation_runs"
COMPARISON_OUTPUT_DIR = BASE_OUTPUT_DIR / "cross_model_comparison"
FILTERING_ALPHA = 0.05
COMPARISON_SAVE_ALPHA = 0.05

# Metadata tracking file for sweep results
SWEEP_METADATA_FILE = BASE_OUTPUT_DIR / "sweep_results.json"

# Default hyperparameters (will be overridden by sweep)
DEFAULT_ENCODERS_N_LAYERS = 1
DEFAULT_DECODERS_N_LAYERS = 2
DEFAULT_TEMPERATURE = 0.7
DEFAULT_ALPHA = 1.3
DEFAULT_CYCLE = 0.8
DEFAULT_DEMI_CYCLE = 3.4
DEFAULT_TRADUCTION = 0.7
DEFAULT_CONTRASTIVE = 0.08
DEFAULT_WD = 0.0000072
DEFAULT_LR = 0.00028
DEFAULT_MAX_LR = 0.0026

# ======================================================
#     SWEEP CONFIGURATION DEFINITION
# ======================================================
def get_sweep_config():
    """Define the sweep configuration for W&B"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val/loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'seed': {
                'values': list(range(5))  # Start with 5 different seeds
            },
            'encoders_n_layers': {
                'values': [1, 2, 3]
            },
            'decoders_n_layers': {
                'values': [1, 2, 3]
            },
            'temperature': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.0
            },
            'alpha': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 1.8
            },
            'cycle': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.2
            },
            'demi_cycle': {
                'distribution': 'uniform',
                'min': 2.0,
                'max': 4.5
            },
            'traduction': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.0
            },
            'contrastive': {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.2
            },
            'weight_decay': {
                'distribution': 'log_uniform',
                'min': -8,  # 10^-8
                'max': -4   # 10^-4
            },
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': -5,  # 10^-5
                'max': -3   # 10^-3
            },
            'max_lr': {
                'distribution': 'log_uniform',
                'min': -4,  # 10^-4
                'max': -2   # 10^-2
            }
        }
    }
    return sweep_config

# ======================================================
#     UTILITY FUNCTIONS FOR KL DIVERGENCE ANALYSIS
# ======================================================
def calculate_average_kl_divergence(analysis_results):
    """Calculate the average KL divergence across all paths and attributes"""
    paths_to_check = ['translated', 'half_cycle', 'full_cycle']
    all_kl_values = []
    
    for path_name in paths_to_check:
        if path_name not in analysis_results:
            continue
            
        path_data = analysis_results[path_name]
        for attr, attr_data in path_data.items():
            for bin_name, bin_data in attr_data.items():
                # Ensure bin_data is a dictionary with kl_divergence
                if isinstance(bin_data, dict) and 'kl_divergence' in bin_data:
                    kl_value = bin_data['kl_divergence']
                    if isinstance(kl_value, (int, float)) and not np.isnan(kl_value):
                        all_kl_values.append(kl_value)
    
    if not all_kl_values:
        return 0.0
        
    return statistics.mean(all_kl_values)

def load_sweep_metadata():
    """Load the sweep metadata from file"""
    if not SWEEP_METADATA_FILE.exists():
        return {}
        
    try:
        with open(SWEEP_METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        LOGGER.error(f"Error loading sweep metadata: {e}")
        return {}
        
def save_sweep_metadata(metadata):
    """Save the sweep metadata to file"""
    SWEEP_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(SWEEP_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Error saving sweep metadata: {e}")

# ======================================================
#      TRAINING FUNCTION ADAPTED FOR SWEEP
# ======================================================
def train_global_workspace_sweep(config=None, specified_seed=None):
    """Sets up and trains the Global Workspace model with W&B sweep configuration."""
    # Initialize wandb run
    with wandb.init(config=config) as run:
        # Access all hyperparameters through wandb.config
        config = wandb.config
        
        # Override seed if specified (for additional models with same config)
        seed = specified_seed if specified_seed is not None else int(config.seed)
        
        # Extract parameters from sweep config
        encoders_n_layers = int(config.encoders_n_layers)
        decoders_n_layers = int(config.decoders_n_layers)
        temperature = float(config.temperature)
        alpha = float(config.alpha)
        cycle = float(config.cycle)
        demi_cycle = float(config.demi_cycle)
        traduction = float(config.traduction)
        contrastive = float(config.contrastive)
        weight_decay = float(config.weight_decay)
        learning_rate = float(config.learning_rate)
        max_lr = float(config.max_lr)
        
        # Set run name to include seed information
        if specified_seed is not None:
            run.name = f"additional_seed{seed}_from_{run.name}"
        
        # Log hyperparameters
        LOGGER.info(f"Starting run with parameters: seed={seed}, encoders_n_layers={encoders_n_layers}, "
                   f"decoders_n_layers={decoders_n_layers}, temperature={temperature:.4f}, alpha={alpha:.4f}, "
                   f"cycle={cycle:.4f}, demi_cycle={demi_cycle:.4f}, traduction={traduction:.4f}, "
                   f"contrastive={contrastive:.4f}, wd={weight_decay:.8f}, lr={learning_rate:.8f}, max_lr={max_lr:.8f}")
        
        # Generate run ID and output directories
        run_id = run.id
        if specified_seed is not None:
            run_id = f"{run_id}_additional{seed}"
        
        run_logger_name = f"GW_Sweep_{run_id}_seed{seed}_layers{encoders_n_layers}_{decoders_n_layers}"
        run_output_dir = BASE_OUTPUT_DIR / "sweep_runs" / run_logger_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        seed_everything(seed, workers=True)
        
        # --- Load and configure the model ---
        LOGGER.info(f"Loading configuration from: {BASE_CONFIG_PATH}")
        model_config = load_config(BASE_CONFIG_PATH, use_cli=False, load_files=["train_gw.yaml"])
        
        # Apply all the hyperparameters from the sweep
        model_config.domain_data_args["v_latents"]["presaved_path"] = "domain_v.npy"
        model_config.global_workspace.latent_dim = 12
        model_config.domain_proportions = { frozenset(["v"]): 1.0, frozenset(["attr"]): 1.0, frozenset(["v", "attr"]): 1.0 }
        model_config.global_workspace.encoders.hidden_dim = 256
        model_config.global_workspace.decoders.hidden_dim = 256
        model_config.global_workspace.encoders.n_layers = encoders_n_layers
        model_config.global_workspace.decoders.n_layers = decoders_n_layers
        model_config.global_workspace.loss_coefficients = {
            "cycles": cycle,
            "contrastives": contrastive,
            "demi_cycles": demi_cycle,
            "translations": traduction
        }
        model_config.training.optim.lr = learning_rate
        model_config.training.optim.max_lr = max_lr
        model_config.training.optim.weight_decay = weight_decay
        model_config.training.max_steps = MAX_TRAINING_STEPS
        model_config.training.batch_size = 2056 if not DEBUG_MODE else 64
        model_config.training.accelerator = "gpu"
        model_config.training.devices = 1
        model_config.seed = seed
        model_config.default_root_dir = run_output_dir
        
        # --- Domain Setup ---
        domain_classes = get_default_domains(["v_latents", "attr"])
        attr_variant = DomainModuleVariant.attr_legacy_no_color if EXCLUDE_COLORS_TRAINING else DomainModuleVariant.attr_legacy
        model_config.domains = [
            LoadedDomainConfig(domain_type=DomainModuleVariant.v_latents, checkpoint_path=BASE_DOMAIN_CHECKPOINT_PATH/"domain_v.ckpt"),
            LoadedDomainConfig(domain_type=attr_variant, checkpoint_path=BASE_DOMAIN_CHECKPOINT_PATH/"domain_attr.ckpt", 
                              args={"temperature": temperature, "alpha": alpha}),
        ]
        
        # --- Data Module ---
        LOGGER.info("Setting up DataModule...")
        collate_fn = custom_collate_factory(EXCLUDE_COLORS_TRAINING)
        if not Path(model_config.dataset.path).exists():
            LOGGER.error(f"Dataset path not found: {model_config.dataset.path}. Please ensure the dataset exists.")
            return None, None, None
            
        data_module = SimpleShapesDataModule(
            model_config.dataset.path, domain_classes, model_config.domain_proportions,
            batch_size=model_config.training.batch_size, num_workers=model_config.training.num_workers,
            seed=model_config.seed, domain_args=model_config.domain_data_args, collate_fn=collate_fn
        )
        
        # --- Model Components & GW Module ---
        LOGGER.info("Setting up Global Workspace module...")
        try:
            domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
                model_config.domains, model_config.global_workspace.latent_dim,
                model_config.global_workspace.encoders.hidden_dim, model_config.global_workspace.encoders.n_layers,
                model_config.global_workspace.decoders.hidden_dim, model_config.global_workspace.decoders.n_layers,
            )
            def scheduler_factory(optimizer: Optimizer) -> OneCycleLR: return get_scheduler(optimizer, model_config)
            global_workspace = GlobalWorkspace2Domains(
                domain_modules, gw_encoders, gw_decoders, model_config.global_workspace.latent_dim,
                model_config.global_workspace.loss_coefficients, model_config.training.optim.lr,
                model_config.training.optim.weight_decay, scheduler=scheduler_factory,
            )
            
            # Apply custom weight initialization
            global_workspace.apply(init_weights)
            
        except Exception as e:
            LOGGER.error(f"Error setting up GW model components: {e}", exc_info=True)
            return None, None, None
        
        # --- Set up WandB Logger (already initialized) ---
        logger = WandbLogger(log_model=False)
        
        # --- Callbacks ---
        LOGGER.info("Setting up callbacks...")
        checkpoint_dir = run_output_dir / "checkpoints"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, filename="{epoch}-{step}-{val/loss:.2f}",
            monitor="val/loss", mode="min", save_last="link", save_top_k=1,
            auto_insert_metric_name=False,
        )
        callbacks = [checkpoint_callback]
        
        # --- Trainer ---
        LOGGER.info("Setting up Trainer...")
        trainer = Trainer(
            logger=logger, max_steps=model_config.training.max_steps,
            callbacks=callbacks, precision=model_config.training.precision,
            accelerator=model_config.training.accelerator, devices=model_config.training.devices,
        )
        
        # --- Training & Validation ---
        best_model_path = None
        val_loss = float('inf')
        avg_kl_divergence = 0.0
        
        try:
            LOGGER.info(f"Starting training for {run_logger_name}...")
            trainer.fit(global_workspace, datamodule=data_module)
            LOGGER.info(f"Training finished for {run_logger_name}. Validating best model...")
            validation_results = trainer.validate(datamodule=data_module, ckpt_path="best")
            LOGGER.info(f"Validation results for best model: {validation_results}")
            best_model_path = checkpoint_callback.best_model_path
            
            # Get validation loss
            if validation_results and isinstance(validation_results, list) and len(validation_results) > 0:
                val_loss = validation_results[0].get("val/loss", float('inf'))
                wandb.log({"best_val_loss": val_loss})
            
            # Run evaluation on the best model to calculate KL divergence
            if best_model_path:
                LOGGER.info(f"Running evaluation for the best model: {best_model_path}")
                eval_result = run_evaluation(
                    full_attr=not EXCLUDE_COLORS_TRAINING,
                    run_id=f"sweep_{run_id}",
                    gw_checkpoint_path=best_model_path,
                    model_version=MODEL_VERSION_NAME,
                    output_parent_dir=EVALUATION_PARENT_DIR,
                    encoders_n_layers=encoders_n_layers,
                    decoders_n_layers=decoders_n_layers,
                    debug_mode=DEBUG_MODE
                )
                
                if eval_result:
                    results_path, analysis_results = eval_result
                    LOGGER.info(f"Evaluation successful. Results saved to: {results_path}")
                    
                    # Calculate average KL divergence
                    avg_kl_divergence = calculate_average_kl_divergence(analysis_results)
                    LOGGER.info(f"Average KL Divergence: {avg_kl_divergence}")
                    wandb.log({"avg_kl_divergence": avg_kl_divergence})
                    
                    # Log if this is a promising model (high KL, low loss)
                    is_promising = avg_kl_divergence > KL_THRESHOLD and val_loss < float('inf')
                    wandb.log({"is_promising_model": is_promising})
                    
                    # Save metadata about this run
                    metadata = load_sweep_metadata()
                    config_dict = {k: v for k, v in config.items()}
                    
                    run_metadata = {
                        "run_id": run_id,
                        "checkpoint_path": best_model_path,
                        "val_loss": val_loss,
                        "avg_kl_divergence": avg_kl_divergence,
                        "seed": seed,
                        "config": config_dict,
                        "is_promising": is_promising,
                        "additional_seeds": [],
                        "parent_run_id": None if specified_seed is None else run_id.split("_additional")[0]
                    }
                    
                    metadata[run_id] = run_metadata
                    save_sweep_metadata(metadata)
                else:
                    LOGGER.error(f"Evaluation failed for the best model.")
                    
            # Return the path, loss, and KL divergence
            return best_model_path, val_loss, avg_kl_divergence
            
        except Exception as e:
            LOGGER.error(f"Error during training or validation: {e}", exc_info=True)
            return None, float('inf'), 0.0
        finally:
            # Cleanup
            del global_workspace, trainer, data_module, domain_modules, gw_encoders, gw_decoders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

# ======================================================
#     TRAIN ADDITIONAL MODELS FOR PROMISING CONFIGS
# ======================================================
def train_additional_models_for_promising_configs():
    """Train additional models for promising configurations"""
    metadata = load_sweep_metadata()
    
    # Find promising configurations (high KL, low loss) that don't have additional models yet
    promising_configs = []
    
    for run_id, run_data in metadata.items():
        # Skip runs that are already additional models
        if run_data.get("parent_run_id") is not None:
            continue
            
        # Check if this is a promising configuration
        if run_data.get("is_promising", False) and len(run_data.get("additional_seeds", [])) < ADDITIONAL_MODELS_PER_CONFIG:
            promising_configs.append(run_id)
    
    LOGGER.info(f"Found {len(promising_configs)} promising configurations for additional training")
    
    for run_id in promising_configs:
        run_data = metadata[run_id]
        config = run_data["config"]
        current_additional_seeds = set(run_data.get("additional_seeds", []))
        
        # Find seeds that haven't been used yet
        used_seeds = {run_data["seed"]} | current_additional_seeds
        additional_seeds_needed = ADDITIONAL_MODELS_PER_CONFIG - len(current_additional_seeds)
        
        # Generate new seeds
        new_seeds = []
        seed = 100  # Start with a high seed number to avoid overlaps
        while len(new_seeds) < additional_seeds_needed:
            if seed not in used_seeds:
                new_seeds.append(seed)
            seed += 1
        
        LOGGER.info(f"Training {len(new_seeds)} additional models for configuration {run_id}")
        
        # Train models with these new seeds
        for new_seed in new_seeds:
            LOGGER.info(f"Training model with seed {new_seed} using configuration from {run_id}")
            
            # Create a wandb config object
            wandb_config = wandb.Config()
            for key, value in config.items():
                setattr(wandb_config, key, value)
            
            # Train model with this configuration but new seed
            result = train_global_workspace_sweep(config=wandb_config, specified_seed=new_seed)
            
            if result[0]:  # If training was successful
                # Update metadata
                metadata = load_sweep_metadata()  # Reload to get any new changes
                if run_id in metadata:  # Check if the run still exists
                    metadata[run_id]["additional_seeds"].append(new_seed)
                    save_sweep_metadata(metadata)
                    LOGGER.info(f"Successfully trained additional model with seed {new_seed}")
                else:
                    LOGGER.warning(f"Original run {run_id} no longer in metadata, cannot update additional seeds")
            else:
                LOGGER.error(f"Failed to train additional model with seed {new_seed}")

# ======================================================
#     COMPARE MODELS FROM PROMISING CONFIGURATIONS
# ======================================================
def compare_models_from_promising_configs():
    """Compare models from promising configurations"""
    metadata = load_sweep_metadata()
    
    # Find promising configurations with additional models
    promising_configs_with_models = []
    
    for run_id, run_data in metadata.items():
        # Skip runs that are already additional models
        if run_data.get("parent_run_id") is not None:
            continue
            
        # Check if this is a promising configuration with additional models
        if run_data.get("is_promising", False) and len(run_data.get("additional_seeds", [])) > 0:
            promising_configs_with_models.append(run_id)
    
    LOGGER.info(f"Found {len(promising_configs_with_models)} promising configurations with additional models for comparison")
    
    for run_id in promising_configs_with_models:
        run_data = metadata[run_id]
        additional_seeds = run_data.get("additional_seeds", [])
        
        # Get all run IDs for this configuration family
        family_run_ids = [run_id]
        for seed in additional_seeds:
            # Find the additional run ID for this seed
            for additional_id, additional_data in metadata.items():
                if additional_data.get("parent_run_id") == run_id and additional_data.get("seed") == seed:
                    family_run_ids.append(additional_id)
                    break
        
        # Only compare if we have at least 2 models
        if len(family_run_ids) < 2:
            LOGGER.warning(f"Configuration {run_id} has less than 2 models for comparison, skipping")
            continue
            
        LOGGER.info(f"Comparing {len(family_run_ids)} models for configuration {run_id}: {family_run_ids}")
        
        # Get model checkpoints for comparison
        model_checkpoints = {}
        for model_id in family_run_ids:
            if model_id in metadata and "checkpoint_path" in metadata[model_id]:
                model_checkpoints[model_id] = metadata[model_id]["checkpoint_path"]
        
        if len(model_checkpoints) < 2:
            LOGGER.warning(f"Configuration {run_id} has less than 2 valid checkpoints, skipping")
            continue
            
        # Now run the comparison logic similar to the original script
        # -------------------------------------------------
        # PHASE 3-4: Filter Attributes & Comparison
        # -------------------------------------------------
        
        # Set up comparison directory
        comparison_dir = COMPARISON_OUTPUT_DIR / f"config_{run_id}"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose the first model for significance filtering
        first_model_id = family_run_ids[0]
        first_model_eval_path = None
        
        # Find the evaluation results path for the first model
        for model_id, model_path in model_checkpoints.items():
            run_specific_dir_name = f"results_{MODEL_VERSION_NAME}_{'sans' if EXCLUDE_COLORS_TRAINING else 'avec'}_couleurs_sweep_{model_id}"
            eval_path = EVALUATION_PARENT_DIR / run_specific_dir_name / "analysis_results.pkl"
            if eval_path.exists():
                if model_id == first_model_id:
                    first_model_eval_path = eval_path
                    break
        
        if not first_model_eval_path:
            LOGGER.error(f"Cannot find evaluation results for first model {first_model_id}, skipping comparison")
            continue
            
        # Load first model results for filtering
        try:
            with open(first_model_eval_path, 'rb') as f:
                first_model_results = pickle.load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load first model results: {e}, skipping comparison")
            continue
        
        # Find significant attributes for filtering
        paths_to_check = ['translated', 'half_cycle', 'full_cycle']
        all_significant_attributes_union = {}
        
        for path_name in paths_to_check:
            significant_attributes_current_path = find_significant_bin_comparisons(
                analysis_results=first_model_results, path_name=path_name,
                alpha=FILTERING_ALPHA
            )
            for attr, bins_set in significant_attributes_current_path.items():
                all_significant_attributes_union.setdefault(attr, set()).update(bins_set)
        
        if not all_significant_attributes_union:
            LOGGER.info(f"No attributes met the filtering criteria for {first_model_id}, skipping comparison")
            continue
            
        # Find evaluation results paths for all models
        model_eval_paths = {}
        for model_id, model_path in model_checkpoints.items():
            run_specific_dir_name = f"results_{MODEL_VERSION_NAME}_{'sans' if EXCLUDE_COLORS_TRAINING else 'avec'}_couleurs_sweep_{model_id}"
            eval_path = EVALUATION_PARENT_DIR / run_specific_dir_name / "analysis_results.pkl"
            if eval_path.exists():
                model_eval_paths[model_id] = eval_path
        
        if len(model_eval_paths) < 2:
            LOGGER.warning(f"Less than 2 models have evaluation results for {run_id}, skipping comparison")
            continue
            
        # Create all pairs for comparison
        model_pairs = list(itertools.combinations(model_eval_paths.keys(), 2))
        
        # Log comparison details
        LOGGER.info(f"Running comparison for {len(model_pairs)} pairs with {len(all_significant_attributes_union)} significant attributes")
        
        # Setup output directory for this specific comparison
        comparison_output_dir_final = comparison_dir / f"filter{FILTERING_ALPHA}_save{COMPARISON_SAVE_ALPHA}"
        comparison_output_dir_final.mkdir(parents=True, exist_ok=True)
        
        # Perform the comparisons
        num_significant_comparisons_found = 0
        
        for id1, id2 in model_pairs:
            path1 = model_eval_paths[id1]
            path2 = model_eval_paths[id2]
            
            for attribute, bin_names_set in all_significant_attributes_union.items():
                for path_name_for_comparison in paths_to_check:
                    for bin_name in sorted(list(bin_names_set)):
                        metrics_dict = compare_hue_distribution_for_bin_across_models(
                            model_results_paths=[str(path1), str(path2)],
                            target_attribute=attribute, target_bin_name=bin_name,
                            output_dir=comparison_output_dir_final,
                            path_name=path_name_for_comparison,
                            model_labels=[id1, id2], num_histogram_bins=50,
                            significance_alpha=COMPARISON_SAVE_ALPHA
                        )
                        if metrics_dict and metrics_dict.get('is_significant', False):
                            num_significant_comparisons_found += 1
                        elif metrics_dict is None:
                            LOGGER.warning(f"Comparison failed for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2}")
        
        # Log summary for this configuration
        LOGGER.info(f"Found {num_significant_comparisons_found} significant comparisons for configuration {run_id}")
        
        # If no significant comparisons found, the directory will be mostly empty
        if num_significant_comparisons_found == 0:
            LOGGER.info(f"No significant comparisons found for {run_i