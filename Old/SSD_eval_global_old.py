# ======================================================
#                 IMPORTS & SETUP
# ======================================================
# --- Standard Library ---
import logging
import sys
import pickle
from pathlib import Path
import itertools
from typing import Dict, Set


# --- Project-Specific Imports (Core Logic) ---
# Configuration & Base Modules (Likely needed by imported functions)
from shimmer_ssd import LOGGER as SSD_LOGGER # Use the project's logger setup

# Training Function (Imported)
from SSD_eval.SSD_Train_old import train_global_workspace

# Evaluation & Analysis Functions (Imported)
from SSD_eval_arbitrary import (
    run_evaluation,
    find_significant_bin_comparisons,
    compare_hue_distribution_for_bin_across_models
)

# --- Setup Logging ---
# Use the imported SSD_LOGGER
LOGGER = SSD_LOGGER # Assign to local variable if preferred, or use SSD_LOGGER directly
if not LOGGER.hasHandlers(): # Avoid adding handlers multiple times if run in interactive env
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)



# --- Constants ---
MODEL_VERSION_NAME = "arbitrary/best_val_loss" # A name for this experiment run
NUM_MODELS_TO_TRAIN = 5 # Train models with seed 0 to 9
BASE_CONFIG_PATH = Path("./config")
BASE_DOMAIN_CHECKPOINT_PATH = Path("./checkpoints") # Location of domain_v.ckpt, domain_attr.ckpt
BASE_OUTPUT_DIR = Path("./test_arbitrary/")
PROJECT_NAME_WANDB = "SSD_eval" # WandB project name
EXCLUDE_COLORS_TRAINING = True # Set based on the domains/config you want to use
USE_WANDB_TRAINING = True # Use WandB for logging training runs?
APPLY_CUSTOM_INIT_TRAINING = True # Apply Kaiming init during training?
DEBUG_MODE = False # Set True for quick test runs (reduces train steps, etc.)


# --- CONFIG & PARAMETERS ---

ENCODERS_N_LAYERS = 1 # Number of layers for encoders
DECODERS_N_LAYERS = 2 # Number of layers for decoders
DECODERS_SIZE = 256 # TO BE APSSER
ENCODERS_SIZE = 128 # TO BE APSSER
TEMPERATURE = 0.945658866085691
ALPHA = 1.5933701965984028
CONTRASTIVE = 0.012151164367796036
CYCLE = 0.12001107692740286
DEMI_CYCLE = .386784952034022
TRADUCTION = 0.10038888929359012
WD = 0.00003943910241216694
LR = 0.00006006431894370282
MAX_LR = 0.0037296884855757342

MAX_TRAINING_STEPS = 243 if DEBUG_MODE else 12150 # Reduce steps drastically for debu




# Evaluation & Comparison Constants
EVALUATION_PARENT_DIR = BASE_OUTPUT_DIR / "evaluation_runs"
COMPARISON_OUTPUT_DIR = BASE_OUTPUT_DIR / "cross_model_comparison"
FILTERING_ALPHA = 0.05 # Alpha for filtering attributes based on run 1
COMPARISON_SAVE_ALPHA = 0.05 # Alpha for saving details of cross-model comparison


# ======================================================
#        TRAINING FUNCTION 
# ======================================================

# ======================================================
#        HELPER & ANALYSIS FUNCTIONS
# ======================================================
# ======================================================
#                     MAIN EXECUTION
# ======================================================

    # trained_model_checkpoints = {f"seed{i}": f"gw_training_and_analysis/training_logs/seed_{i}/GW_Train_seed{i}_colorTrue/checkpoints/last.ckpt" for i in range(0, 7)}

trained = {f"seed{i}": f"/home/alexis/Desktop/test_arbitrary/training_logs/seed_{i}/GW_Train_seed{i}_colorTrue/checkpoints/last.ckpt" for i in range(0, 5)}

if __name__ == "__main__":

    LOGGER.info("===== Starting Multi-Seed Training and Analysis Workflow =====")


    if trained : 
        trained_model_checkpoints = trained
    else : 

        # --- Phase 1: Train Multiple Models ---
        
        trained_model_checkpoints = {} # {run_id: ckpt_path}
        training_output_base = BASE_OUTPUT_DIR / "training_logs"

        LOGGER.info(f"--- Starting Training Phase for {NUM_MODELS_TO_TRAIN} models ---")
        for seed in range(NUM_MODELS_TO_TRAIN):
            LOGGER.info(f"--- Training Model Seed {seed}/{NUM_MODELS_TO_TRAIN-1} ---")
            # Define the output dir for this specific training run's logs/checkpoints
            seed_training_output_dir = training_output_base / f"seed_{seed}"

            # Check if a final checkpoint already exists for this seed to potentially skip
            # This check assumes 'last.ckpt' is created by ModelCheckpoint(save_last='link')
            potential_last_ckpt = seed_training_output_dir / "checkpoints" / "last.ckpt"
            if potential_last_ckpt.exists() and not DEBUG_MODE: # Don't skip in debug mode
                # A more robust check might involve loading the checkpoint or checking modification time
                LOGGER.warning(f"Checkpoint {potential_last_ckpt} already exists. Assuming training for seed {seed} is complete. Skipping training.")
                # We need the *best* checkpoint path, which might be different from 'last'.
                # This simple skip assumes 'last' points to or is the best.
                # A better approach would be to parse the checkpoint dir or log file if needed.
                # For now, we'll assume 'last.ckpt' is sufficient if skipping.
                # **CRITICAL**: This assumes 'last.ckpt' links to the *best* model after validation.
                # If not, this logic needs adjustment, maybe finding the file matching the 'best' criteria.
                run_id_from_skip = f"seed{seed}" # Construct the run ID
                trained_model_checkpoints[run_id_from_skip] = str(potential_last_ckpt) # Store the path
                continue # Skip to the next seed

            best_ckpt_path, run_version_id = train_global_workspace(
                seed=seed,
                config_base_path=BASE_CONFIG_PATH,
                domain_checkpoint_path=BASE_DOMAIN_CHECKPOINT_PATH,
                output_dir=seed_training_output_dir, # Pass specific dir for this seed's training run
                logger_name_prefix="GW_Train",
                project_name=PROJECT_NAME_WANDB,
                exclude_colors=EXCLUDE_COLORS_TRAINING,
                custom_hparams={"temperature": TEMPERATURE, "alpha": ALPHA}, # Example hparams
                use_wandb=USE_WANDB_TRAINING,
                apply_custom_init=APPLY_CUSTOM_INIT_TRAINING,
                max_steps=MAX_TRAINING_STEPS
            )

            if best_ckpt_path:
                run_id = f"seed{seed}" # Consistent ID based on seed
                trained_model_checkpoints[run_id] = best_ckpt_path
                LOGGER.info(f"Training for seed {seed} successful. Best Checkpoint: {best_ckpt_path}")
            else:
                LOGGER.error(f"Training failed for seed {seed}. This model will be excluded.")
    

    # --- Check if enough models trained ---
    if len(trained_model_checkpoints) < 2:
        # print(list(trained_model_checkpoints.keys()))
        LOGGER.error(f"Only {len(list(trained_model_checkpoints.keys()))} models trained successfully. Need at least 2 for comparison. Exiting.")
        exit()

    LOGGER.info(f"\n--- Training Phase Complete. Successfully trained models: {list(trained_model_checkpoints.keys())} ---")


    # --- Phase 2: Run Evaluation on Trained Models ---
    model_eval_results_paths = {} # {run_id: path_to_eval_pkl}
    analysis_results_first_model = None
    successful_eval_runs = []

    LOGGER.info(f"\n--- Starting Evaluation Phase for {len(trained_model_checkpoints)} Trained Models ---")
    # Ensure evaluation runs have unique parent dir
    EVALUATION_PARENT_DIR.mkdir(parents=True, exist_ok=True)

    first_run_id = sorted(trained_model_checkpoints.keys())[0] # ID of the first model (e.g., 'seed0')

    for i, (run_id, ckpt_path) in enumerate(trained_model_checkpoints.items()):
        LOGGER.info(f"--- Evaluating Model {i+1}/{len(trained_model_checkpoints)} (ID: {run_id}, Ckpt: {ckpt_path}) ---")

        # Check if evaluation results already exist for this run_id
        run_specific_dir_name = f"results_{MODEL_VERSION_NAME}_{'sans' if EXCLUDE_COLORS_TRAINING else 'avec'}_couleurs_{run_id}"
        eval_output_dir = EVALUATION_PARENT_DIR / run_specific_dir_name
        potential_eval_pkl = eval_output_dir / "analysis_results.pkl"

        if potential_eval_pkl.exists() and not DEBUG_MODE:
             LOGGER.warning(f"Evaluation results {potential_eval_pkl} already exist for {run_id}. Skipping evaluation.")
             # Load results if it's the first model and we need them for filtering
             if run_id == first_run_id:
                 try:
                     with open(potential_eval_pkl, 'rb') as f:
                         analysis_results_first_model = pickle.load(f)
                     LOGGER.info(f"Loaded existing analysis results for first model {run_id}.")
                 except Exception as e:
                     LOGGER.error(f"Failed to load existing analysis results for {run_id}: {e}. Evaluation needed.")
                     # Force re-evaluation if loading fails
                     # Fall through to run_evaluation
                 else: # If loading succeeded
                     model_eval_results_paths[run_id] = str(potential_eval_pkl)
                     successful_eval_runs.append(run_id)
                     continue # Skip evaluation call
             else: # Not the first model, just record path and skip
                  model_eval_results_paths[run_id] = str(potential_eval_pkl)
                  successful_eval_runs.append(run_id)
                  continue

        # Run evaluation if not skipped
        eval_result = run_evaluation(
            full_attr= not EXCLUDE_COLORS_TRAINING, # Use same setting as training
            run_id=run_id,
            gw_checkpoint_path=ckpt_path,
            model_version=MODEL_VERSION_NAME, # Use experiment name
            output_parent_dir=EVALUATION_PARENT_DIR,
            encoders_n_layers=ENCODERS_N_LAYERS,
            decoders_n_layers=DECODERS_N_LAYERS, # Use same as encoders
            encoders_hidden_dim=ENCODERS_SIZE, # Add encoder size
            decoders_hidden_dim=DECODERS_SIZE, # Add decoder size
            debug_mode=DEBUG_MODE # Pass debug mode
        )

        if eval_result:
            results_path, analysis_results = eval_result
            model_eval_results_paths[run_id] = results_path
            successful_eval_runs.append(run_id)
            if run_id == first_run_id: # Store results dict only for the first model
                analysis_results_first_model = analysis_results
            del analysis_results # Free memory
            LOGGER.info(f"Evaluation successful for {run_id}.")
        else:
            LOGGER.error(f"Evaluation failed for run {run_id}. It will be excluded from comparison.")


    # --- Check if Comparison is Possible ---
    if analysis_results_first_model is None:
        LOGGER.error(f"Evaluation results missing or failed for the first model ({first_run_id}). Cannot perform significance filtering. Exiting.")
        exit()
    if len(successful_eval_runs) < 2:
        LOGGER.error(f"Fewer than two models evaluated successfully ({len(successful_eval_runs)}). Cannot perform pairwise comparison. Exiting.")
        exit()

    LOGGER.info(f"\nSuccessfully evaluated models: {successful_eval_runs}")


    # --- Phase 3: Filter Attributes based on FIRST Model ---
    paths_to_check = ['translated', 'half_cycle', 'full_cycle']
    all_significant_attributes_union: Dict[str, Set[str]] = {}

    LOGGER.info(f"\n--- Filtering Attributes Based on FIRST Model ({first_run_id}) Significance (p < {FILTERING_ALPHA}) ---")
    # ... (Filtering logic using find_significant_bin_comparisons - same as before) ...
    for path_name in paths_to_check:
        LOGGER.info(f"--- Checking Path: {path_name} ---")
        significant_attributes_current_path = find_significant_bin_comparisons(
            analysis_results=analysis_results_first_model, path_name=path_name,
            alpha=FILTERING_ALPHA
        )
        for attr, bins_set in significant_attributes_current_path.items():
            all_significant_attributes_union.setdefault(attr, set()).update(bins_set)

    LOGGER.info("\n--- Significance Filtering Summary ---")
    # ... (Logging summary - same as before) ...
    if not all_significant_attributes_union:
         LOGGER.info(f"No attributes met the filtering criteria (p < {FILTERING_ALPHA}) in ANY path of the first model.")


    # --- Phase 4: Pairwise Cross-Model Comparison (Conditionally Saving) ---
    if not all_significant_attributes_union:
        LOGGER.info("Skipping pairwise cross-model comparison.")
    else:
        LOGGER.info(f"\n--- Starting Pairwise Cross-Model Comparison for {len(successful_eval_runs)} Models ---")
        LOGGER.info(f"--- Details will be saved ONLY if comparison KS p-value < {COMPARISON_SAVE_ALPHA} ---")
        # Update comparison output dir name
        COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        comparison_output_dir_final = COMPARISON_OUTPUT_DIR / f"{MODEL_VERSION_NAME}_N{len(successful_eval_runs)}_filt{FILTERING_ALPHA}_save{COMPARISON_SAVE_ALPHA}"

        model_pairs = list(itertools.combinations(successful_eval_runs, 2))
        LOGGER.info(f"Performing {len(model_pairs)} pairwise comparisons.")
        num_significant_comparisons_found = 0

        # Iterate through pairs, attributes, paths, bins
        for id1, id2 in model_pairs:
            LOGGER.debug(f"\n--- Comparing Pair: {id1} vs {id2} ---")
            path1 = model_eval_results_paths[id1]
            path2 = model_eval_results_paths[id2]
            # ... (Nested loops for attribute, path_name_for_comparison, bin_name - same as before) ...
            for attribute, bin_names_set in all_significant_attributes_union.items():
                for path_name_for_comparison in paths_to_check:
                     for bin_name in sorted(list(bin_names_set)):
                          metrics_dict = compare_hue_distribution_for_bin_across_models(
                               model_results_paths=[path1, path2],
                               target_attribute=attribute, target_bin_name=bin_name,
                               output_dir=comparison_output_dir_final, # Pass final dir
                               path_name=path_name_for_comparison,
                               model_labels=[id1, id2], num_histogram_bins=50,
                               significance_alpha=COMPARISON_SAVE_ALPHA # Alpha for saving
                          )
                          if metrics_dict and metrics_dict.get('is_significant', False):
                              num_significant_comparisons_found += 1
                          elif metrics_dict is None:
                               LOGGER.warning(f"      Comparison failed for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2}.")

        # ... (Final summary logging and optional directory removal - same as before) ...
        LOGGER.info("\nPairwise cross-model comparisons finished.")

    LOGGER.info("\n===== Full Workflow Completed =====")