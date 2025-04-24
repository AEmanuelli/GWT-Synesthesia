# imports 
import logging
import sys
import pickle
from pathlib import Path
import itertools
from typing import Dict, Set

from SSD_Train import train_global_workspace
from SSD_eval_arbitrary import run_evaluation, find_significant_bin_comparisons, compare_hue_distribution_for_bin_across_models


from shimmer_ssd import LOGGER as SSD_LOGGER
from shimmer_ssd.config import load_config

import logging
import sys
import pickle
import itertools
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Any

def run_ssd_multiseed_workflow(
    num_models_to_train=5,
    model_version_name="idiosyncratic/to_be_named",
    base_output_dir=Path("./test_arbitrary/"),
    exclude_colors_training=True,
    use_wandb_training=True,
    apply_custom_init_training=True,
    debug_mode=False,
    filtering_alpha=0.05,
    comparison_save_alpha=0.05,
    trained_model_checkpoints=None,
    config=None,
    custom_hparams=None,
    project_name="SSD_eval"
):
    """
    Run a complete workflow for training and evaluating multiple SSD models with different seeds,
    then perform cross-model comparison on significant attributes.

    Args:
        num_models_to_train (int): Number of models to train with different seeds
        model_version_name (str): Name for this experiment run
        base_output_dir (Path): Base directory for all output files
        exclude_colors_training (bool): Whether to exclude colors during training
        use_wandb_training (bool): Whether to use WandB for logging
        apply_custom_init_training (bool): Whether to apply Kaiming init during training
        debug_mode (bool): If True, reduces training steps for quick testing
        filtering_alpha (float): Alpha threshold for filtering attributes based on first model
        comparison_save_alpha (float): Alpha threshold for saving cross-model comparison details
        trained_model_checkpoints (dict): Optional dict of {run_id: ckpt_path} if models already trained
        config: Configuration object for training
        custom_hparams: Custom hyperparameters for training
        project_name (str): WandB project name

    Returns:
        dict: Results containing trained model paths, evaluation results, and comparison outcomes
    """
    # Set up directories and logger
    logger = setup_logger()
    directories = setup_directories(base_output_dir)
    
    # Initialize workflow results
    workflow_results = initialize_workflow_results()
    
    logger.info("===== Starting Multi-Seed Training and Analysis Workflow =====")

    # Phase 1: Train multiple models with different seeds
    workflow_results = run_training_phase(
        workflow_results, 
        trained_model_checkpoints, 
        num_models_to_train, 
        directories["training"], 
        config,
        custom_hparams,
        exclude_colors_training,
        apply_custom_init_training,
        project_name,
        debug_mode,
        logger
    )

    # Check if enough models are trained
    if len(workflow_results["trained_models"]) < 2:
        logger.error(f"Only {len(list(workflow_results['trained_models'].keys()))} models trained successfully. Need at least 2 for comparison. Exiting.")
        return workflow_results

    logger.info(f"\n--- Training Phase Complete. Successfully trained models: {list(workflow_results['trained_models'].keys())} ---")

    # Phase 2: Evaluate trained models
    workflow_results, first_model_results = run_evaluation_phase(
        workflow_results, 
        model_version_name,
        directories["evaluation"],
        exclude_colors_training,
        config,
        debug_mode,
        logger
    )

    # Check if comparison is possible
    if not can_perform_comparison(first_model_results, workflow_results["successful_evaluations"], logger):
        return workflow_results

    # Phase 3: Filter attributes based on first model
    significant_attributes = filter_significant_attributes(
        first_model_results, 
        filtering_alpha,
        logger
    )
    workflow_results["significant_attributes"] = significant_attributes

    # Phase 4: Run pairwise cross-model comparison
    workflow_results = run_cross_model_comparison(
        workflow_results,
        significant_attributes,
        directories["comparison"],
        model_version_name,
        comparison_save_alpha,
        logger
    )

    logger.info("\n===== Full Workflow Completed =====")
    return workflow_results

def setup_logger():
    """Setup and configure logger."""
    logger = SSD_LOGGER if 'SSD_LOGGER' in globals() else logging.getLogger("SSD_Workflow")
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def setup_directories(base_output_dir: Path) -> Dict[str, Path]:
    """Create and return directory structure for outputs."""
    return {
        "training": base_output_dir / "training_logs",
        "evaluation": base_output_dir / "evaluation_runs",
        "comparison": base_output_dir / "cross_model_comparison"
    }


def initialize_workflow_results() -> Dict[str, Any]:
    """Initialize the workflow results dictionary."""
    return {
        "trained_models": {},
        "evaluation_results": {},
        "successful_evaluations": [],
        "significant_attributes": {},
        "comparison_results": {}
    }


def run_training_phase(
    workflow_results: Dict,
    trained_model_checkpoints: Dict,
    num_models_to_train: int,
    training_output_base: Path,
    config,
    custom_hparams,
    exclude_colors_training: bool,
    apply_custom_init_training: bool,
    project_name: str,
    debug_mode: bool,
    logger
) -> Dict:
    """Run the training phase for multiple models with different seeds."""
    # Skip training if checkpoints already provided
    if trained_model_checkpoints:
        workflow_results["trained_models"] = trained_model_checkpoints
        return workflow_results
    
    trained_model_checkpoints = {}
    logger.info(f"--- Starting Training Phase for {num_models_to_train} models ---")
    
    for seed in range(num_models_to_train):
        config.seed = seed
        logger.info(f"--- Training Model Seed {seed}/{num_models_to_train-1} ---")
        
        # Define output directory for this seed
        seed_training_output_dir = training_output_base / f"seed_{seed}"
        
        # Check if checkpoint already exists
        model_checkpoint = check_existing_checkpoint(
            seed_training_output_dir, 
            seed, 
            debug_mode, 
            logger
        )
        
        if model_checkpoint:
            trained_model_checkpoints[f"seed{seed}"] = model_checkpoint
            continue
        
        # Train new model
        experiment_name = f"GW_Train_seed{seed}_color{not exclude_colors_training}"
        model, best_ckpt_path = train_global_workspace(
            config,
            custom_hparams,
            project_name,
            experiment_name,
            apply_custom_init_training,
            exclude_colors_training
        )
        
        if model:
            trained_model_checkpoints[f"seed{seed}"] = model
            logger.info(f"Training for seed {seed} successful. Best Checkpoint: {best_ckpt_path}")
        else:
            logger.error(f"Training failed for seed {seed}. This model will be excluded.")
    
    workflow_results["trained_models"] = trained_model_checkpoints
    return workflow_results


def check_existing_checkpoint(
    seed_training_output_dir: Path, 
    seed: int, 
    debug_mode: bool,
    logger
) -> Optional[str]:
    """Check if checkpoint already exists for the given seed."""
    potential_last_ckpt = seed_training_output_dir / "checkpoints" / "last.ckpt"
    if potential_last_ckpt.exists() and not debug_mode:
        logger.warning(f"Checkpoint {potential_last_ckpt} already exists. Assuming training for seed {seed} is complete. Skipping training.")
        return str(potential_last_ckpt)
    return None


def run_evaluation_phase(
    workflow_results: Dict, 
    model_version_name: str,
    evaluation_parent_dir: Path,
    exclude_colors_training: bool,
    config,
    debug_mode: bool,
    logger
) -> Tuple[Dict, Any]:
    """Run evaluation on all trained models."""
    model_eval_results_paths = {}
    analysis_results_first_model = None
    successful_eval_runs = []
    trained_model_checkpoints = workflow_results["trained_models"]

    logger.info(f"\n--- Starting Evaluation Phase for {len(trained_model_checkpoints)} Trained Models ---")
    evaluation_parent_dir.mkdir(parents=True, exist_ok=True)

    first_run_id = sorted(trained_model_checkpoints.keys())[0]

    for i, (run_id, gw) in enumerate(trained_model_checkpoints.items()):
        logger.info(f"--- Evaluating Model {i+1}/{len(trained_model_checkpoints)} (ID: {run_id}) ---")
        
        # Define evaluation directory
        color_suffix = 'sans' if exclude_colors_training else 'avec'
        run_specific_dir_name = f"results_{model_version_name}_{color_suffix}_couleurs_{run_id}"
        eval_output_dir = evaluation_parent_dir / run_specific_dir_name
        
        # Check for existing evaluation results
        eval_result = process_model_evaluation(
            run_id,
            eval_output_dir,
            gw,
            model_version_name,
            evaluation_parent_dir,
            config,
            exclude_colors_training,
            debug_mode,
            first_run_id,
            logger
        )
        
        if eval_result:
            results_path, analysis_results = eval_result
            model_eval_results_paths[run_id] = results_path
            successful_eval_runs.append(run_id)
            if run_id == first_run_id:
                analysis_results_first_model = analysis_results
            del analysis_results  # Free memory
            logger.info(f"Evaluation successful for {run_id}.")
        else:
            logger.error(f"Evaluation failed for run {run_id}. It will be excluded from comparison.")

    workflow_results["evaluation_results"] = model_eval_results_paths
    workflow_results["successful_evaluations"] = successful_eval_runs
    
    return workflow_results, analysis_results_first_model


def process_model_evaluation(
    run_id: str,
    eval_output_dir: Path,
    ckpt_path: str,
    model_version_name: str,
    evaluation_parent_dir: Path,
    config,
    exclude_colors_training: bool,
    debug_mode: bool,
    first_run_id: str,
    logger
) -> Optional[Tuple[str, Any]]:
    """Process evaluation for a single model, checking for existing results."""
    potential_eval_pkl = eval_output_dir / "analysis_results.pkl"
    
    # Check if evaluation results already exist
    if potential_eval_pkl.exists() and not debug_mode:
        logger.warning(f"Evaluation results {potential_eval_pkl} already exist for {run_id}. Skipping evaluation.")
        
        # Load results for first model if needed for filtering
        if run_id == first_run_id:
            try:
                with open(potential_eval_pkl, 'rb') as f:
                    analysis_results = pickle.load(f)
                logger.info(f"Loaded existing analysis results for first model {run_id}.")
                return str(potential_eval_pkl), analysis_results
            except Exception as e:
                logger.error(f"Failed to load existing analysis results for {run_id}: {e}. Evaluation needed.")
                # Fall through to run_evaluation
        else:
            # Not the first model, just record path and skip
            return str(potential_eval_pkl), None
    
    # Run evaluation
    return run_evaluation(
        full_attr=not exclude_colors_training,
        run_id=run_id,
        gw_checkpoint_path=ckpt_path,
        model_version=model_version_name,
        output_parent_dir=evaluation_parent_dir,
        encoders_n_layers=config.global_workspace.encoders.n_layers,
        decoders_n_layers=config.global_workspace.decoders.n_layers,
        encoders_hidden_dim=config.global_workspace.encoders.hidden_dim,
        decoders_hidden_dim=config.global_workspace.decoders.hidden_dim,
        debug_mode=debug_mode
    )



def can_perform_comparison(
    analysis_results_first_model,
    successful_eval_runs: List[str],
    logger
) -> bool:
    """Check if comparison phase can be performed."""
    if analysis_results_first_model is None:
        logger.error("Evaluation results missing or failed for the first model. Cannot perform significance filtering. Exiting.")
        return False
    
    if len(successful_eval_runs) < 2:
        logger.error(f"Fewer than two models evaluated successfully ({len(successful_eval_runs)}). Cannot perform pairwise comparison. Exiting.")
        return False
    
    logger.info(f"\nSuccessfully evaluated models: {successful_eval_runs}")
    return True


def filter_significant_attributes(
    analysis_results_first_model,
    filtering_alpha: float,
    logger
) -> Dict[str, Set[str]]:
    """Filter attributes based on significance in the first model."""
    paths_to_check = ['translated', 'half_cycle', 'full_cycle']
    all_significant_attributes_union: Dict[str, Set[str]] = {}

    logger.info(f"\n--- Filtering Attributes Based on FIRST Model Significance (p < {filtering_alpha}) ---")
    
    for path_name in paths_to_check:
        logger.info(f"--- Checking Path: {path_name} ---")
        
        # Assuming find_significant_bin_comparisons is defined elsewhere
        significant_attributes_current_path = find_significant_bin_comparisons(
            analysis_results=analysis_results_first_model, 
            path_name=path_name,
            alpha=filtering_alpha
        )
        
        for attr, bins_set in significant_attributes_current_path.items():
            all_significant_attributes_union.setdefault(attr, set()).update(bins_set)

    # Log results
    logger.info("\n--- Significance Filtering Summary ---")
    if not all_significant_attributes_union:
        logger.info(f"No attributes met the filtering criteria (p < {filtering_alpha}) in ANY path of the first model.")
    
    return all_significant_attributes_union


def run_cross_model_comparison(
    workflow_results: Dict,
    significant_attributes: Dict[str, Set[str]],
    comparison_output_dir: Path,
    model_version_name: str,
    comparison_save_alpha: float,
    logger
) -> Dict:
    """Run pairwise cross-model comparison for significant attributes."""
    # Skip comparison if no significant attributes found
    if not significant_attributes:
        logger.info("Skipping pairwise cross-model comparison.")
        return workflow_results
    
    successful_eval_runs = workflow_results["successful_evaluations"]
    model_eval_results_paths = workflow_results["evaluation_results"]
    paths_to_check = ['translated', 'half_cycle', 'full_cycle']
    
    logger.info(f"\n--- Starting Pairwise Cross-Model Comparison for {len(successful_eval_runs)} Models ---")
    logger.info(f"--- Details will be saved ONLY if comparison KS p-value < {comparison_save_alpha} ---")
    
    # Set up directory
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir_final = comparison_output_dir / f"{model_version_name}_N{len(successful_eval_runs)}"#_filt{workflow_results.get('filtering_alpha', 0.05)}_save{comparison_save_alpha}"
    
    # Generate all model pairs
    model_pairs = list(itertools.combinations(successful_eval_runs, 2))
    logger.info(f"Performing {len(model_pairs)} pairwise comparisons.")
    
    # Run comparisons
    comparison_results = compare_all_model_pairs(
        model_pairs,
        model_eval_results_paths,
        significant_attributes,
        paths_to_check,
        comparison_dir_final,
        comparison_save_alpha,
        logger
    )
    
    workflow_results["comparison_results"] = comparison_results
    return workflow_results


def compare_all_model_pairs(
    model_pairs: List[Tuple[str, str]],
    model_eval_results_paths: Dict[str, str],
    significant_attributes: Dict[str, Set[str]],
    paths_to_check: List[str],
    comparison_output_dir: Path,
    comparison_save_alpha: float,
    logger
) -> Dict:
    """Compare all pairs of models for significant attributes and provide global summary."""
    num_significant_comparisons_found = 0
    significant_comparisons = {}
    
    # Create global summary dictionary to track results across all comparisons
    global_summary = {
        "total_comparisons": 0,
        "significant_comparisons": 0,
        "model_pair_stats": {},
        "attribute_stats": {},
        "bin_stats": {},
        "path_stats": {}
    }
    
    for id1, id2 in model_pairs:
        logger.debug(f"\n--- Comparing Pair: {id1} vs {id2} ---")
        path1 = model_eval_results_paths[id1]
        path2 = model_eval_results_paths[id2]
        pair_key = f"{id1}_vs_{id2}"
        significant_comparisons[pair_key] = []
        
        # Initialize pair stats in global summary
        global_summary["model_pair_stats"][pair_key] = {
            "total": 0,
            "significant": 0
        }
        
        for attribute, bin_names_set in significant_attributes.items():
            # Initialize attribute stats if not present
            if attribute not in global_summary["attribute_stats"]:
                global_summary["attribute_stats"][attribute] = {
                    "total": 0,
                    "significant": 0
                }
                
            for path_name in paths_to_check:
                # Initialize path stats if not present
                if path_name not in global_summary["path_stats"]:
                    global_summary["path_stats"][path_name] = {
                        "total": 0,
                        "significant": 0
                    }
                
                for bin_name in sorted(list(bin_names_set)):
                    # Initialize bin stats if not present
                    if bin_name not in global_summary["bin_stats"]:
                        global_summary["bin_stats"][bin_name] = {
                            "total": 0,
                            "significant": 0
                        }
                    
                    # Track total comparisons
                    global_summary["total_comparisons"] += 1
                    global_summary["model_pair_stats"][pair_key]["total"] += 1
                    global_summary["attribute_stats"][attribute]["total"] += 1
                    global_summary["path_stats"][path_name]["total"] += 1
                    global_summary["bin_stats"][bin_name]["total"] += 1
                    
                    comparison_result = compare_single_bin_across_models(
                        path1, path2,
                        attribute, bin_name,
                        path_name,
                        id1, id2,
                        comparison_output_dir,
                        comparison_save_alpha,
                        logger
                    )
                    
                    if comparison_result and comparison_result.get('is_significant', False):
                        num_significant_comparisons_found += 1
                        significant_comparisons[pair_key].append({
                            'attribute': attribute,
                            'bin': bin_name,
                            'path': path_name,
                            'metrics': comparison_result
                        })
                        
                        # Update significant counts in global summary
                        global_summary["significant_comparisons"] += 1
                        global_summary["model_pair_stats"][pair_key]["significant"] += 1
                        global_summary["attribute_stats"][attribute]["significant"] += 1
                        global_summary["path_stats"][path_name]["significant"] += 1
                        global_summary["bin_stats"][bin_name]["significant"] += 1
    

    global_summary_file_path = comparison_output_dir / "all_comparison_results.json"
    import json
    with open(global_summary_file_path, 'w') as f:
        json.dump(global_summary, f, indent=2)

    # Log final summary
    logger.info(f"\nFound {num_significant_comparisons_found} significant cross-model differences (p < {comparison_save_alpha}).")
    if num_significant_comparisons_found == 0:
        logger.info(f"No significant differences found across models. Output directory: {comparison_output_dir}")
    
    # Add global summary to the return value
    return {
        "num_significant": num_significant_comparisons_found,
        "details": significant_comparisons,
        "global_summary": global_summary
    }


def compare_single_bin_across_models(
    path1: str, 
    path2: str,
    attribute: str, 
    bin_name: str,
    path_name: str,
    id1: str, 
    id2: str,
    comparison_output_dir: Path,
    comparison_save_alpha: float,
    logger
) -> Optional[Dict]:
    """Compare a single bin between two models."""
    # Assuming compare_hue_distribution_for_bin_across_models is defined elsewhere
    metrics_dict = compare_hue_distribution_for_bin_across_models(
        model_results_paths=[path1, path2],
        target_attribute=attribute, 
        target_bin_name=bin_name,
        output_dir=comparison_output_dir,
        path_name=path_name,
        model_labels=[id1, id2], 
        num_histogram_bins=50,
        significance_alpha=comparison_save_alpha
    )
    
    if metrics_dict is None:
        logger.warning(f"Comparison failed for {attribute}/{bin_name}/{path_name} between {id1} and {id2}.")
    
    return metrics_dict


# Usage example:
if __name__ == "__main__":
    config = load_config("./config", use_cli=False, load_files=["base_params.yaml"])
    config.training.max_steps = 243
    custom_hparams = {
        "temperature": 1.0,
        "alpha": 1.0
    }

    trained_model_checkpoints = None#{f"seed{i}": f"/home/alexis/Desktop/test_arbitrary/training_logs/seed_{i}/GW_Train_seed{i}_colorTrue/checkpoints/last.ckpt" for i in range(0, 5)}

    results = run_ssd_multiseed_workflow(
        num_models_to_train=3,
        model_version_name="base_params_DEBUG",
        base_output_dir=Path(f"./test_global_debugging/num_steps_{config.training.max_steps}"),
        exclude_colors_training=True,
        apply_custom_init_training=True,
        debug_mode=False, 
        config=config,
        custom_hparams=custom_hparams,
        project_name="DEBUG",
        trained_model_checkpoints=trained_model_checkpoints
    )



# TODO 
# 0. Be sure they are logged separately : done
# 1. Understand the scheduler : done
# 2. Make a big run with low temperature : done
# 3. 
