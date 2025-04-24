import logging
import sys
import itertools
import pickle
from pathlib import Path
from typing import Dict, Set, List, Optional, Any # Added typing for clarity

from SSD_Train import train_global_workspace
from SSD_eval_arbitrary import run_evaluation
from shimmer_ssd import LOGGER as SSD_LOGGER
from shimmer_ssd.config import load_config  

# Set up logger (consider moving this outside the function if called multiple times)
LOGGER = logging.getLogger("SSDWorkflow") # Give it a specific name
if not LOGGER.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def format_results_summary(results: Dict[str, Any]) -> str:
    """Formats the workflow results into a readable string summary."""
    lines = ["\n===== Workflow Execution Summary ====="]

    # --- Config ---
    lines.append("\n--- Configuration ---")
    if "config_summary" in results:
        for key, val in results["config_summary"].items():
            lines.append(f"  {key}: {val}")

    # --- Outputs ---
    lines.append("\n--- Output Locations ---")
    if "output_locations" in results:
        for key, val in results["output_locations"].items():
            lines.append(f"  {key}: {val}")

    # --- Training ---
    lines.append("\n--- Training Summary ---")
    if "training_summary" in results:
        ts = results["training_summary"]
        lines.append(f"  Models Requested: {ts.get('requested', 'N/A')}")
        lines.append(f"  Models Attempted: {ts.get('attempted', 'N/A')}")
        lines.append(f"  Models Successfully Trained: {ts.get('successful', 'N/A')}")
        if ts.get("failed_seeds"):
            lines.append(f"  Failed Seeds: {', '.join(map(str, ts['failed_seeds']))}")
        elif ts.get('successful', 0) < ts.get('attempted', 0):
             lines.append("  Some training runs failed (details in logs).") # Fallback message
        lines.append(f"  Checkpoint paths available in results['training_details']['checkpoints']")

    # --- Evaluation ---
    lines.append("\n--- Evaluation Summary ---")
    if "evaluation_summary" in results:
        es = results["evaluation_summary"]
        lines.append(f"  Models Attempted Evaluation: {es.get('attempted', 'N/A')}")
        lines.append(f"  Models Successfully Evaluated: {es.get('successful', 'N/A')}")
        if es.get("failed_runs"):
            lines.append(f"  Failed Evaluation Runs: {', '.join(es['failed_runs'])}")
        elif es.get('successful', 0) < es.get('attempted', 0):
             lines.append("  Some evaluations failed (details in logs).") # Fallback message
        lines.append(f"  Result paths available in results['evaluation_details']['results_paths']")

    # --- Filtering ---
    lines.append("\n--- Attribute Filtering Summary ---")
    if "filtering_summary" in results:
        fs = results["filtering_summary"]
        if fs.get("performed"):
            lines.append(f"  Filtering Performed: Yes (Based on model: {fs.get('based_on_model', 'N/A')})")
            lines.append(f"  Significance Threshold (alpha): {fs.get('alpha', 'N/A')}")
            lines.append(f"  Significant Attributes Found: {fs.get('num_significant_attributes', 'N/A')}")
            lines.append(f"  Total Significant Attribute-Bins: {fs.get('total_significant_attribute_bins', 'N/A')}")
            lines.append(f"  Details in results['filtering_details']['significant_attributes']")
        else:
            lines.append("  Filtering Performed: No (Insufficient successful evaluations or no significant attributes found in first model).")


    # --- Comparison ---
    lines.append("\n--- Cross-Model Comparison Summary ---")
    if "comparison_summary" in results:
        cs = results["comparison_summary"]
        if cs.get("performed"):
            lines.append(f"  Comparison Performed: Yes")
            lines.append(f"  Models Compared: {cs.get('num_models_compared', 'N/A')}")
            lines.append(f"  Pairs Compared: {cs.get('num_pairs_compared', 'N/A')}")
            lines.append(f"  Significance Threshold (alpha for saving): {cs.get('save_alpha', 'N/A')}")
            lines.append(f"  Significant Differences Found (p < alpha): {cs.get('num_significant_differences_found', 'N/A')}")
            lines.append(f"  Details in results['comparison_details']['significant_comparison_details']")
            lines.append(f"  Comparison plots/data saved under: {results.get('output_locations', {}).get('comparison_output_dir', 'N/A')}")
        else:
            lines.append("  Comparison Performed: No (Insufficient successful evaluations or no attributes passed filtering).")

    lines.append("\n===== End of Summary =====")
    return "\n".join(lines)


def run_ssd_multiseed_workflow(
    num_models_to_train: int = 2,
    model_version_name: str = "idiosyncratic/to_be_named",
    base_output_dir: Path = Path("./test_arbitrary/"),
    exclude_colors_training: bool = True,
    use_wandb_training: bool = True, # Note: use_wandb_training is passed but not used in provided snippet
    apply_custom_init_training: bool = True,
    debug_mode: bool = False,
    filtering_alpha: float = 0.05,
    comparison_save_alpha: float = 0.05,
    trained_model_checkpoints: Optional[Dict[str, str]] = None,
    config: Optional[Any] = None, # Replace Any with actual config type if known
    custom_hparams: Optional[Dict[str, Any]] = None,
    project_name: str = "SSD_eval"
) -> Dict[str, Any]:
    """
    Run a complete workflow for training and evaluating multiple SSD models...
    (Keep original docstring)

    Returns:
        dict: A structured dictionary containing configuration, summaries,
              output locations, and detailed results for each phase.
    """
    # Logger setup done globally or outside this function

    # --- Setup ---
    LOGGER.info("===== Initializing Multi-Seed Training and Analysis Workflow =====")
    if config is None:
        LOGGER.error("Configuration object 'config' is required.")
        # Or load a default config here
        return {"error": "Missing configuration"}

    # --- Directory Setup ---
    training_output_base = base_output_dir / "training_logs" # Define earlier
    evaluation_parent_dir = base_output_dir / "evaluation_runs"
    comparison_output_dir = base_output_dir / "cross_model_comparison"
    # Final comparison dir name is determined later
    comparison_output_dir_final_placeholder = comparison_output_dir / f"{model_version_name}_comparison_results"

    # --- Initialize Results Dictionary ---
    # Use a more structured approach from the start
    workflow_results = {
        "config_summary": {
            "num_models_requested": num_models_to_train,
            "model_version_name": model_version_name,
            "exclude_colors_training": exclude_colors_training,
            "apply_custom_init_training": apply_custom_init_training,
            "debug_mode": debug_mode,
            "filtering_alpha": filtering_alpha,
            "comparison_save_alpha": comparison_save_alpha,
            "project_name": project_name,
            # Add other key config items if needed, e.g., from 'config' object
            "config_encoders_layers": config.global_workspace.encoders.n_layers,
            "config_decoders_layers": config.global_workspace.decoders.n_layers,
            "config_encoders_dim": config.global_workspace.encoders.hidden_dim,
            "config_decoders_dim": config.global_workspace.decoders.hidden_dim,
        },
        "output_locations": {
            "base_output_dir": str(base_output_dir),
            "training_log_base": str(training_output_base),
            "evaluation_parent_dir": str(evaluation_parent_dir),
            "comparison_output_dir": str(comparison_output_dir_final_placeholder) # Placeholder, updated later
        },
        "training_summary": {
            "requested": num_models_to_train,
            "attempted": 0,
            "successful": 0,
            "failed_seeds": [],
        },
        "training_details": {
             "checkpoints": {}, # Stores {run_id: ckpt_path}
        },
        "evaluation_summary": {
            "attempted": 0,
            "successful": 0,
            "failed_runs": [],
        },
         "evaluation_details": {
            "results_paths": {}, # Stores {run_id: results_path}
            "first_model_analysis_raw": None # Store the actual data if needed, else remove
        },
        "filtering_summary": {
            "performed": False,
            "based_on_model": None,
            "alpha": filtering_alpha,
            "num_significant_attributes": 0,
            "total_significant_attribute_bins": 0,
        },
        "filtering_details": {
            "significant_attributes": {}, # Stores {attr: {bin_set}}
        },
        "comparison_summary": {
            "performed": False,
            "num_models_compared": 0,
            "num_pairs_compared": 0,
            "save_alpha": comparison_save_alpha,
            "num_significant_differences_found": 0,
        },
        "comparison_details": {
            "significant_comparisons": {} # Stores {pair_key: [{...}]}
        }
    }

    # --- Phase 1: Train Multiple Models ---
    current_trained_checkpoints = {} # Use a temporary dict
    if trained_model_checkpoints:
        LOGGER.info("--- Using Pre-trained Model Checkpoints ---")
        current_trained_checkpoints = trained_model_checkpoints
        # Assume pre-trained count matches requested for summary, or adjust logic
        workflow_results["training_summary"]["attempted"] = len(trained_model_checkpoints)
        workflow_results["training_summary"]["successful"] = len(trained_model_checkpoints)

    else:
        LOGGER.info(f"--- Starting Training Phase for {num_models_to_train} models ---")
        seeds_to_train = list(range(num_models_to_train))
        workflow_results["training_summary"]["attempted"] = len(seeds_to_train)

        for seed in seeds_to_train:
            config.seed = seed # Ensure seed is set in config

            LOGGER.info(f"--- Training Model Seed {seed}/{num_models_to_train-1} ---")
            seed_training_output_dir = training_output_base / f"seed_{seed}"
            run_id = f"seed{seed}" # Define run_id consistently

            # Check if checkpoint already exists
            potential_last_ckpt = seed_training_output_dir / "checkpoints" / "last.ckpt" # Example path
            # NOTE: Your original code checks `best_ckpt_path`, this uses `last.ckpt` based on the check. Adjust if needed.
            # ALSO: Original code uses `run_version_id` from training, here we simplify to `seedX`. Adjust if run_id needs wandb ID.

            if potential_last_ckpt.exists() and not debug_mode:
                LOGGER.warning(f"Checkpoint {potential_last_ckpt} already exists. Assuming training for seed {seed} is complete. Skipping training.")
                current_trained_checkpoints[run_id] = str(potential_last_ckpt)
                continue # Skip to next seed

            # Define experiment name for this training run
            experiment_name = f"GW_Train_seed{seed}_color{not exclude_colors_training}"

            try:
                # Train the model
                # Assuming train_global_workspace returns (best_ckpt_path, run_version_id_or_similar)
                best_ckpt_path, _ = train_global_workspace( # Discard run_version_id if run_id is seed-based
                    config,
                    custom_hparams=custom_hparams,
                    project_name=project_name,
                    experiment_name=experiment_name,
                    apply_custom_init=apply_custom_init_training,
                    exclude_colors=exclude_colors_training,
                    # Pass use_wandb directly if needed by train_global_workspace
                    )

                if best_ckpt_path and Path(best_ckpt_path).exists(): # Check path validity
                    current_trained_checkpoints[run_id] = best_ckpt_path
                    LOGGER.info(f"Training for seed {seed} successful. Best Checkpoint: {best_ckpt_path}")
                else:
                    LOGGER.error(f"Training failed or returned invalid path for seed {seed}. This model will be excluded.")
                    workflow_results["training_summary"]["failed_seeds"].append(seed)

            except Exception as e:
                 LOGGER.error(f"Exception during training for seed {seed}: {e}", exc_info=debug_mode)
                 workflow_results["training_summary"]["failed_seeds"].append(seed)

        # Update summary after loop
        workflow_results["training_summary"]["successful"] = len(current_trained_checkpoints)

    # Populate training details
    workflow_results["training_details"]["checkpoints"] = current_trained_checkpoints

    # --- Check if enough models trained ---
    if workflow_results["training_summary"]["successful"] < 2:
        LOGGER.error(f"Only {workflow_results['training_summary']['successful']} models available. Need at least 2 for comparison. Exiting early.")
        # Log final summary before exiting
        LOGGER.info(format_results_summary(workflow_results))
        return workflow_results

    LOGGER.info(f"\n--- Training Phase Complete. Successfully trained models: {list(current_trained_checkpoints.keys())} ---")

    # --- Phase 2: Run Evaluation on Trained Models ---
    model_eval_results_paths = {}
    analysis_results_first_model = None
    successful_eval_runs = []
    failed_eval_runs = []

    LOGGER.info(f"\n--- Starting Evaluation Phase for {len(current_trained_checkpoints)} Trained Models ---")
    evaluation_parent_dir.mkdir(parents=True, exist_ok=True) # Ensure exists

    # Determine the first model ID based on sorted keys (e.g., 'seed0', 'seed1', ...)
    available_run_ids = sorted(current_trained_checkpoints.keys())
    first_run_id = available_run_ids[0]
    workflow_results["evaluation_summary"]["attempted"] = len(available_run_ids)

    for i, run_id in enumerate(available_run_ids):
        ckpt_path = current_trained_checkpoints[run_id]
        LOGGER.info(f"--- Evaluating Model {i+1}/{len(available_run_ids)} (ID: {run_id}, Ckpt: {ckpt_path}) ---")

        # Define output dir based on run_id
        run_specific_dir_name = f"results_{model_version_name}_{'sans' if exclude_colors_training else 'avec'}_couleurs_{run_id}"
        eval_output_dir = evaluation_parent_dir / run_specific_dir_name
        potential_eval_pkl = eval_output_dir / "analysis_results.pkl" # Standardized name

        analysis_results_current = None # To hold loaded or generated results

        # Check if evaluation results already exist
        if potential_eval_pkl.exists() and not debug_mode:
            LOGGER.warning(f"Evaluation results {potential_eval_pkl} already exist for {run_id}. Attempting to load.")
            try:
                with open(potential_eval_pkl, 'rb') as f:
                    analysis_results_current = pickle.load(f)
                LOGGER.info(f"Successfully loaded existing analysis results for {run_id}.")
                model_eval_results_paths[run_id] = str(potential_eval_pkl)
                successful_eval_runs.append(run_id)
                if run_id == first_run_id:
                    analysis_results_first_model = analysis_results_current
                continue # Skip to next run_id
            except Exception as e:
                LOGGER.error(f"Failed to load existing analysis results for {run_id} from {potential_eval_pkl}: {e}. Re-running evaluation.")
                # Proceed to run evaluation below

        # Run evaluation if not loaded/skipped
        try:
            # Assuming run_evaluation returns (results_pkl_path, analysis_results_dict) or None
            eval_result = run_evaluation(
                full_attr=not exclude_colors_training,
                run_id=run_id, # Pass run_id for logging/naming inside evaluation
                gw_checkpoint_path=ckpt_path,
                model_version=model_version_name,
                output_parent_dir=evaluation_parent_dir, # Let evaluation create its specific subdir
                encoders_n_layers=config.global_workspace.encoders.n_layers,
                decoders_n_layers=config.global_workspace.decoders.n_layers,
                encoders_hidden_dim=config.global_workspace.encoders.hidden_dim,
                decoders_hidden_dim=config.global_workspace.decoders.hidden_dim,
                debug_mode=debug_mode
                # Ensure run_evaluation saves to `eval_output_dir / "analysis_results.pkl"`
            )

            if eval_result:
                results_path, analysis_results_current = eval_result
                # Verify path matches expectation
                if Path(results_path).resolve() != potential_eval_pkl.resolve():
                     LOGGER.warning(f"Evaluation function returned path {results_path}, but expected {potential_eval_pkl}. Using returned path.")
                     # Consider standardizing the save path within run_evaluation if possible

                model_eval_results_paths[run_id] = results_path # Store the actual returned path
                successful_eval_runs.append(run_id)
                if run_id == first_run_id:
                    analysis_results_first_model = analysis_results_current
                LOGGER.info(f"Evaluation successful for {run_id}. Results: {results_path}")
                del analysis_results_current # Free memory if large
            else:
                LOGGER.error(f"Evaluation function returned None or False for run {run_id}. It will be excluded.")
                failed_eval_runs.append(run_id)

        except Exception as e:
            LOGGER.error(f"Exception during evaluation for run {run_id}: {e}", exc_info=debug_mode)
            failed_eval_runs.append(run_id)

    # Update evaluation summary and details
    workflow_results["evaluation_summary"]["successful"] = len(successful_eval_runs)
    workflow_results["evaluation_summary"]["failed_runs"] = failed_eval_runs
    workflow_results["evaluation_details"]["results_paths"] = model_eval_results_paths
    # Optionally store the raw analysis data for the first model if needed later
    # workflow_results["evaluation_details"]["first_model_analysis_raw"] = analysis_results_first_model

    # --- Check if Comparison is Possible ---
    if analysis_results_first_model is None:
        LOGGER.error(f"Evaluation results missing or failed for the first model ({first_run_id}). Cannot perform significance filtering. Exiting.")
        LOGGER.info(format_results_summary(workflow_results))
        return workflow_results
    if len(successful_eval_runs) < 2:
        LOGGER.error(f"Fewer than two models evaluated successfully ({len(successful_eval_runs)}). Cannot perform pairwise comparison. Exiting.")
        LOGGER.info(format_results_summary(workflow_results))
        return workflow_results

    LOGGER.info(f"\n--- Evaluation Phase Complete. Successfully evaluated models: {successful_eval_runs} ---")

    # --- Phase 3: Filter Attributes based on FIRST Model ---
    paths_to_check = ['translated', 'half_cycle', 'full_cycle'] # Make this configurable?
    all_significant_attributes_union: Dict[str, Set[str]] = {}

    LOGGER.info(f"\n--- Filtering Attributes Based on FIRST Model ({first_run_id}) Significance (p < {filtering_alpha}) ---")
    workflow_results["filtering_summary"]["performed"] = True
    workflow_results["filtering_summary"]["based_on_model"] = first_run_id

    try:
        for path_name in paths_to_check:
            LOGGER.info(f"--- Checking Path: {path_name} ---")
            # Assuming find_significant_bin_comparisons returns Dict[str, Set[str]]
            significant_attributes_current_path = find_significant_bin_comparisons(
                analysis_results=analysis_results_first_model, # Use the loaded/generated dict
                path_name=path_name,
                alpha=filtering_alpha
            )
            if significant_attributes_current_path:
                LOGGER.info(f"Found {len(significant_attributes_current_path)} significant attributes in path '{path_name}' for model {first_run_id}.")
                for attr, bins_set in significant_attributes_current_path.items():
                    all_significant_attributes_union.setdefault(attr, set()).update(bins_set)
            else:
                 LOGGER.info(f"No significant attributes found in path '{path_name}' for model {first_run_id}.")

        workflow_results["filtering_details"]["significant_attributes"] = all_significant_attributes_union
        workflow_results["filtering_summary"]["num_significant_attributes"] = len(all_significant_attributes_union)
        workflow_results["filtering_summary"]["total_significant_attribute_bins"] = sum(len(v) for v in all_significant_attributes_union.values())

        LOGGER.info("\n--- Significance Filtering Summary ---")
        LOGGER.info(f"Total unique significant attributes across paths: {len(all_significant_attributes_union)}")
        LOGGER.info(f"Total significant attribute-bin combinations: {workflow_results['filtering_summary']['total_significant_attribute_bins']}")

    except Exception as e:
         LOGGER.error(f"Exception during significance filtering: {e}", exc_info=debug_mode)
         # Decide how to handle this - maybe exit, maybe continue without filtering
         LOGGER.warning("Proceeding without filtered attributes due to error.")
         all_significant_attributes_union = {} # Reset to prevent comparison
         workflow_results["filtering_summary"]["performed"] = False # Mark as not performed


    # --- Phase 4: Pairwise Cross-Model Comparison (Conditionally Saving) ---
    num_significant_comparisons_found = 0
    significant_comparisons_details = {}
    comparison_performed_flag = False # Track if comparison logic runs

    if not all_significant_attributes_union:
        LOGGER.info("\nSkipping pairwise cross-model comparison as no attributes passed the initial filtering.")
    else:
        comparison_performed_flag = True
        LOGGER.info(f"\n--- Starting Pairwise Cross-Model Comparison for {len(successful_eval_runs)} Models ---")
        LOGGER.info(f"--- Comparing attributes/bins found significant in {first_run_id} (p < {filtering_alpha}) ---")
        LOGGER.info(f"--- Comparison details/plots saved ONLY if KS p-value < {comparison_save_alpha} ---")

        # Define final comparison output dir name based on actual runs and settings
        comparison_output_dir_final = comparison_output_dir / f"{model_version_name}_N{len(successful_eval_runs)}_filt{filtering_alpha}_save{comparison_save_alpha}"
        comparison_output_dir_final.mkdir(parents=True, exist_ok=True)
        workflow_results["output_locations"]["comparison_output_dir"] = str(comparison_output_dir_final) # Update final path

        model_pairs = list(itertools.combinations(successful_eval_runs, 2))
        LOGGER.info(f"Performing {len(model_pairs)} pairwise comparisons.")
        workflow_results["comparison_summary"]["num_models_compared"] = len(successful_eval_runs)
        workflow_results["comparison_summary"]["num_pairs_compared"] = len(model_pairs)


        for id1, id2 in model_pairs:
            LOGGER.debug(f"\n--- Comparing Pair: {id1} vs {id2} ---")
            path1_pkl = model_eval_results_paths[id1]
            path2_pkl = model_eval_results_paths[id2]
            pair_key = f"{id1}_vs_{id2}"
            significant_comparisons_details[pair_key] = [] # Initialize list for this pair

            # Iterate through the filtered attributes/bins
            for attribute, bin_names_set in all_significant_attributes_union.items():
                for path_name_for_comparison in paths_to_check: # Compare across same paths as filtering
                    for bin_name in sorted(list(bin_names_set)):
                        try:
                            # Assuming compare_hue_distribution... handles loading pkls internally now
                            # It should return None on error, or a dict with metrics including 'is_significant'
                            metrics_dict = compare_hue_distribution_for_bin_across_models(
                                model_results_paths=[path1_pkl, path2_pkl], # Pass paths to pkl files
                                target_attribute=attribute,
                                target_bin_name=bin_name,
                                output_dir=comparison_output_dir_final, # Dir for saving plots/data
                                path_name=path_name_for_comparison,
                                model_labels=[id1, id2],
                                num_histogram_bins=50, # Make configurable?
                                significance_alpha=comparison_save_alpha # Alpha for KS test significance
                            )

                            if metrics_dict and metrics_dict.get('is_significant', False):
                                num_significant_comparisons_found += 1
                                significant_comparisons_details[pair_key].append({
                                    'attribute': attribute,
                                    'bin': bin_name,
                                    'path': path_name_for_comparison,
                                    'metrics': metrics_dict # Contains KS p-value, statistic, etc.
                                })
                                LOGGER.debug(f"  Significant difference found for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2} (p={metrics_dict.get('ks_p_value', 'N/A'):.4f})")
                            # Optional: Log non-significant comparisons if needed (can be very verbose)
                            # elif metrics_dict:
                            #     LOGGER.debug(f"  No significant difference for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2} (p={metrics_dict.get('ks_p_value', 'N/A'):.4f})")
                            elif metrics_dict is None:
                                LOGGER.warning(f"      Comparison function returned None for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2}. Check logs.")

                        except Exception as e:
                            LOGGER.error(f"Exception during comparison for {attribute}/{bin_name}/{path_name_for_comparison} between {id1} and {id2}: {e}", exc_info=debug_mode)


        # Update comparison summary and details
        workflow_results["comparison_summary"]["performed"] = comparison_performed_flag
        workflow_results["comparison_summary"]["num_significant_differences_found"] = num_significant_comparisons_found
        workflow_results["comparison_details"]["significant_comparisons"] = significant_comparisons_details

        # Log final comparison summary
        LOGGER.info(f"\n--- Comparison Phase Summary ---")
        LOGGER.info(f"Found {num_significant_comparisons_found} significant cross-model differences (p < {comparison_save_alpha}) across {len(model_pairs)} pairs.")
        if num_significant_comparisons_found > 0:
             LOGGER.info(f"Details saved in comparison output directory: {comparison_output_dir_final}")
        elif comparison_performed_flag: # Comparison ran but found nothing significant
            LOGGER.info(f"No significant differences found meeting the criteria. Output directory: {comparison_output_dir_final}")


    LOGGER.info("\n===== Full Workflow Completed =====")

    # --- Final Summary Logging ---
    LOGGER.info(format_results_summary(workflow_results))

    return workflow_results


if __name__ == "__main__":
    config = load_config("./config", use_cli=False, load_files=["base_params.yaml"])
    config.training.max_steps = 243
    custom_hparams = {
        "temperature": 1.0,
        "alpha": 1.0
    }   
    results = run_ssd_multiseed_workflow(
        num_models_to_train=2,
        model_version_name="debugging",
        base_output_dir=Path("./test_arbitrary/"),
        exclude_colors_training=False,
        apply_custom_init_training=True,    
        config=config,
        custom_hparams=custom_hparams,
        project_name="debugging"
    )
    print(results)