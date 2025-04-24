import collections.abc
from pathlib import Path
import os
import math
import ast
import io
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union, cast, Literal

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import patches
import cv2
import pandas as pd
from PIL import Image

from tqdm import tqdm
from itertools import combinations

from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from SSD_utils import (
    generate_fixed_colors,
    normalize_position,
    normalize_size,
    normalize_rotation,
    kl_divergence,
    bin_attribute,
    segment_shape,
    extract_shape_color,
    initialize_h_binning_structures,
    bin_h_processed_samples_with_paths, # Assuming this exists and handles 3 paths
    preprocess_dataset, # Assuming this exists
    process_through_global_workspace,
    save_binned_results,
    load_binned_results,
    comparison_metrics,
    default_binning_config,
    binning_config_6144,
)
from SSD_visualize_functions import (
    visualize_color_distributions_by_attribute,
    visualize_input_output_distributions, # Should be adaptable or replaced
    visualize_kl_heatmap,
    visualize_examples_by_attribute, 
    visualize_distribution_comparison,
)

# Mock LOGGER if not imported
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add json import at the top of the file if not already present
import json

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper function to perform comparison for a single bin and path
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Updated HueShapeAnalyzer class using the modified functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HueShapeAnalyzer:
    """Class to analyze shape data across multiple attribute dimensions using Hue channel."""

    def __init__(
        self,
        global_workspace, # Assume this is the trained model/pipeline
        device: torch.device,
        shape_names: List[str] = ["diamond", "egg", "triangle"],
        color: bool = True,
        output_dir: str = ".",
        seed=0,
        debug = False, 
        num_bins = 50,
        reverb_n = 1, # Number of reverberation cycles for full cycle and translation path
        binning_config: Dict = default_binning_config, # Use the default config
        rgb = False
    ):
        """
        Initialize the hue shape analyzer.
        """
        self.global_workspace = global_workspace
        self.device = device
        self.shape_names = shape_names
        self.color = color # Controls if color attributes are used in input vectors
        self.output_dir = output_dir
        self.seed = seed
        self.debug = debug
        self.num_bins = num_bins # Number of bins for histogram visualization adb statistical tests
        self.reverb_n = reverb_n # Number of reverberation cycles for full cycle and translation path
        os.makedirs(self.output_dir, exist_ok=True)
        # Use a fixed set of colors for reproducibility if needed
        self.rgb_colors, self.hls_colors = generate_fixed_colors(100)
        self.binning_config = binning_config # Use the default config
        self.rgb = rgb # Flag to indicate if RGB colors are used

    def _process_csv(
        self,
        csv_path: str,
        attributes: List[str], # analysis_attributes
        use_fixed_reference: bool = False,
        reference_color_idx: int = 0,
        im_dir: str = "./evaluation_set" # Added default
    ) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
        """
        Load the CSV file, preprocess all samples, and process them through the global workspace.
        """
        df = pd.read_csv(csv_path)
        # Handle potential string representation of lists/tuples
        if isinstance(df["location"].iloc[0], str):
             df["location"] = df["location"].apply(ast.literal_eval)
        if use_fixed_reference:
            fixed_color = self.rgb_colors[reference_color_idx]
            # Ensure it's a list of lists/tuples if needed by downstream code
            df["fixed_color"] = [list(fixed_color)] * len(df) # Example: store as list

        LOGGER.info(f"Preprocessing data from {csv_path}...")

        
        preprocessed_samples = preprocess_dataset(
            df,
            attributes,
            self.shape_names,
            self.color, # Use instance attribute
            self.rgb_colors,
            self.device,
            fixed_reference=use_fixed_reference,
            reference_color_idx=reference_color_idx,
            im_dir=im_dir
        )
        LOGGER.info(f"Processing {len(preprocessed_samples)} samples through global workspace...")
        # Assuming process_through_global_workspace returns dicts with keys like:
        # 'translated_shape_color', 'half_cycle_shape_color', 'full_cycle_shape_color',
        # 'translated_image', 'half_cycle_image', 'full_cycle_image' etc.
        processed_samples = process_through_global_workspace(
            self.global_workspace,
            preprocessed_samples,
            self.device,
            debug = self.debug, 
            reverb_n=self.reverb_n,
        )
        return df, preprocessed_samples, processed_samples

    def analyze_dataset(
        self,
        csv_path: str,
        analysis_attributes: List[str] = None,
        display_examples: bool = True,
        seed=None,
        binning_config=None,
        im_dir: str = "./evaluation_set" # Added default image dir
    ) -> Dict[str, Any]:
        """
        Analyze a dataset of shape images using Hue channel across multiple processing paths.

        Args:
            csv_path (str): Path to the dataset CSV file.
            analysis_attributes (List[str], optional): List of attributes to analyze.
                Defaults to ['shape', 'rotation', 'size', 'position_x', 'position_y'].
            display_examples (bool, optional): Whether to store and visualize example images per bin.
                Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to instance seed.
            binning_config (Dict, optional): Configuration for binning attributes.
                Defaults to instance default config.
            im_dir (str, optional): Directory containing the images referenced by the CSV.
                Defaults to "./evaluation_set".


        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, including
                            KL divergences for each path (input vs translated, vs half-cycle,
                            vs full-cycle) and the binned color distributions.
        """
        if analysis_attributes is None:
            analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        if binning_config is None:
            binning_config = self.binning_config # Use instance default
        if seed is None:
            seed = self.seed
        seed_everything(seed) # Set seed if using PyTorch Lightning

        results = {attr: {} for attr in analysis_attributes} # Initialize results dict

        LOGGER.info(f"Starting analysis for {csv_path}")

        save_path = os.path.join(self.output_dir, 'binned_results.pkl')

        LOGGER.info("Initializing binning structures...")
            # Initialize binning structures for H channel, now expecting structures for all paths
            # (Make sure initialize_h_binning_structures creates dicts for all 4 sets of colors)
        (
            input_colors_by_attr,
            translated_colors_by_attr, # Renamed for clarity
            half_cycle_colors_by_attr,
            full_cycle_colors_by_attr,
            examples_by_attr
        ) = initialize_h_binning_structures(
            analysis_attributes,
            binning_config, self.rgb
        )
        if os.path.exists(save_path):
            LOGGER.warning(f"Loading existing binned results from {save_path}")


            (
                input_colors_by_attr,
                translated_colors_by_attr,
                half_cycle_colors_by_attr,
                full_cycle_colors_by_attr,
                examples_by_attr,
                binning_config,
                analysis_attributes
            ) = load_binned_results(self.output_dir)
            

        else : 
            df, preprocessed_samples, processed_samples = self._process_csv(
                csv_path,
                analysis_attributes, # Pass attributes for preprocessing reference
                use_fixed_reference=False, # Standard analysis uses original colors
                im_dir=im_dir
            )

            

            LOGGER.info("Binning processed samples...")
            # Bin processed samples for H channel analysis across all paths
            # (Make sure bin_h_processed_samples_with_paths populates all 4 color dicts correctly)
            bin_h_processed_samples_with_paths(
                preprocessed_samples=preprocessed_samples,
                processed_samples=processed_samples,
                analysis_attributes=analysis_attributes,
                binning_config=binning_config,
                input_colors_by_attr=input_colors_by_attr,
                translated_colors_by_attr=translated_colors_by_attr, # Pass the correct target dict
                half_cycle_colors_by_attr=half_cycle_colors_by_attr, # Pass the correct target dict
                full_cycle_colors_by_attr=full_cycle_colors_by_attr, # Pass the correct target dict
                examples_by_attr=examples_by_attr,
                display_examples=display_examples, 
                rgb=self.rgb # Pass the RGB flag
            )
            LOGGER.info("Binning complete, saving the results")
            # Save the results to the output directory
            save_binned_results(self.output_dir, input_colors_by_attr, translated_colors_by_attr, 
                            half_cycle_colors_by_attr, full_cycle_colors_by_attr, 
                            examples_by_attr, binning_config, analysis_attributes)


        LOGGER.info("Processing analysis attributes and comparing paths...")
        # Process analysis attributes using the new multi-path function
        results = process_analysis_attributes(
            analysis_attributes=analysis_attributes,
            output_dir=self.output_dir,
            color=self.color, # Pass color flag for directory naming
            binning_config=binning_config,
            input_colors_by_attr=input_colors_by_attr,
            translated_colors_by_attr=translated_colors_by_attr, # Pass renamed dict
            half_cycle_colors_by_attr=half_cycle_colors_by_attr,
            full_cycle_colors_by_attr=full_cycle_colors_by_attr,
            examples_by_attr=examples_by_attr,
            results=results,  # Pass the results dict to be updated
            num_bins = self.num_bins, # Pass the number of bins for visualization
            significance_alpha=0.05 # Pass the significance threshold
        )

        LOGGER.info(f"Analysis complete. Results saved in {self.output_dir}")
        return results

   


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NEW Heatmap function for bin-pair comparisons (KL or KS)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def visualize_metric_heatmap_bin_pairs(
    metric_results: Dict[Tuple[str, str], Dict[str, float]], # e.g., {('bin1', 'bin2'): {'kl_symmetric': val, ...}}
    metric_key: str, # e.g., 'kl_symmetric', 'ks_pval'
    bin_names: List[str],
    attribute: str,
    path_name: str,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    invert_cmap: bool = False, # e.g., for p-values, smaller is more significant
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotation_fmt: str = ".2f" # Format for annotations in cells
):
    """
    Visualizes a heatmap of a specific metric (KL, KS p-value, etc.)
    calculated between pairs of bins for a given attribute and path.

    Args:
        metric_results: Dict mapping (bin1_name, bin2_name) tuples to metric dicts.
        metric_key: The key within the inner metric dict to plot (e.g., 'kl_symmetric').
        bin_names: Ordered list of bin names for the axes.
        attribute: Name of the attribute.
        path_name: Name of the path.
        save_path: Full path to save the heatmap image.
        title: Optional title for the plot.
        cmap: Matplotlib colormap name.
        invert_cmap: If True, use the reversed version of the colormap.
        vmin, vmax: Optional min/max values for the color scale.
        annotation_fmt: String format for cell annotations.
    """
    n_bins = len(bin_names)
    if n_bins < 2:
        LOGGER.warning(f"Skipping heatmap for {attribute} ({path_name}) - requires at least 2 bins.")
        return

    matrix = np.full((n_bins, n_bins), np.nan) # Initialize with NaNs

    for i in range(n_bins):
        for j in range(n_bins):
            if i == j:
                # Set diagonal based on metric (e.g., 0 for KL/distance, 1 for p-value)
                matrix[i, j] = 0.0 if 'kl' in metric_key or 'ks_stat' in metric_key else 1.0 if 'pval' in metric_key else np.nan
                continue

            # Try fetching result for (bin_i, bin_j) or (bin_j, bin_i)
            bin_i = bin_names[i]
            bin_j = bin_names[j]
            result_dict = metric_results.get((bin_i, bin_j), metric_results.get((bin_j, bin_i), None))

            if result_dict and metric_key in result_dict:
                value = result_dict[metric_key]
                if value is not None and np.isfinite(value):
                    matrix[i, j] = value
                # else: keep as NaN if metric is missing, None, or non-finite

    if np.isnan(matrix).all():
        LOGGER.warning(f"Skipping heatmap for {attribute} ({path_name}), metric '{metric_key}' - no valid data found.")
        return

    # Determine vmin/vmax if not provided
    valid_values = matrix[~np.isnan(matrix)]
    if vmin is None:
        vmin = np.min(valid_values) if len(valid_values) > 0 else 0
    if vmax is None:
        # Special handling for p-values often capped at 0.05 or 1.0
        if 'pval' in metric_key:
             vmax = 1.0 # Typically p-values range 0-1
             # Or maybe focus on significance: vmax = 0.1
        else:
             vmax = np.max(valid_values) if len(valid_values) > 0 else 1

    # Handle colormap inversion
    if invert_cmap:
        try:
            if not cmap.endswith("_r"):
                cmap = cmap + "_r"
        except Exception: # Catch potential issues with cmap name manipulation
             pass # Use original cmap if inversion fails

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 0.8), max(5, n_bins * 0.7)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    # Configure axes
    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels(bin_names)
    ax.set_yticklabels(bin_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add annotations
    for i in range(n_bins):
        for j in range(n_bins):
            value = matrix[i, j]
            if not np.isnan(value):
                # Adjust text color based on background intensity
                bg_color_val = (value - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else 0.5
                text_color = "white" if bg_color_val < 0.5 else "black"
                # Special highlight for significant p-values?
                if 'pval' in metric_key and value < 0.05:
                     text_color = "red" # Example: highlight significant p-values
                ax.text(j, i, f"{value:{annotation_fmt}}", ha="center", va="center", color=text_color)
            else:
                 ax.text(j, i, "N/A", ha="center", va="center", color="grey")


    # Add colorbar and title
    fig.colorbar(im, ax=ax, label=metric_key.replace('_', ' ').title())
    if title is None:
        title = f"{metric_key.replace('_', ' ').title()} between Bins\nAttribute: {attribute.title()}, Path: {path_name.title()}"
    ax.set_title(title, wrap=True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        # LOGGER.info(f"Saved heatmap to {save_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save heatmap {save_path}: {e}")
    plt.close(fig)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# REPLACEMENT main analysis function focusing on BIN-PAIR comparisons per PATH
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def process_analysis_attributes( # Renamed function
    analysis_attributes: List[str],
    output_dir: str,
    color: bool, # Keep for directory naming consistency
    binning_config: Dict,
    input_colors_by_attr: Dict, # Still useful context
    translated_colors_by_attr: Dict,
    half_cycle_colors_by_attr: Dict,
    full_cycle_colors_by_attr: Dict,
    examples_by_attr: Dict,
    results: Dict, # Pass results dict to update it
    num_bins: int = 50, # Default number of bins for histograms
    significance_alpha: float = 0.05, 
    rgb = False # Threshold for significance counting
) -> Dict:
    """
    Processes analysis attributes by comparing pairs of bins within each attribute
    for each processing path (translated, half-cycle, full-cycle).

    Calculates KL divergence and KS tests between H channel distributions of bin pairs.
    Visualizes bin-pair distribution comparisons and generates metric heatmaps.
    Generates and saves a global summary of comparison statistics.

    Args:
        analysis_attributes: List of attributes to analyze (e.g., ['shape', 'rotation']).
        output_dir: Base directory for saving results.
        color: Boolean indicating if color was used (affects directory naming).
        binning_config: Dictionary defining bins for each attribute.
        input_colors_by_attr: Dict mapping attr -> bin_name -> {'H': [values...]} (Stored for context).
        translated_colors_by_attr: Dict for translated path results.
        half_cycle_colors_by_attr: Dict for half-cycle path results.
        full_cycle_colors_by_attr: Dict for full-cycle path results.
        examples_by_attr: Dict mapping attr -> bin_name -> [example_dict].
        results: Dictionary to store the detailed analysis results.
        num_bins: Number of bins for histograms and metric calculations.
        significance_alpha: P-value threshold for counting significant comparisons.

    Returns:
        The updated results dictionary containing detailed comparisons.
    """

    # --- Initialize Global Summary Statistics ---
    total_comparisons_overall = 0
    total_significant_comparisons_overall = 0
    attribute_stats_summary: Dict[str, Dict[str, int]] = {}
    path_stats_summary: Dict[str, Dict[str, int]] = {}
    # Optional: bin_stats_summary could be added if needed

    # Define the path data sources and their names
    path_data_sources = {
        "translated": translated_colors_by_attr,
        "half_cycle": half_cycle_colors_by_attr,
        "full_cycle": full_cycle_colors_by_attr,
    }

    # Initialize path stats summary structure
    for path_name in path_data_sources.keys():
        path_stats_summary[path_name] = {'total': 0, 'significant': 0}

    for attr in analysis_attributes:
        LOGGER.info(f"Processing attribute for bin-pair comparisons: {attr}")
        attr_base_dir = os.path.join(output_dir, f"{attr}{'_nocolor' if not color else ''}_BinComparisons")
        os.makedirs(attr_base_dir, exist_ok=True)

        # --- Initialize results storage and summary stats for this attribute ---
        results[attr] = results.get(attr, {})
        attribute_stats_summary[attr] = {'total': 0, 'significant': 0}
        # Store original distributions for context (optional, but can be useful)
        results[attr]['input_distributions'] = input_colors_by_attr.get(attr, {})
        for path_name, source_dict in path_data_sources.items():
            results[attr][f'{path_name}_distributions'] = source_dict.get(attr, {})
            # Initialize storage for bin-pair metrics for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = {}


        # Get bins and pairs for this attribute
        bin_names = binning_config.get(attr, {}).get('bin_names', [])
        if len(bin_names) < 2:
            LOGGER.warning(f"Attribute '{attr}' has fewer than 2 bins ({bin_names}). Skipping bin-pair comparisons.")
            continue
        bin_pairs = list(combinations(bin_names, 2))

        # --- Iterate through each PATH ---
        for path_name, path_data in path_data_sources.items():

            # Directory for this path's comparison plots and heatmaps
            path_comparison_dir = os.path.join(attr_base_dir, f"{path_name}_bin_comparisons")

            if attr not in path_data:
                LOGGER.warning(f"    No data found for attribute '{attr}' in path '{path_name}'. Skipping.")
                continue
            path_data_for_attr = path_data[attr] # e.g., translated_colors_by_attr['shape']

            # Dictionary to store results for this specific path and attribute
            bin_comparison_results_for_path = {}

            # --- Iterate through BIN PAIRS ---
            for bin1_name, bin2_name in bin_pairs:
                total_comparisons_overall += 1 # Increment global total
                attribute_stats_summary[attr]['total'] += 1 # Increment attr total
                path_stats_summary[path_name]['total'] += 1 # Increment path total

                # Ensure the helper function _compare_distributions_between_bins exists or replace call
                # Assuming it calls calculate_distribution_metrics internally or directly:
                metrics = _compare_distributions_between_bins(
                    path_name=path_name,
                    attribute=attr,
                    bin1_name=bin1_name,
                    bin2_name=bin2_name,
                    path_data_for_attribute=path_data_for_attr,
                    output_dir_for_path_comparison=path_comparison_dir,
                    num_bins = num_bins
                )


                bin_comparison_results_for_path[(bin1_name, bin2_name)] = metrics
                # Check significance for summary
                ks_pval = metrics.get('ks_pval')
                # Ensure ks_pval is not None and is finite before comparison
                if ks_pval is not None and np.isfinite(ks_pval) and ks_pval < significance_alpha:
                    total_significant_comparisons_overall += 1
                    attribute_stats_summary[attr]['significant'] += 1
                    path_stats_summary[path_name]['significant'] += 1


            # Store the detailed bin-pair results for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = bin_comparison_results_for_path

            # --- Generate Heatmaps for the current path (after comparing all its bin pairs) ---
            if bin_comparison_results_for_path: # Check if any results were stored
                # Ensure the output directory exists for heatmaps
                os.makedirs(path_comparison_dir, exist_ok=True)

                # Symmetric KL Heatmap
                if 'visualize_metric_heatmap_bin_pairs' in globals():
                    try:
                        visualize_metric_heatmap_bin_pairs(
                             metric_results=bin_comparison_results_for_path,
                             metric_key='kl_symmetric',
                             bin_names=bin_names,
                             attribute=attr,
                             path_name=path_name,
                             save_path=os.path.join(path_comparison_dir, f'{path_name}_{attr}_heatmap_KL_symmetric.png'),
                             vmin=0 # KL >= 0
                         )
                    except Exception as e:
                         LOGGER.error(f"Failed to generate KL heatmap for {attr}, path {path_name}: {e}", exc_info=True)

                    # KS p-value Heatmap
                    try:
                         visualize_metric_heatmap_bin_pairs(
                             metric_results=bin_comparison_results_for_path,
                             metric_key='ks_pval',
                             bin_names=bin_names,
                             attribute=attr,
                             path_name=path_name,
                             save_path=os.path.join(path_comparison_dir, f'{path_name}_{attr}_heatmap_KS_pvalue.png'),
                             cmap='viridis_r', # Invert cmap for p-values (low = significant)
                             vmin=0, vmax=1.0,
                             annotation_fmt=".3f" # More precision for p-values
                         )
                    except Exception as e:
                         LOGGER.error(f"Failed to generate KS p-value heatmap for {attr}, path {path_name}: {e}", exc_info=True)
                else:
                    LOGGER.warning("`visualize_metric_heatmap_bin_pairs` function not found. Skipping heatmap generation.")
            else:
                 LOGGER.warning(f"No valid bin comparison results found for path '{path_name}', attribute '{attr}'. Skipping heatmaps.")


        # --- Visualize examples (outside path loop, only needs attr data) ---
        if 'visualize_examples_by_attribute' in globals():
            if attr in examples_by_attr and any(examples_by_attr[attr].values()):
                try: visualize_examples_by_attribute( examples_by_attr[attr], attr, bin_names, os.path.join(attr_base_dir, f'examples_by_{attr}.png'), )
                except Exception as e: LOGGER.error(f"Failed to visualize examples for {attr}: {e}", exc_info=True)
            else: LOGGER.warning(f"Skipping examples visualization for {attr} due to missing data.")
        else: LOGGER.warning("`visualize_examples_by_attribute` not found. Skipping examples.")

    # --- Assemble and Save Global Summary ---
    global_summary = {
        "total_comparisons": total_comparisons_overall,
        "significant_comparisons": total_significant_comparisons_overall,
        "significance_alpha": significance_alpha,
        "attribute_stats": attribute_stats_summary,
        "path_stats": path_stats_summary,
        # Add bin_stats_summary here if implemented
    }

    summary_file_path = os.path.join(output_dir, "within_model_comparison_summary.json")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(global_summary, f, indent=2)
        LOGGER.info(f"Within-model comparison summary saved to: {summary_file_path}")
    except TypeError as e:
         LOGGER.error(f"Failed to serialize within-model summary to JSON: {e}. Summary was: {global_summary}")
    except Exception as e:
         LOGGER.error(f"Failed to save within-model comparison summary: {e}", exc_info=True)


    return results


