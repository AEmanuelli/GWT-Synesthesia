import collections.abc
from pathlib import Path
import os
import math
import ast
import io
import warnings
import json # Make sure json is imported
from typing import Any, Dict, List, Tuple, Optional, Union, cast, Literal
from itertools import combinations # Import combinations

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
from scipy.stats import ks_2samp
from tqdm import tqdm

from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Assuming these utilities are adapted or available for RGB as well
from SSD_utils import (
    generate_fixed_colors,
    normalize_position,
    normalize_size,
    normalize_rotation,
    kl_divergence,
    bin_attribute,
    segment_shape,
    # extract_shape_color, # Less relevant if process_through_global_workspace provides colors
    # initialize_h_binning_structures, # Replaced
    # bin_h_processed_samples_with_paths, # Replaced
    preprocess_dataset, # Assuming this exists and handles inputs
    process_through_global_workspace, # Assuming this can return multiple paths
    safe_extract_channel, # Used for extracting R, G, B
    save_binned_results, # Needs adaptation for multiple paths & RGB
    load_binned_results, # Needs adaptation for multiple paths & RGB
    comparison_metrics, # Reused from Hue version
)
# Assuming these visualization functions are adapted or available
from SSD_visualize_functions import (
    # visualize_color_distributions_by_attribute, # Used in compare_color_distributions_across_shapes
    visualize_distribution_comparison_rgb, # NEW/ADAPTED: Shows R,G,B comparison
    # visualize_input_output_distributions, # Replaced by visualize_distribution_comparison_rgb
    visualize_metric_heatmap_bin_pairs_rgb, # NEW/ADAPTED: From visualize_metric_heatmap_bin_pairs
    visualize_examples_by_attribute,
    # visualize_distribution_comparison, # Replaced by visualize_distribution_comparison_rgb
)

# Mock LOGGER if not imported
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Default Binning Config (remains the same)
default_binning_config = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 4,
        'range': (0, 2 * np.pi),
        'bin_names': ['0-90', '90-180', '180-270', '270-360']
    },
    'size': {
        'n_bins': 4,
        'range': (7, 14),
        'bin_names': ['Very Small', 'Small', 'Medium', 'Large']
    },
    'position_x': {
        'n_bins': 2,
        'range': (0, 32), # Assuming 32x32 images
        'bin_names': ['Left', 'Right']
    },
    'position_y': {
        'n_bins': 2,
        'range': (0, 32), # Assuming 32x32 images
        'bin_names': ['Bottom', 'Top']
    }
}



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helper functions for RGB Analysis (Adapted from Hue version)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def initialize_rgb_binning_structures(
    analysis_attributes: List[str],
    binning_config: Dict,
    channels: List[str] = ["R", "G", "B"] # Default to R, G, B
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Initialize data structures for storing binned RGB results for multiple paths.

    Args:
        analysis_attributes: List of attributes to analyze.
        binning_config: Configuration for binning.
        channels: List of color channels (default: R, G, B).

    Returns:
        Five dictionaries: input_colors, translated_colors, half_cycle_colors,
                         full_cycle_colors, and examples_by_attr.
    """
    input_colors_by_attr = {}
    translated_colors_by_attr = {}
    half_cycle_colors_by_attr = {}
    full_cycle_colors_by_attr = {}
    examples_by_attr = {} # Stores example images/data per bin

    all_path_dicts = [
        input_colors_by_attr,
        translated_colors_by_attr,
        half_cycle_colors_by_attr,
        full_cycle_colors_by_attr
    ]

    for attr in analysis_attributes:
        examples_by_attr[attr] = {}
        bin_names = binning_config.get(attr, {}).get('bin_names', [])
        if not bin_names:
            LOGGER.warning(f"No bin names found for attribute '{attr}' in binning_config.")
            continue

        for path_dict in all_path_dicts:
            path_dict[attr] = {}
            for bin_name in bin_names:
                path_dict[attr][bin_name] = {ch: [] for ch in channels}

        for bin_name in bin_names:
            examples_by_attr[attr][bin_name] = [] # Initialize example list for the bin

    return (
        input_colors_by_attr,
        translated_colors_by_attr,
        half_cycle_colors_by_attr,
        full_cycle_colors_by_attr,
        examples_by_attr
    )


def bin_rgb_processed_samples_with_paths(
    preprocessed_samples: List[Dict],
    processed_samples: List[Dict],
    analysis_attributes: List[str],
    binning_config: Dict,
    input_colors_by_attr: Dict,
    translated_colors_by_attr: Dict,
    half_cycle_colors_by_attr: Dict,
    full_cycle_colors_by_attr: Dict,
    examples_by_attr: Dict,
    channels: List[str] = ["R", "G", "B"], # Default to R, G, B
    display_examples: bool = True,
    max_examples_per_bin: int = 10
) -> None:
    """
    Bins processed samples based on attributes into the provided dictionaries
    for multiple paths, extracting R, G, B channel values.

    Args:
        preprocessed_samples: List of preprocessed sample dictionaries.
        processed_samples: List of processed sample dictionaries from the model.
                           Expected to contain keys like 'input_shape_color',
                           'translated_shape_color', 'half_cycle_shape_color', etc.
                           and potentially image keys like 'input_image', 'translated_image'.
        analysis_attributes: List of attributes to bin by.
        binning_config: Configuration for binning.
        input_colors_by_attr: Dictionary to store input R,G,B values.
        translated_colors_by_attr: Dictionary to store translated R,G,B values.
        half_cycle_colors_by_attr: Dictionary to store half-cycle R,G,B values.
        full_cycle_colors_by_attr: Dictionary to store full-cycle R,G,B values.
        examples_by_attr: Dictionary to store example images/data.
        channels: List of channels to extract (default R, G, B).
        display_examples: Whether to store example data.
        max_examples_per_bin: Max examples to store per bin.
    """
    path_mapping = {
        'input': (input_colors_by_attr, 'input_shape_color', 'input_image'),
        'translated': (translated_colors_by_attr, 'translated_shape_color', 'translated_image'),
        'half_cycle': (half_cycle_colors_by_attr, 'half_cycle_shape_color', 'half_cycle_image'),
        'full_cycle': (full_cycle_colors_by_attr, 'full_cycle_shape_color', 'full_cycle_image'),
    }

    for i, preprocessed_sample in enumerate(tqdm(preprocessed_samples, desc="Binning RGB Samples")):
        processed_sample = processed_samples[i]

        for attr in analysis_attributes:
            if attr not in preprocessed_sample:
                LOGGER.warning(f"Attribute '{attr}' not found in preprocessed sample {i}. Skipping binning for this attribute.")
                continue

            attr_value = preprocessed_sample[attr]
            bin_name = bin_attribute(attr_value, attr, binning_config)

            if bin_name is None:
                LOGGER.debug(f"Sample {i} could not be binned for attribute '{attr}' with value {attr_value}. Skipping.")
                continue

            # Store colors for each path
            for path_name, (target_dict, color_key, _) in path_mapping.items():
                if color_key not in processed_sample:
                     LOGGER.warning(f"Color key '{color_key}' not found in processed sample {i} for path '{path_name}'.")
                     continue
                # Ensure the color value is a list/tuple of numbers
                color_val = processed_sample[color_key]
                if not isinstance(color_val, (list, tuple)) or len(color_val) < 3:
                    LOGGER.warning(f"Unexpected color format for '{color_key}' in sample {i}: {color_val}. Expected list/tuple of length >= 3.")
                    continue

                # Extract R, G, B using safe_extract_channel or direct indexing if format is known
                # Assuming color_val is [R, G, B, ...] or similar standard order
                try:
                    r_val = float(color_val[0])
                    g_val = float(color_val[1])
                    b_val = float(color_val[2])
                except (IndexError, ValueError, TypeError) as e:
                    LOGGER.warning(f"Could not extract R,G,B from '{color_key}' in sample {i}: {color_val}. Error: {e}. Skipping color entry.")
                    continue


                if attr in target_dict and bin_name in target_dict[attr]:
                    if "R" in channels: target_dict[attr][bin_name]["R"].append(r_val)
                    if "G" in channels: target_dict[attr][bin_name]["G"].append(g_val)
                    if "B" in channels: target_dict[attr][bin_name]["B"].append(b_val)
                else:
                    LOGGER.warning(f"Bin '{bin_name}' for attribute '{attr}' not initialized correctly in '{path_name}' dictionary. Skipping color entry.")


            # Store examples
            if display_examples and attr in examples_by_attr and bin_name in examples_by_attr[attr]:
                if len(examples_by_attr[attr][bin_name]) < max_examples_per_bin:
                    example_data = {'index': i}
                    # Add images from all paths if available
                    for path_name, (_, _, image_key) in path_mapping.items():
                         if image_key in processed_sample:
                            example_data[f'{path_name}_image'] = processed_sample[image_key]
                         if f'{path_name}_shape_color' in processed_sample: # Store colors too
                            example_data[f'{path_name}_color'] = processed_sample[f'{path_name}_shape_color']
                    # Add original attributes
                    for key, val in preprocessed_sample.items():
                         if key not in example_data: # Avoid overwriting images/colors
                            example_data[key] = val
                    examples_by_attr[attr][bin_name].append(example_data)


def _compare_rgb_distributions_between_bins(
    path_name: str,
    attribute: str,
    bin1_name: str,
    bin2_name: str,
    path_data_for_attribute: Dict[str, Dict[str, List[float]]],
    output_dir_for_path_comparison: str,
    channels: List[str] = ["R", "G", "B"],
    num_bins: int = 50,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Compares R, G, B channel distributions between two specified bins for a
    single attribute within a single processing path. Calculates KL divergence
    and KS test for each channel.

    Args:
        path_name: Name of the processing path (e.g., "translated").
        attribute: Name of the attribute being analyzed (e.g., "shape").
        bin1_name: Name of the first bin for comparison.
        bin2_name: Name of the second bin for comparison.
        path_data_for_attribute: Dict containing binned data for the attribute
                                 under the specific path (e.g., {'diamond': {'R': [...], 'G':...}, ...}).
        output_dir_for_path_comparison: Directory to save visualizations.
        channels: List of channels to compare (default R, G, B).
        num_bins: Number of bins for histograms and KS test calculation.

    Returns:
        Dictionary mapping each channel to its metric results
        {'R': {'kl_1_vs_2': ..., 'ks_pval': ...}, 'G': {...}, 'B': {...}},
        or None if data is insufficient for all channels.
    """
    dist1_data = path_data_for_attribute.get(bin1_name, {})
    dist2_data = path_data_for_attribute.get(bin2_name, {})

    results_per_channel = {}
    has_any_data = False

    for ch in channels:
        values1 = np.array([x for x in dist1_data.get(ch, []) if x is not None and not np.isnan(x)])
        values2 = np.array([x for x in dist2_data.get(ch, []) if x is not None and not np.isnan(x)])

        if len(values1) <= 1 or len(values2) <= 1:
            LOGGER.warning(f"Skipping comparison between bins '{bin1_name}' and '{bin2_name}' for "
                           f"attribute '{attribute}', path '{path_name}', channel '{ch}' - insufficient data "
                           f"({bin1_name}: {len(values1)}, {bin2_name}: {len(values2)})")
            results_per_channel[ch] = None # Mark channel as having insufficient data
            continue

        has_any_data = True # Mark that we found data for at least one channel
        LOGGER.debug(f"  Comparing bins for path '{path_name}', attr '{attribute}', channel '{ch}': {bin1_name} ({len(values1)}) vs {bin2_name} ({len(values2)})")

        # Calculate Metrics using the comparison_metrics helper
        try:
            # Ensure comparison_metrics handles potential errors / non-finite results gracefully
            kl_12, kl_21, kl_sym, ks_stat, ks_pval = comparison_metrics(values1, values2, num_bins)

            if ks_pval < 0.05: # Log only significant or interesting results if needed
                 LOGGER.info(f"    Path '{path_name}', {attribute}, Ch '{ch}': '{bin1_name}' vs '{bin2_name}' -> "
                             f"Sym KL: {kl_sym:.4f}, KS Stat ({num_bins} bins): {ks_stat:.4f}, KS p-val: {ks_pval:.4f}")

            results_per_channel[ch] = {
                'kl_1_vs_2': kl_12,
                'kl_2_vs_1': kl_21,
                'kl_symmetric': kl_sym,
                'ks_stat': ks_stat,
                'ks_pval': ks_pval
            }
            if not np.isfinite(kl_sym):
                 LOGGER.warning(f"Symmetric KL non-finite for {path_name}, {attribute}, Ch {ch}, {bin1_name} vs {bin2_name}")

        except Exception as e:
             LOGGER.error(f"Failed metric calculation for {path_name}, {attribute}, Ch {ch}, {bin1_name} vs {bin2_name}: {e}", exc_info=True)
             results_per_channel[ch] = None # Mark channel as failed

    if not has_any_data:
        return None # Return None if no channel had enough data

    # --- Visualize Comparison (Combined R,G,B plot) ---
    os.makedirs(output_dir_for_path_comparison, exist_ok=True)
    safe_bin1 = bin1_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    safe_bin2 = bin2_name.replace("/", "_").replace(" ", "_").replace(":", "-")
    plot_filename = os.path.join(
        output_dir_for_path_comparison,
        f'{path_name}_{attribute}_{safe_bin1}_vs_{safe_bin2}_RGB_comparison.png'
    )
    try:
        # Assumes visualize_distribution_comparison_rgb exists and takes these args
        if 'visualize_distribution_comparison_rgb' in globals():
             visualize_distribution_comparison_rgb(
                 dist1_data=dist1_data, # Pass full bin data dict {R:[], G:[], B:[]}
                 dist2_data=dist2_data, # Pass full bin data dict
                 dist1_label=f"{path_name.title()} ({bin1_name})",
                 dist2_label=f"{path_name.title()} ({bin2_name})",
                 bin_name=f"{bin1_name} vs {bin2_name}",
                 attr=attribute,
                 save_path=plot_filename,
                 channels=channels,
                 num_bins=num_bins,
                 title=f"RGB Dist Comparison ({path_name.title()} Path)\n{attribute.title()}: {bin1_name} vs {bin2_name}"
             )
        else:
            LOGGER.warning("`visualize_distribution_comparison_rgb` function not found. Skipping plot generation.")
    except Exception as e:
        LOGGER.error(f"Failed to generate RGB comparison plot for {path_name}, {attribute}, {bin1_name} vs {bin2_name}: {e}", exc_info=True)


    return results_per_channel # Return dict with results per channel


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main Analysis Function (Adapted from Hue version's bin-pair focus)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def process_analysis_attributes_rgb(
    analysis_attributes: List[str],
    output_dir: str,
    color: bool, # Keep for directory naming consistency
    channels: List[str], # Should be ["R", "G", "B"] typically
    binning_config: Dict,
    input_colors_by_attr: Dict, # Kept for context/examples if needed
    translated_colors_by_attr: Dict,
    half_cycle_colors_by_attr: Dict,
    full_cycle_colors_by_attr: Dict,
    examples_by_attr: Dict,
    results: Dict, # Pass results dict to update it
    num_bins: int = 50, # Bins for histograms and metrics
    significance_alpha: float = 0.05 # Threshold for significance counting
) -> Dict:
    """
    Processes analysis attributes by comparing pairs of bins within each attribute
    for each processing path (translated, half-cycle, full-cycle), focusing on R, G, B channels.

    Calculates KL divergence and KS tests between R, G, B distributions of bin pairs.
    Visualizes bin-pair distribution comparisons (RGB) and generates metric heatmaps per channel.
    Generates and saves a global summary of comparison statistics.

    Args:
        analysis_attributes: List of attributes to analyze (e.g., ['shape', 'rotation']).
        output_dir: Base directory for saving results.
        color: Boolean indicating if color was used (affects directory naming).
        channels: List of channels to analyze (typically ['R', 'G', 'B']).
        binning_config: Dictionary defining bins for each attribute.
        input_colors_by_attr: Dict mapping attr -> bin_name -> {'R': [], 'G': [], 'B': []}.
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
    total_comparisons_overall = 0
    total_significant_comparisons_overall = 0
    attribute_stats_summary: Dict[str, Dict[str, int]] = {}
    path_stats_summary: Dict[str, Dict[str, int]] = {}

    path_data_sources = {
        "translated": translated_colors_by_attr,
        "half_cycle": half_cycle_colors_by_attr,
        "full_cycle": full_cycle_colors_by_attr,
    }

    for path_name in path_data_sources.keys():
        path_stats_summary[path_name] = {'total': 0, 'significant': 0}

    for attr in analysis_attributes:
        LOGGER.info(f"Processing attribute for RGB bin-pair comparisons: {attr}")
        # Base directory for all comparisons related to this attribute
        attr_base_dir = os.path.join(output_dir, f"{attr}{'_nocolor' if not color else ''}_RGB_BinComparisons")
        os.makedirs(attr_base_dir, exist_ok=True)

        results[attr] = results.get(attr, {})
        attribute_stats_summary[attr] = {'total': 0, 'significant': 0}
        results[attr]['input_distributions'] = input_colors_by_attr.get(attr, {}) # Store for context
        for path_name, source_dict in path_data_sources.items():
            results[attr][f'{path_name}_distributions'] = source_dict.get(attr, {})
            results[attr][f'bin_comparison_metrics_{path_name}'] = {} # Initialize storage

        bin_names = binning_config.get(attr, {}).get('bin_names', [])
        if len(bin_names) < 2:
            LOGGER.warning(f"Attribute '{attr}' has < 2 bins ({bin_names}). Skipping bin-pair comparisons.")
            continue
        bin_pairs = list(combinations(bin_names, 2))

        # --- Iterate through each PATH ---
        for path_name, path_data in path_data_sources.items():
            # Directory for this specific path's plots/heatmaps within the attribute dir
            path_comparison_dir = os.path.join(attr_base_dir, f"{path_name}_bin_comparisons")
            # No need to create it here, _compare_rgb_distributions_between_bins will do it

            if attr not in path_data:
                LOGGER.warning(f"    No data found for attribute '{attr}' in path '{path_name}'. Skipping.")
                continue
            path_data_for_attr = path_data[attr]

            bin_comparison_results_for_path = {}

            # --- Iterate through BIN PAIRS ---
            for bin1_name, bin2_name in bin_pairs:
                total_comparisons_overall += 1
                attribute_stats_summary[attr]['total'] += 1
                path_stats_summary[path_name]['total'] += 1

                # Compare RGB distributions between the two bins for this path
                metrics_per_channel = _compare_rgb_distributions_between_bins(
                    path_name=path_name,
                    attribute=attr,
                    bin1_name=bin1_name,
                    bin2_name=bin2_name,
                    path_data_for_attribute=path_data_for_attr,
                    output_dir_for_path_comparison=path_comparison_dir, # Pass dir for plots
                    channels=channels,
                    num_bins=num_bins
                )

                # Store results {('bin1','bin2'): {'R': {metrics}, 'G': {metrics}, 'B': {metrics}}}
                bin_comparison_results_for_path[(bin1_name, bin2_name)] = metrics_per_channel

                # Check significance: significant if *any* channel is significant
                is_significant = False
                if metrics_per_channel: # Check if comparison was successful
                    for ch in channels:
                        ch_metrics = metrics_per_channel.get(ch)
                        if ch_metrics:
                            ks_pval = ch_metrics.get('ks_pval')
                            if ks_pval is not None and np.isfinite(ks_pval) and ks_pval < significance_alpha:
                                is_significant = True
                                break # Found significance in one channel, no need to check others
                if is_significant:
                    total_significant_comparisons_overall += 1
                    attribute_stats_summary[attr]['significant'] += 1
                    path_stats_summary[path_name]['significant'] += 1

            # Store detailed results for this path
            results[attr][f'bin_comparison_metrics_{path_name}'] = bin_comparison_results_for_path

            # --- Generate Heatmaps per channel for the current path ---
            if bin_comparison_results_for_path and 'visualize_metric_heatmap_bin_pairs_rgb' in globals():
                os.makedirs(path_comparison_dir, exist_ok=True) # Ensure dir exists
                for ch in channels:
                    # Symmetric KL Heatmap for Channel ch
                    try:
                        visualize_metric_heatmap_bin_pairs_rgb(
                             metric_results=bin_comparison_results_for_path,
                             metric_key='kl_symmetric',
                             channel=ch, # Specify the channel
                             bin_names=bin_names,
                             attribute=attr,
                             path_name=path_name,
                             save_path=os.path.join(path_comparison_dir, f'{path_name}_{attr}_heatmap_KL_symmetric_{ch}.png'),
                             vmin=0
                         )
                    except Exception as e:
                         LOGGER.error(f"Failed to generate KL heatmap for {attr}, path {path_name}, channel {ch}: {e}", exc_info=True)

                    # KS p-value Heatmap for Channel ch
                    try:
                         visualize_metric_heatmap_bin_pairs_rgb(
                             metric_results=bin_comparison_results_for_path,
                             metric_key='ks_pval',
                             channel=ch, # Specify the channel
                             bin_names=bin_names,
                             attribute=attr,
                             path_name=path_name,
                             save_path=os.path.join(path_comparison_dir, f'{path_name}_{attr}_heatmap_KS_pvalue_{ch}.png'),
                             cmap='viridis_r',
                             vmin=0, vmax=1.0,
                             annotation_fmt=".3f"
                         )
                    except Exception as e:
                         LOGGER.error(f"Failed to generate KS p-value heatmap for {attr}, path {path_name}, channel {ch}: {e}", exc_info=True)
            elif not 'visualize_metric_heatmap_bin_pairs_rgb' in globals():
                 LOGGER.warning("`visualize_metric_heatmap_bin_pairs_rgb` function not found. Skipping heatmaps.")
            else:
                 LOGGER.warning(f"No valid bin comparison results for path '{path_name}', attribute '{attr}'. Skipping heatmaps.")


        # --- Visualize examples (outside path loop) ---
        if 'visualize_examples_by_attribute' in globals():
            if attr in examples_by_attr and any(examples_by_attr[attr].values()):
                try:
                     visualize_examples_by_attribute(
                         examples_by_attr[attr],
                         attr,
                         bin_names,
                         os.path.join(attr_base_dir, f'examples_by_{attr}.png')
                     )
                except Exception as e:
                     LOGGER.error(f"Failed to visualize examples for {attr}: {e}", exc_info=True)
            else:
                 LOGGER.warning(f"Skipping examples visualization for {attr} due to missing data.")
        else:
            LOGGER.warning("`visualize_examples_by_attribute` not found. Skipping examples.")

    # --- Assemble and Save Global Summary ---
    global_summary = {
        "analysis_type": "RGB Bin-Pair Comparison",
        "channels_analyzed": channels,
        "total_comparisons": total_comparisons_overall,
        "significant_comparisons": total_significant_comparisons_overall,
        "significance_alpha": significance_alpha,
        "significance_condition": f"KS p-value < {significance_alpha} in any channel ({', '.join(channels)})",
        "attribute_stats": attribute_stats_summary,
        "path_stats": path_stats_summary,
    }

    summary_file_path = os.path.join(output_dir, "rgb_within_model_comparison_summary.json")
    try:
        # Convert numpy types to standard Python types for JSON serialization
        def convert_numpy_types(obj):
             if isinstance(obj, np.integer): return int(obj)
             elif isinstance(obj, np.floating): return float(obj)
             elif isinstance(obj, np.ndarray): return obj.tolist()
             elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
             elif isinstance(obj, (list, tuple)): return [convert_numpy_types(i) for i in obj]
             return obj # Keep other types as is

        serializable_summary = convert_numpy_types(global_summary)

        with open(summary_file_path, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        LOGGER.info(f"RGB Within-model comparison summary saved to: {summary_file_path}")
    except TypeError as e:
         LOGGER.error(f"Failed to serialize RGB within-model summary to JSON: {e}. Summary structure might contain unsupported types. Summary head: {str(global_summary)[:500]}")
    except Exception as e:
         LOGGER.error(f"Failed to save RGB within-model comparison summary: {e}", exc_info=True)

    return results



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Updated RGB Shape Analyzer Class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class RGBShapeAnalyzer:
    """Class to analyze shape data across multiple attribute dimensions using R, G, B channels."""

    def __init__(
        self,
        global_workspace, # Assume this is the trained model/pipeline
        device: torch.device,
        shape_names: List[str] = ["diamond", "egg", "triangle"],
        channels: List[str] = ["R", "G", "B"], # Focus on RGB
        color: bool = True, # Controls if color attributes are used in input vectors (preprocessing)
        output_dir: str = ".",
        seed: int = 0,
        debug: bool = False,
        num_bins: int = 50, # Bins for histogram/metrics calculation
        reverb_n: int = 1, # Number of reverberation cycles for paths
    ):
        """Initialize the RGB shape analyzer."""
        self.global_workspace = global_workspace
        self.device = device
        self.shape_names = shape_names
        self.channels = channels # Store R, G, B
        self.color = color
        self.output_dir = output_dir
        self.seed = seed
        self.debug = debug
        self.num_bins = num_bins
        self.reverb_n = reverb_n
        os.makedirs(self.output_dir, exist_ok=True)
        self.rgb_colors, _ = generate_fixed_colors(100) # Use RGB colors
        self.binning_config = default_binning_config # Use the default config

    def _process_csv(
        self,
        csv_path: str,
        attributes: List[str], # analysis_attributes
        use_fixed_reference: bool = False,
        reference_color_idx: int = 0,
        im_dir: str = "./evaluation_set"
    ) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
        """
        Load CSV, preprocess samples, and process through the global workspace
        to get multiple path outputs.
        """
        df = pd.read_csv(csv_path)
        if isinstance(df["location"].iloc[0], str):
             df["location"] = df["location"].apply(ast.literal_eval)
        if use_fixed_reference:
            fixed_color = self.rgb_colors[reference_color_idx]
            df["fixed_color"] = [list(fixed_color)] * len(df)

        LOGGER.info(f"Preprocessing data from {csv_path}...")
        preprocessed_samples = preprocess_dataset(
            df,
            attributes,
            self.shape_names,
            self.color,
            self.rgb_colors,
            self.device,
            fixed_reference=use_fixed_reference,
            reference_color_idx=reference_color_idx,
            im_dir=im_dir
        )
        LOGGER.info(f"Processing {len(preprocessed_samples)} samples through global workspace (reverb_n={self.reverb_n})...")
        # Ensure process_through_global_workspace returns dicts with keys like:
        # 'input_shape_color', 'translated_shape_color', 'half_cycle_shape_color', 'full_cycle_shape_color',
        # 'input_image', 'translated_image', 'half_cycle_image', 'full_cycle_image' etc.
        processed_samples = process_through_global_workspace(
            self.global_workspace,
            preprocessed_samples,
            self.device,
            debug=self.debug,
            reverb_n=self.reverb_n, # Pass reverberation cycles
        )
        return df, preprocessed_samples, processed_samples

    def analyze_dataset(
        self,
        csv_path: str,
        analysis_attributes: List[str] = None,
        display_examples: bool = True,
        seed=None,
        binning_config=None,
        im_dir: str = "./evaluation_set",
        skip_processing: bool = False, # Option to skip processing if binned data exists
        significance_alpha: float = 0.05 # Allow setting alpha level
    ) -> Dict[str, Any]:
        """
        Analyze a dataset using R, G, B channels across multiple processing paths,
        focusing on comparisons between bins within each path.

        Args:
            csv_path (str): Path to the dataset CSV file.
            analysis_attributes (List[str]): Attributes to analyze. Defaults to standard set.
            display_examples (bool): Store and visualize example images per bin. Defaults to True.
            seed (int): Random seed. Defaults to instance seed.
            binning_config (Dict): Configuration for binning. Defaults to instance default.
            im_dir (str): Directory containing images referenced by CSV. Defaults to "./evaluation_set".
            skip_processing (bool): If True and binned results file exists, load it and skip CSV processing/binning.
            significance_alpha (float): P-value threshold for significance.

        Returns:
            Dict[str, Any]: Dictionary containing analysis results (binned distributions,
                            bin-pair comparison metrics per path, summary stats).
        """
        if analysis_attributes is None:
            analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        if binning_config is None:
            binning_config = self.binning_config
        if seed is None:
            seed = self.seed
        seed_everything(seed)

        results = {attr: {} for attr in analysis_attributes}
        LOGGER.info(f"Starting RGB analysis for {csv_path}")

        # Define paths for saving/loading binned data
        # Adapt save/load functions if necessary for RGB/multiple paths
        binned_data_filename = 'binned_rgb_results.pkl' # Specific filename for RGB results
        save_path = os.path.join(self.output_dir, binned_data_filename)

        # Initialize or Load Binned Data
        input_colors_by_attr = {}
        translated_colors_by_attr = {}
        half_cycle_colors_by_attr = {}
        full_cycle_colors_by_attr = {}
        examples_by_attr = {}

        if skip_processing and os.path.exists(save_path):
            LOGGER.info(f"Attempting to load existing binned RGB results from {save_path}")
            try:
                 # Ensure load_binned_results is adapted for RGB and multiple paths
                 # It should return the 5 dictionaries + config + attributes
                (
                    input_colors_by_attr,
                    translated_colors_by_attr,
                    half_cycle_colors_by_attr,
                    full_cycle_colors_by_attr,
                    examples_by_attr,
                    loaded_binning_config, # Load config used for the saved data
                    loaded_analysis_attributes # Load attributes used
                ) = load_binned_results(self.output_dir, filename=binned_data_filename) # Pass filename

                # Optional: Validate loaded config/attributes against current ones
                if binning_config != loaded_binning_config:
                     LOGGER.warning("Loaded binning config differs from current config. Using loaded config.")
                     binning_config = loaded_binning_config
                if set(analysis_attributes) != set(loaded_analysis_attributes):
                     LOGGER.warning("Loaded analysis attributes differ from current ones. Using loaded attributes.")
                     analysis_attributes = loaded_analysis_attributes
                LOGGER.info("Successfully loaded binned results.")

            except Exception as e:
                 LOGGER.error(f"Failed to load binned results from {save_path}: {e}. Proceeding with processing.", exc_info=True)
                 # Reset structures and ensure processing happens
                 input_colors_by_attr, translated_colors_by_attr, half_cycle_colors_by_attr, \
                 full_cycle_colors_by_attr, examples_by_attr = initialize_rgb_binning_structures(
                      analysis_attributes, binning_config, self.channels
                 )
                 skip_processing = False # Force processing since loading failed
        else:
            LOGGER.info("Initializing empty binning structures for RGB.")
            # Initialize fresh structures if not skipping or file doesn't exist
            input_colors_by_attr, translated_colors_by_attr, half_cycle_colors_by_attr, \
            full_cycle_colors_by_attr, examples_by_attr = initialize_rgb_binning_structures(
                analysis_attributes, binning_config, self.channels
            )


        # Process CSV and Bin Samples if not loaded from file
        if not skip_processing or not input_colors_by_attr: # Process if not skipping or if loading failed
            LOGGER.info("Processing CSV and binning samples...")
            df, preprocessed_samples, processed_samples = self._process_csv(
                csv_path,
                analysis_attributes,
                use_fixed_reference=False, # Standard analysis uses original colors
                im_dir=im_dir
            )

            LOGGER.info("Binning processed RGB samples across paths...")
            bin_rgb_processed_samples_with_paths(
                preprocessed_samples=preprocessed_samples,
                processed_samples=processed_samples,
                analysis_attributes=analysis_attributes,
                binning_config=binning_config,
                input_colors_by_attr=input_colors_by_attr,
                translated_colors_by_attr=translated_colors_by_attr,
                half_cycle_colors_by_attr=half_cycle_colors_by_attr,
                full_cycle_colors_by_attr=full_cycle_colors_by_attr,
                examples_by_attr=examples_by_attr,
                channels=self.channels,
                display_examples=display_examples
            )

            LOGGER.info("Binning complete. Saving RGB results...")
            # Save the results - ensure save_binned_results handles the 5 dicts + config + attributes
            try:
                save_binned_results(
                    self.output_dir,
                    input_colors_by_attr, translated_colors_by_attr,
                    half_cycle_colors_by_attr, full_cycle_colors_by_attr,
                    examples_by_attr, binning_config, analysis_attributes,
                    filename=binned_data_filename # Pass filename
                 )
            except Exception as e:
                 LOGGER.error(f"Failed to save binned RGB results: {e}", exc_info=True)


        LOGGER.info("Processing analysis attributes: Comparing RGB distributions between bins within paths...")
        # Call the rewritten analysis function
        results = process_analysis_attributes_rgb(
            analysis_attributes=analysis_attributes,
            output_dir=self.output_dir,
            color=self.color,
            channels=self.channels,
            binning_config=binning_config,
            input_colors_by_attr=input_colors_by_attr,
            translated_colors_by_attr=translated_colors_by_attr,
            half_cycle_colors_by_attr=half_cycle_colors_by_attr,
            full_cycle_colors_by_attr=full_cycle_colors_by_attr,
            examples_by_attr=examples_by_attr,
            results=results,
            num_bins=self.num_bins,
            significance_alpha=significance_alpha
        )

        LOGGER.info(f"RGB Analysis complete. Results saved in {self.output_dir}")
        return results


    def compare_color_distributions_across_shapes(
        self,
        csv_path: str,
        shape_names: List[str] = None,
        display_distributions: bool = True, # Controls overall plot generation
        display_ks_test: bool = True, # Controls KS heatmap
        display_kl_divergence: bool = True, # Controls KL heatmaps
        im_dir: str = "./evaluation_set",
        num_bins: int = 50 # Bins for metrics in this specific comparison
    ) -> Dict[str, Any]:
        """
        Compare *output* (translated) R, G, B distributions across different shapes.
        (This function is kept separate but updated to use R, G, B).

        Note: This uses only the 'translated' path implicitly via _process_csv
              if reverb_n=0 or just the final output otherwise. If comparison across
              paths is needed for shapes, use analyze_dataset.
        """
        output_dir = os.path.join(self.output_dir, "shape_rgb_comparison")
        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info(f"Comparing RGB distributions across shapes in: {output_dir}")

        if shape_names is None:
            shape_names = self.shape_names
        else:
            # Validate shape names
            shape_names = [s for s in shape_names if s in self.shape_names]
            if not shape_names:
                 LOGGER.error("No valid shape names provided or found in analyzer's list. Aborting comparison.")
                 return {}

        # Process CSV - note: this might re-run model inference.
        # Consider adapting to use already binned 'translated' data if available.
        df, preprocessed_samples, processed_samples = self._process_csv(
            csv_path,
            attributes=['shape'], # Only need shape attribute for this
            use_fixed_reference=False,
            im_dir=im_dir
            # Note: _process_csv uses self.reverb_n. The 'processed_samples' will contain
            # keys based on that (e.g., 'translated_shape_color', potentially others).
            # We will focus on 'translated_shape_color' as the "output" here.
        )

        shape_color_distributions = {shape_name: {ch: [] for ch in self.channels} for shape_name in shape_names}

        output_color_key = 'translated_shape_color' # Assuming this is the primary output key
        if self.reverb_n > 0:
             output_color_key = 'full_cycle_shape_color' # Or use full_cycle if reverb_n > 0? Choose one consistently.
             LOGGER.info(f"Using '{output_color_key}' for shape comparison as reverb_n > 0.")
        else:
             LOGGER.info(f"Using '{output_color_key}' for shape comparison.")


        for i, preprocessed_sample in enumerate(preprocessed_samples):
             shape_label = preprocessed_sample.get('shape') # Get shape name directly
             if shape_label in shape_names:
                 processed_sample = processed_samples[i]
                 color_val = processed_sample.get(output_color_key)

                 if not isinstance(color_val, (list, tuple)) or len(color_val) < 3:
                      LOGGER.warning(f"Sample {i} (shape {shape_label}): Unexpected format for '{output_color_key}': {color_val}. Skipping.")
                      continue

                 try:
                      # Extract R, G, B from the chosen output color key
                      r_val, g_val, b_val = float(color_val[0]), float(color_val[1]), float(color_val[2])
                      if "R" in self.channels: shape_color_distributions[shape_label]["R"].append(r_val)
                      if "G" in self.channels: shape_color_distributions[shape_label]["G"].append(g_val)
                      if "B" in self.channels: shape_color_distributions[shape_label]["B"].append(b_val)
                 except (IndexError, ValueError, TypeError) as e:
                      LOGGER.warning(f"Sample {i} (shape {shape_label}): Could not extract RGB from '{output_color_key}': {color_val}. Error: {e}. Skipping.")
                      continue


        # Basic visualization of distributions per shape (if function exists)
        if display_distributions and 'visualize_color_distributions_by_attribute' in globals():
             try:
                visualize_color_distributions_by_attribute(
                    shape_color_distributions, # Dict: shape -> channel -> values
                    "shape", # Attribute name
                    shape_names, # List of bins (shapes)
                    os.path.join(output_dir, 'shape_rgb_distributions.png'),
                    channels=self.channels # Specify channels
                )
             except Exception as e:
                LOGGER.error(f"Failed shape distribution visualization: {e}", exc_info=True)
        elif display_distributions:
             LOGGER.warning("`visualize_color_distributions_by_attribute` not found. Skipping visualization.")


        # --- KS and KL Calculations ---
        ks_test_results = {}
        kl_divergence_results = {}
        shape_pairs = list(combinations(shape_names, 2))

        for shape1, shape2 in shape_pairs:
            ks_test_results[(shape1, shape2)] = {}
            kl_divergence_results[(shape1, shape2)] = {}

            for ch in self.channels:
                dist1 = np.array([x for x in shape_color_distributions[shape1].get(ch, []) if not np.isnan(x)])
                dist2 = np.array([x for x in shape_color_distributions[shape2].get(ch, []) if not np.isnan(x)])

                if len(dist1) <= 1 or len(dist2) <= 1:
                    LOGGER.warning(f"Insufficient data for {shape1} vs {shape2}, channel {ch}. Skipping metrics.")
                    ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': np.nan, 'p_value': np.nan, 'error': 'Insufficient data'}
                    kl_divergence_results[(shape1, shape2)][ch] = {'kl_1_vs_2': np.nan, 'kl_2_vs_1': np.nan, 'kl_symmetric': np.nan, 'error': 'Insufficient data'}
                    continue

                try:
                     # Use comparison_metrics helper
                     kl_12, kl_21, kl_sym, ks_stat, ks_pval = comparison_metrics(dist1, dist2, num_bins)

                     ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': ks_stat, 'p_value': ks_pval}
                     kl_divergence_results[(shape1, shape2)][ch] = {'kl_1_vs_2': kl_12, 'kl_2_vs_1': kl_21, 'kl_symmetric': kl_sym}

                     LOGGER.info(f"Metrics ({shape1} vs {shape2}, Ch {ch}): Sym KL={kl_sym:.3f}, KS p-val={ks_pval:.3f}")

                except Exception as e:
                     LOGGER.error(f"Error calculating metrics for {shape1} vs {shape2}, Ch {ch}: {e}", exc_info=True)
                     ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': np.nan, 'p_value': np.nan, 'error': 'Calculation failed'}
                     kl_divergence_results[(shape1, shape2)][ch] = {'kl_1_vs_2': np.nan, 'kl_2_vs_1': np.nan, 'kl_symmetric': np.nan, 'error': 'Calculation failed'}


        # --- Heatmap Generation ---
        # Requires visualize_metric_heatmap_bin_pairs_rgb to be available
        if 'visualize_metric_heatmap_bin_pairs_rgb' in globals():
            # Reshape results for the heatmap function: {('s1','s2'): {'R':{metrics}, 'G':{metrics}}}
            heatmap_metrics = {}
            for pair, ch_results in kl_divergence_results.items():
                 heatmap_metrics[pair] = ch_results # Structure matches


            if display_ks_test and shape_pairs:
                for ch in self.channels:
                     try:
                         visualize_metric_heatmap_bin_pairs_rgb(
                             metric_results=heatmap_metrics,
                             metric_key='ks_pval',
                             channel=ch,
                             bin_names=shape_names,
                             attribute='shape',
                             path_name=output_color_key.replace('_shape_color',''), # e.g., 'translated'
                             save_path=os.path.join(output_dir, f'shape_comparison_heatmap_KS_pvalue_{ch}.png'),
                             cmap='viridis_r', vmin=0, vmax=1.0, annotation_fmt=".3f",
                             title=f"Shape Comparison KS p-value ({ch})"
                         )
                     except Exception as e:
                         LOGGER.error(f"Failed KS heatmap ({ch}) for shape comparison: {e}", exc_info=True)

            if display_kl_divergence and shape_pairs:
                for ch in self.channels:
                     try:
                         visualize_metric_heatmap_bin_pairs_rgb(
                             metric_results=heatmap_metrics,
                             metric_key='kl_symmetric',
                             channel=ch,
                             bin_names=shape_names,
                             attribute='shape',
                             path_name=output_color_key.replace('_shape_color',''),
                             save_path=os.path.join(output_dir, f'shape_comparison_heatmap_KL_symmetric_{ch}.png'),
                             vmin=0, annotation_fmt=".2f",
                             title=f"Shape Comparison Symmetric KL ({ch})"
                         )
                     except Exception as e:
                         LOGGER.error(f"Failed KL heatmap ({ch}) for shape comparison: {e}", exc_info=True)
        elif display_ks_test or display_kl_divergence:
            LOGGER.warning("`visualize_metric_heatmap_bin_pairs_rgb` not found. Skipping shape comparison heatmaps.")


        # --- Save Summary Text File ---
        summary_path = os.path.join(output_dir, 'shape_rgb_comparison_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write(f"Shape RGB Distribution Comparison Summary ({output_color_key})\n")
                f.write(f"Channels: {', '.join(self.channels)}\n")
                f.write(f"Num Bins for Metrics: {num_bins}\n")

                for (shape1, shape2), ch_results_ks in ks_test_results.items():
                    f.write(f"\nComparison: {shape1} vs {shape2}\n")
                    f.write("  KS Test Results (Statistic / p-value):\n")
                    for ch in self.channels:
                        ks_vals = ch_results_ks.get(ch, {})
                        stat = ks_vals.get('ks_statistic', 'N/A')
                        pval = ks_vals.get('p_value', 'N/A')
                        err = ks_vals.get('error')
                        if err: f.write(f"    {ch}: Error - {err}\n")
                        else: f.write(f"    {ch}: {stat:.3f} / {pval:.3f}{' (Sig.)' if isinstance(pval, float) and pval < 0.05 else ''}\n")

                    f.write("  KL Divergence Results (Symm KL / S1->S2 / S2->S1):\n")
                    ch_results_kl = kl_divergence_results.get((shape1, shape2), {})
                    for ch in self.channels:
                         kl_vals = ch_results_kl.get(ch, {})
                         kl_s = kl_vals.get('kl_symmetric', 'N/A')
                         kl_12 = kl_vals.get('kl_1_vs_2', 'N/A')
                         kl_21 = kl_vals.get('kl_2_vs_1', 'N/A')
                         err = kl_vals.get('error')
                         if err: f.write(f"    {ch}: Error - {err}\n")
                         else: f.write(f"    {ch}: {kl_s:.3f} / {kl_12:.3f} / {kl_21:.3f}\n")
            LOGGER.info(f"Shape comparison summary saved to: {summary_path}")
        except Exception as e:
             LOGGER.error(f"Failed to write shape comparison summary: {e}", exc_info=True)


        # Return structured results
        return {
            "shape_color_distributions": shape_color_distributions,
            "ks_test_results": ks_test_results,
            "kl_divergence_results": kl_divergence_results,
            "output_dir": output_dir
        }


# Example Placeholder Visualization Functions (Implement these based on your plotting needs)

def visualize_distribution_comparison_rgb(dist1_data, dist2_data, dist1_label, dist2_label, bin_name, attr, save_path, channels=["R", "G", "B"], num_bins=50, title=None):
    """Placeholder: Visualizes R,G,B distributions for two sets of data."""
    n_channels = len(channels)
    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 4), sharey=True)
    if n_channels == 1: axes = [axes] # Make iterable if only one channel

    if title is None:
        title = f"Distribution Comparison: {attr} - {bin_name}"
    fig.suptitle(title)

    for i, ch in enumerate(channels):
        ax = axes[i]
        vals1 = np.array([x for x in dist1_data.get(ch, []) if x is not None and not np.isnan(x)])
        vals2 = np.array([x for x in dist2_data.get(ch, []) if x is not None and not np.isnan(x)])

        if len(vals1) > 0 or len(vals2) > 0:
             min_val = 0 # Assume 0-255 range for RGB
             max_val = 255
             bins = np.linspace(min_val, max_val, num_bins + 1)
             if len(vals1) > 0:
                 ax.hist(vals1, bins=bins, alpha=0.6, label=f"{dist1_label} (n={len(vals1)})", density=True)
             if len(vals2) > 0:
                 ax.hist(vals2, bins=bins, alpha=0.6, label=f"{dist2_label} (n={len(vals2)})", density=True)
             ax.set_title(f"Channel: {ch}")
             ax.set_xlabel("Value")
             if i == 0: ax.set_ylabel("Density")
             ax.legend()
        else:
             ax.set_title(f"Channel: {ch} (No Data)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    try:
        plt.savefig(save_path)
        # LOGGER.debug(f"Saved RGB comparison plot to {save_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save RGB comparison plot {save_path}: {e}")
    plt.close(fig)


def visualize_metric_heatmap_bin_pairs_rgb(
    metric_results: Dict[Tuple[str, str], Optional[Dict[str, Dict[str, float]]]], # e.g., {('b1','b2'): {'R': {'kl_sym':..}, 'G':..}} or None
    metric_key: str, # e.g., 'kl_symmetric', 'ks_pval'
    channel: str, # Specific channel: 'R', 'G', or 'B'
    bin_names: List[str],
    attribute: str,
    path_name: str,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    invert_cmap: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotation_fmt: str = ".2f"
):
    """
    Visualizes a heatmap of a specific metric for a specific RGB channel,
    calculated between pairs of bins for a given attribute and path.
    Adapts the Hue version.
    """
    n_bins = len(bin_names)
    if n_bins < 2:
        LOGGER.warning(f"Skipping RGB heatmap for {attribute} ({path_name}, {channel}) - requires at least 2 bins.")
        return

    matrix = np.full((n_bins, n_bins), np.nan) # Initialize with NaNs

    for i in range(n_bins):
        for j in range(n_bins):
            if i == j:
                matrix[i, j] = 0.0 if 'kl' in metric_key or 'ks_stat' in metric_key else 1.0 if 'pval' in metric_key else np.nan
                continue

            bin_i = bin_names[i]
            bin_j = bin_names[j]
            # Get results for the pair, which contains dicts per channel {R:{...}, G:{...}, B:{...}} or is None
            pair_results_per_channel = metric_results.get((bin_i, bin_j), metric_results.get((bin_j, bin_i), None))

            value = np.nan # Default to NaN
            if pair_results_per_channel and channel in pair_results_per_channel:
                 channel_metrics = pair_results_per_channel[channel]
                 if channel_metrics and metric_key in channel_metrics:
                     metric_val = channel_metrics[metric_key]
                     # Check if metric_val is not None and is finite
                     if metric_val is not None and np.isfinite(metric_val):
                         value = metric_val
            matrix[i, j] = value


    if np.isnan(matrix).all():
        LOGGER.warning(f"Skipping RGB heatmap for {attribute} ({path_name}), metric '{metric_key}', channel '{channel}' - no valid data found.")
        return

    # Determine vmin/vmax if not provided (using only non-NaN values)
    valid_values = matrix[~np.isnan(matrix)]
    if vmin is None:
        # Handle case where all valid values might be the same (e.g., all 0 for KL)
        vmin = np.min(valid_values) if len(valid_values) > 0 else 0
    if vmax is None:
        if 'pval' in metric_key: vmax = 1.0
        else: vmax = np.max(valid_values) if len(valid_values) > 0 else 1
        # Prevent vmin == vmax if possible, unless all values are identical
        if vmin == vmax and len(np.unique(valid_values)) > 1 :
             vmax = vmin + 1e-6 # Add tiny offset if values aren't truly identical
        elif vmin == vmax:
             vmax = vmin + 1.0 # Add larger offset if all values are identical

    # Handle colormap inversion
    cmap_actual = cmap
    if invert_cmap:
        try:
            if not cmap.endswith("_r"): cmap_actual = cmap + "_r"
        except Exception: pass

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 0.8), max(5, n_bins * 0.7)))

    # Ensure vmin is not greater than vmax before plotting
    if vmin > vmax:
        LOGGER.warning(f"vmin ({vmin}) > vmax ({vmax}) for heatmap. Swapping them.")
        vmin, vmax = vmax, vmin
    elif vmin == vmax:
        # If they are equal after adjustments, add a small range to avoid imshow error
        vmax = vmin + 1e-6


    im = ax.imshow(matrix, cmap=cmap_actual, vmin=vmin, vmax=vmax, aspect='auto')


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
                # Determine text color based on background
                # Normalize value to 0-1 range relative to colormap limits
                norm_val = (value - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else 0.5
                # Simple threshold for text color inversion
                text_color = "white" if 0.2 < norm_val < 0.8 else "black" # Adjust threshold as needed

                # Highlight significant p-values?
                is_significant = 'pval' in metric_key and value < 0.05
                if is_significant:
                    text_color = "red" # Example highlight

                ax.text(j, i, f"{value:{annotation_fmt}}", ha="center", va="center", color=text_color,
                        fontweight='bold' if is_significant else 'normal')
            else:
                 ax.text(j, i, "N/A", ha="center", va="center", color="grey")


    # Add colorbar and title
    metric_label = f"{metric_key.replace('_', ' ').title()} ({channel})"
    fig.colorbar(im, ax=ax, label=metric_label)
    if title is None:
        title = f"{metric_label} between Bins\nAttribute: {attribute.title()}, Path: {path_name.title()}"
    ax.set_title(title, wrap=True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        # LOGGER.debug(f"Saved RGB heatmap to {save_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save RGB heatmap {save_path}: {e}")
    plt.close(fig)



if __name__=="main": 
    # Assuming 'model' is your loaded global_workspace and 'device' is set
    analyzer = RGBShapeAnalyzer(
        global_workspace=model,
        device=device,
        output_dir="./rgb_analysis_output",
        seed=42,
        reverb_n=1, # Or 0 if you only want input vs translated comparison implicitly
        num_bins=50, # For metrics/histograms
        debug=False
    )

    analysis_results = analyzer.analyze_dataset(
        csv_path="evaluation_set/attributes.csv",
        im_dir="evaluation_set",
        display_examples=True,
        significance_alpha=0.05,
        skip_processing=False # Set to True to try loading saved binned data
    )

    # Optionally run the shape-specific comparison
    shape_comparison_results = analyzer.compare_color_distributions_across_shapes(
        csv_path="evaluation_set/attributes.csv",
        im_dir="evaluation_set",
        shape_names=["diamond", "egg"] # Optional subset
    )
# The analysis_results will contain the binned distributions, metrics, and paths to saved plots.
# analysis_results will contain the detailed metrics and paths to plots/heatmaps.