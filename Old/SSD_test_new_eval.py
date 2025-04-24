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

# Assuming these imports are correct and available in the environment
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

# from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
# from simple_shapes_dataset.cli import generate_image, get_transformed_coordinates
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
    process_through_global_workspace, # Assuming this exists
    rgb_to_hsv,
)
from SSD_visualize_functions import (
    visualize_color_distributions_by_attribute,
    visualize_input_output_distributions, # Should be adaptable or replaced
    visualize_kl_heatmap,
    visualize_examples_by_attribute
)

# Mock LOGGER if not imported
import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

default_binning_config = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 4,
        'range': (0, 2 * np.pi),
        'bin_names': ['0-90', '90-180', '180-270', '270-360'] # Adjusted for clarity
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
# Helper function to perform comparison for a single bin and path
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _compare_distributions_for_bin(
    input_dist: Dict[str, List[float]],
    output_dist: Dict[str, List[float]],
    bin_name: str,
    attr: str,
    comparison_label: str, # e.g., "input_vs_translated", "input_vs_half_cycle"
    output_dir_for_comparison: str,
) -> Optional[Dict[str, float]]:
    """
    Compares input and output distributions (H channel) for a single bin,
    visualizes, calculates KL divergence and KS test.

    Args:
        input_dist: Dictionary containing input 'H' values for the bin.
        output_dist: Dictionary containing output 'H' values (e.g., translated, half_cycle) for the bin.
        bin_name: Name of the current bin.
        attr: Name of the attribute being analyzed.
        comparison_label: String identifying the comparison (e.g., "input_vs_translated").
        output_dir_for_comparison: Directory to save visualizations for this specific comparison.

    Returns:
        Dictionary with KL divergence for 'H' channel, or None if data is insufficient.
    """
    # Extract H channel values, handling potential missing data and NaNs
    input_values = np.array([x for x in input_dist.get('H', []) if not np.isnan(x)])
    output_values = np.array([x for x in output_dist.get('H', []) if not np.isnan(x)])

    if len(input_values) <= 1 or len(output_values) <= 1:
        LOGGER.warning(f"Skipping bin '{bin_name}' for {attr} ({comparison_label}) - insufficient valid data (Input: {len(input_values)}, Output: {len(output_values)})")
        return None

    LOGGER.info(f"  - Comparing {comparison_label} for bin '{bin_name}': {len(input_values)} input vs {len(output_values)} output H values.")

    # Visualize comparison
    os.makedirs(output_dir_for_comparison, exist_ok=True)
    plot_filename = os.path.join(
        output_dir_for_comparison,
        f'{comparison_label}_{bin_name.replace("/", "_")}.png'
    )
    visualize_input_output_distributions(
        input_dist, # Pass the original dict which might contain other channels if needed by viz func
        output_dist,
        bin_name,
        attr,
        plot_filename,
        channels=["H"], # Focus on H channel
        title_suffix=f"({comparison_label.replace('_', ' ').title()})" # Add comparison type to title
    )

    # Calculate KL divergence and K-S test for H channel
    kl_h = kl_divergence(input_values, output_values)
    ks_stat, ks_pval = ks_2samp(input_values, output_values)

    LOGGER.info(f"    KL divergence (H): {kl_h:.4f}")
    LOGGER.info(f"    K-S test (H): stat={ks_stat:.4f}, p={ks_pval:.4f}")

    return {'H': kl_h}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modified process_analysis_attributes function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def process_analysis_attributes(
    analysis_attributes: List[str],
    output_dir: str,
    color: bool, # Keep for directory naming consistency
    binning_config: Dict,
    input_colors_by_attr: Dict,
    translated_colors_by_attr: Dict, # Renamed from output_colors_by_attr
    half_cycle_colors_by_attr: Dict,
    full_cycle_colors_by_attr: Dict,
    examples_by_attr: Dict,
    results: Dict, # Pass results dict to update it
) -> Dict:
    """
    Processes analysis attributes for multiple paths (translated, half-cycle, full-cycle).

    Calculates KL divergence between input and each output path's H channel distributions,
    visualizes distributions and examples, and stores results.

    Args:
        analysis_attributes: List of attributes to analyze (e.g., ['shape', 'rotation']).
        output_dir: Base directory for saving results.
        color: Boolean indicating if color was used (affects directory naming).
        binning_config: Dictionary defining bins for each attribute.
        input_colors_by_attr: Dict mapping attr -> bin_name -> {'H': [values...]}.
        translated_colors_by_attr: Dict for translated path results.
        half_cycle_colors_by_attr: Dict for half-cycle path results.
        full_cycle_colors_by_attr: Dict for full-cycle path results.
        examples_by_attr: Dict mapping attr -> bin_name -> [example_dict].
        results: Dictionary to store the analysis results.

    Returns:
        The updated results dictionary.
    """
    for attr in analysis_attributes:
        LOGGER.info(f"Processing attribute: {attr}")
        # Main directory for the attribute
        attr_base_dir = os.path.join(output_dir, f"{attr}{'_nocolor' if not color else ''}")
        os.makedirs(attr_base_dir, exist_ok=True)

        # Subdirectories for each comparison path
        dir_in_vs_translated = os.path.join(attr_base_dir, "input_vs_translated")
        dir_in_vs_half = os.path.join(attr_base_dir, "input_vs_half_cycle")
        dir_in_vs_full = os.path.join(attr_base_dir, "input_vs_full_cycle")

        # Initialize KL divergence dictionaries for this attribute
        kl_in_vs_translated = {}
        kl_in_vs_half = {}
        kl_in_vs_full = {}

        # Ensure the attribute exists in the results dictionary
        results[attr] = results.get(attr, {})
        results[attr]['input_distributions'] = input_colors_by_attr.get(attr, {})
        results[attr]['translated_distributions'] = translated_colors_by_attr.get(attr, {})
        results[attr]['half_cycle_distributions'] = half_cycle_colors_by_attr.get(attr, {})
        results[attr]['full_cycle_distributions'] = full_cycle_colors_by_attr.get(attr, {})

        # Iterate through bins defined for the current attribute
        bin_names = binning_config.get(attr, {}).get('bin_names', [])
        if not bin_names:
            LOGGER.warning(f"No bin names found for attribute '{attr}' in binning_config. Skipping.")
            continue

        for bin_name in bin_names:
            LOGGER.debug(f"Processing bin: {bin_name} for attribute: {attr}")

            # Get input distribution for the current bin
            input_dist_bin = input_colors_by_attr.get(attr, {}).get(bin_name, {})
            if not input_dist_bin.get('H'):
                 LOGGER.warning(f"No input H data for bin '{bin_name}', attribute '{attr}'. Skipping comparisons for this bin.")
                 continue


            # --- 1. Compare Input vs. Translated ---
            translated_dist_bin = translated_colors_by_attr.get(attr, {}).get(bin_name, {})
            if translated_dist_bin.get('H'):
                kl_result = _compare_distributions_for_bin(
                    input_dist=input_dist_bin,
                    output_dist=translated_dist_bin,
                    bin_name=bin_name,
                    attr=attr,
                    comparison_label="input_vs_translated",
                    output_dir_for_comparison=dir_in_vs_translated,
                )
                if kl_result:
                    kl_in_vs_translated[bin_name] = kl_result
            else:
                LOGGER.warning(f"No translated H data for bin '{bin_name}', attribute '{attr}'.")


            # --- 2. Compare Input vs. Half-Cycle ---
            half_cycle_dist_bin = half_cycle_colors_by_attr.get(attr, {}).get(bin_name, {})
            if half_cycle_dist_bin.get('H'):
                kl_result = _compare_distributions_for_bin(
                    input_dist=input_dist_bin,
                    output_dist=half_cycle_dist_bin,
                    bin_name=bin_name,
                    attr=attr,
                    comparison_label="input_vs_half_cycle",
                    output_dir_for_comparison=dir_in_vs_half,
                )
                if kl_result:
                    kl_in_vs_half[bin_name] = kl_result
            else:
                 LOGGER.warning(f"No half-cycle H data for bin '{bin_name}', attribute '{attr}'.")


            # --- 3. Compare Input vs. Full-Cycle ---
            full_cycle_dist_bin = full_cycle_colors_by_attr.get(attr, {}).get(bin_name, {})
            if full_cycle_dist_bin.get('H'):
                kl_result = _compare_distributions_for_bin(
                    input_dist=input_dist_bin,
                    output_dist=full_cycle_dist_bin,
                    bin_name=bin_name,
                    attr=attr,
                    comparison_label="input_vs_full_cycle",
                    output_dir_for_comparison=dir_in_vs_full,
                )
                if kl_result:
                    kl_in_vs_full[bin_name] = kl_result
            else:
                LOGGER.warning(f"No full-cycle H data for bin '{bin_name}', attribute '{attr}'.")

        # --- Visualizations after processing all bins for the attribute ---

        # Visualize examples (only needs to be done once per attribute)
        if attr in examples_by_attr and any(examples_by_attr[attr].values()): # Check if there are any examples for this attr
            visualize_examples_by_attribute(
                examples_by_attr[attr],
                attr,
                bin_names, # Pass the bin names for correct labeling
                os.path.join(attr_base_dir, f'examples_by_{attr}.png'),
            )
        else:
            LOGGER.warning(f"Skipping examples visualization for {attr} due to missing examples data.")

        # Visualize KL heatmaps for each comparison path
        if kl_in_vs_translated:
            visualize_kl_heatmap(
                kl_in_vs_translated,
                attr,
                bin_names,
                os.path.join(dir_in_vs_translated, f'kl_heatmap_input_vs_translated.png'),
                channels=["H"],
                title=f"KL Divergence (Input vs Translated H) by {attr.capitalize()}"
            )
        else:
             LOGGER.warning(f"Skipping KL heatmap (Input vs Translated) for {attr} as no KL divergence was calculated.")

        if kl_in_vs_half:
            visualize_kl_heatmap(
                kl_in_vs_half,
                attr,
                bin_names,
                os.path.join(dir_in_vs_half, f'kl_heatmap_input_vs_half_cycle.png'),
                channels=["H"],
                title=f"KL Divergence (Input vs Half-Cycle H) by {attr.capitalize()}"

            )
        else:
            LOGGER.warning(f"Skipping KL heatmap (Input vs Half-Cycle) for {attr} as no KL divergence was calculated.")

        if kl_in_vs_full:
            visualize_kl_heatmap(
                kl_in_vs_full,
                attr,
                bin_names,
                os.path.join(dir_in_vs_full, f'kl_heatmap_input_vs_full_cycle.png'),
                channels=["H"],
                title=f"KL Divergence (Input vs Full-Cycle H) by {attr.capitalize()}"
            )
        else:
            LOGGER.warning(f"Skipping KL heatmap (Input vs Full-Cycle) for {attr} as no KL divergence was calculated.")

        # Store KL divergence results for the attribute
        results[attr]['kl_input_vs_translated'] = kl_in_vs_translated
        results[attr]['kl_input_vs_half_cycle'] = kl_in_vs_half
        results[attr]['kl_input_vs_full_cycle'] = kl_in_vs_full
        LOGGER.info(f"Finished processing attribute: {attr}")

    return results


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
        seed=0
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

        os.makedirs(self.output_dir, exist_ok=True)
        # Use a fixed set of colors for reproducibility if needed
        self.rgb_colors, self.hls_colors = generate_fixed_colors(100)
        self.binning_config = default_binning_config # Use the default config

    # Assuming preprocess_dataset exists and works as intended
    def _preprocess_dataset(
        self,
        df: Any,
        analysis_attributes: List[str], # Note: these are just for reference here
        shape_names: List[str],
        color_mode: bool,
        rgb_colors: np.ndarray,
        device: torch.device,
        fixed_reference: bool = False,
        reference_color_idx: int = 0,
        im_dir: str = "./evaluation_set" # Added default
    ) -> List[dict]:
         # Use the imported preprocess_dataset directly
        return preprocess_dataset(
             df=df,
             analysis_attributes=analysis_attributes, # Pass it along
             shape_names=shape_names,
             color_mode=color_mode,
             rgb_colors=rgb_colors,
             device=device,
             fixed_reference=fixed_reference,
             reference_color_idx=reference_color_idx,
             im_dir=im_dir
         )

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
        preprocessed_samples = self._preprocess_dataset(
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
        df, preprocessed_samples, processed_samples = self._process_csv(
            csv_path,
            analysis_attributes, # Pass attributes for preprocessing reference
            use_fixed_reference=False, # Standard analysis uses original colors
            im_dir=im_dir
        )

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
            binning_config
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
            output_colors_by_attr=translated_colors_by_attr, # Pass the correct target dict
            half_cycle_colors_by_attr=half_cycle_colors_by_attr, # Pass the correct target dict
            full_cycle_colors_by_attr=full_cycle_colors_by_attr, # Pass the correct target dict
            examples_by_attr=examples_by_attr,
            display_examples=display_examples
        )

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
            results=results # Pass the results dict to be updated
        )

        LOGGER.info(f"Analysis complete. Results saved in {self.output_dir}")
        return results