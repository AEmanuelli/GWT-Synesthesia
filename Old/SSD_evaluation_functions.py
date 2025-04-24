from collections.abc import Mapping, Sequence
from pathlib import Path
import os
import math
import ast
import io
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

from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid

from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

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

from shimmer_ssd import DEBUG_MODE, LOGGER, PROJECT_DIR
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig, load_config
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

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
from simple_shapes_dataset.cli import generate_image, get_transformed_coordinates
from SSD_utils import (
    generate_fixed_colors,
    normalize_position,
    normalize_size,
    normalize_rotation,
    kl_divergence,
    bin_attribute,
    segment_shape,
    extract_shape_color,
    initialize_binning_structures,
    bin_processed_samples,
    preprocess_dataset,
    process_through_global_workspace,
    safe_extract_channel
)
from SSD_visualize_functions import (
    visualize_color_distributions_by_attribute,
    visualize_input_output_distributions,
    visualize_kl_heatmap,
    visualize_examples_by_attribute
)

default_binning_config = {
    'shape': {
        'n_bins': 3,
        'range': None,  # Not applicable for categorical data
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 4,
        'range': (0, 2 * np.pi),
        'bin_names': ['0°', '90°', '180°', '270°']  # Matches [0, π/2, π, 3π/2]
    },
    'size': {
        'n_bins': 4,  # Matches n_sizes=4 in the function
        'range': (7, 14),  # Matches min_scale=7, max_scale=14
        'bin_names': ['Very Small', 'Small', 'Medium', 'Large']
    },
    'position_x': {
        'n_bins': 2,
        'range': (0, 32),
        'bin_names': ['Left', 'Right']  # For the corners at margin and imsize-margin
    },
    'position_y': {
        'n_bins': 2,
        'range': (0, 32),
        'bin_names': ['Bottom', 'Top']  # For the corners at margin and imsize-margin
    }
}

def initialize_binning_structures(
    analysis_attributes: List[str],
    binning_config: dict,
    channels: List[str]
) -> Tuple[dict, dict, dict]:
    """
    Initialize the data structures for storing binned results.
    
    Args:
        analysis_attributes: List of attributes to analyze.
        binning_config: Configuration for binning.
        channels: List of color channels.
        
    Returns:
        Three dictionaries for input colors, output colors, and examples.
    """
    input_colors_by_attr = {}
    output_colors_by_attr = {}
    examples_by_attr = {}
    
    for attr in analysis_attributes:
        input_colors_by_attr[attr] = {}
        output_colors_by_attr[attr] = {}
        examples_by_attr[attr] = {}
        
        bin_names = binning_config[attr]['bin_names']
        for bin_name in bin_names:
            input_colors_by_attr[attr][bin_name] = {ch: [] for ch in channels}
            output_colors_by_attr[attr][bin_name] = {ch: [] for ch in channels}
            examples_by_attr[attr][bin_name] = []
    
    return input_colors_by_attr, output_colors_by_attr, examples_by_attr



def compare_bin_in_out(
    bin_name,
    attr,
    channels,
    input_colors_by_attr,
    output_colors_by_attr,
    attr_dir,
    binning_config,
    kl_in_out=None,  # Pass kl_in_out as argument
):
    """
    Processes a single bin for a given attribute, checks for data, visualizes if needed,
    calculates KL divergence, and updates kl_in_out dictionary.

    Args:
        bin_name (str): Name of the bin being processed.
        attr (str): Name of the attribute being analyzed.
        channels (list): List of channels to analyze.
        input_colors_by_attr (dict): Input color distributions by attribute and bin.
        output_colors_by_attr (dict): Output color distributions by attribute and bin.
        attr_dir (str): Directory to save attribute-specific output files.
        binning_config (dict): Configuration for binning attributes.
        kl_in_out (dict, optional): Dictionary to store KL divergence results. Defaults to None (will be initialized if None).

    Returns:
        dict: Updated kl_in_out dictionary.
    """
    if kl_in_out is None:
        kl_in_out = {}

    has_data = False
    for ch in channels:
        input_values = [x for x in input_colors_by_attr.get(attr, {}).get(bin_name, {}).get(ch, []) if not np.isnan(x)]
        output_values = [x for x in output_colors_by_attr.get(attr, {}).get(bin_name, {}).get(ch, []) if not np.isnan(x)]
        if len(input_values) > 1 and len(output_values) > 1:
            has_data = True
            break
    if not has_data:
        LOGGER.warning(f"Skipping bin {bin_name} for {attr} - no valid data")
        return kl_in_out  # Return even if no data

    for ch in channels:
        if attr in input_colors_by_attr and bin_name in input_colors_by_attr[attr] and ch in input_colors_by_attr[attr][bin_name]: # Safe access
            values = input_colors_by_attr[attr][bin_name][ch]
            valid_values = [x for x in values if not np.isnan(x)]
            print(f"  - Channel {ch}: {len(valid_values)} valid values out of {len(values)} total")

    if attr in input_colors_by_attr and bin_name in input_colors_by_attr[attr] and attr in output_colors_by_attr and bin_name in output_colors_by_attr[attr]: # Safe check before calling visualization
        visualize_input_output_distributions(
            input_colors_by_attr[attr][bin_name],
            output_colors_by_attr[attr][bin_name],
            bin_name,
            attr,
            os.path.join(attr_dir, f'input_vs_output_{bin_name.replace("/", "_")}.png')
        )
    else:
        LOGGER.warning(f"Skipping input_vs_output visualization for {attr} - {bin_name} due to missing data.")

    bin_kl = {}
    ks_stat = {}
    ks_pval = {}
    print("___________________________________________________________________________________")
    for ch in channels:
        input_values = np.array([x for x in input_colors_by_attr.get(attr, {}).get(bin_name, {}).get(ch, []) if not np.isnan(x)])
        output_values = np.array([x for x in output_colors_by_attr.get(attr, {}).get(bin_name, {}).get(ch, []) if not np.isnan(x)])
        if len(input_values) > 1 and len(output_values) > 1:
            bin_kl[ch] = kl_divergence(input_values, output_values) # Assuming kl_divergence is defined
            LOGGER.info(f"KL divergence for bin {bin_name}, channel {ch}: {bin_kl[ch]}")
            ks_stat[ch], ks_pval[ch] = ks_2samp(input_values, output_values) # Assuming ks_2samp is defined
            LOGGER.info(f"K-S test for bin {bin_name}, channel {ch}: stat={ks_stat[ch]}, p={ks_pval[ch]}")
    if bin_kl:
        kl_in_out[bin_name] = bin_kl

    return kl_in_out



def process_analysis_attributes(
    analysis_attributes,
    output_dir,
    color,
    channels,
    binning_config,
    input_colors_by_attr,
    output_colors_by_attr,
    examples_by_attr,
    results,
):
    """
    Processes analysis attributes, calculates KL divergence between input and output distributions,
    visualizes distributions and examples, and stores results.

    Args:
        analysis_attributes (list): List of attributes to analyze.
        output_dir (str): Base directory to save output files.
        color (bool): Flag indicating whether to include 'nocolor' suffix in output directories.
        channels (list): List of channels to analyze.
        binning_config (dict): Configuration for binning attributes.
                                 Expected structure: {attr: {'bin_names': list}}
        input_colors_by_attr (dict): Dictionary of input color distributions by attribute and bin.
                                     Expected structure: {attr: {bin_name: {channel: list}}}
        output_colors_by_attr (dict): Dictionary of output color distributions by attribute and bin.
                                      Expected structure: {attr: {bin_name: {channel: list}}}
        examples_by_attr (dict): Dictionary of examples by attribute and bin.
                                 Expected structure: {attr: {bin_name: list}} (structure of list depends on visualize_examples_by_attribute)
        results (dict): Dictionary to store results. Will be updated in place.
                        Expected structure (initially): {}
        display_examples (bool, optional): Whether to display and save example visualizations. Defaults to False.

    Returns:
        dict: The updated `results` dictionary. (Note: it's also updated in place)
    """

    for attr in analysis_attributes:
        attr_dir = os.path.join(output_dir, f"{attr}{'_nocolor' if not color else ''}")
        os.makedirs(attr_dir, exist_ok=True)

        kl_in_out = {}
        results[attr] = results.get(attr, {}) # Initialize if attribute not already in results.
        results[attr]['kl_input_output'] = {}
        results[attr]['input_distributions'] = {}
        results[attr]['output_distributions'] = {}


        for bin_idx, bin_name in enumerate(binning_config[attr]['bin_names']):
            compare_bin = compare_bin_in_out(
                bin_name=bin_name,
                attr=attr,
                channels=channels,
                input_colors_by_attr=input_colors_by_attr,
                output_colors_by_attr=output_colors_by_attr,
                attr_dir=attr_dir,
                binning_config=binning_config,
                kl_in_out=kl_in_out  # Pass and update kl_in_out
            )



        if attr in examples_by_attr: # Safe check before visualization
            # Assuming visualize_examples_by_attribute is defined elsewhere
            visualize_examples_by_attribute(
                examples_by_attr[attr],
                attr,
                binning_config[attr]['bin_names'],
                os.path.join(attr_dir, f'examples_by_{attr}.png')
            )
        else:
            LOGGER.warning(f"Skipping examples_by_attribute visualization for {attr} due to missing examples data.")

        if kl_in_out: # Only visualize heatmap if there is KL data
            # Assuming visualize_kl_heatmap is defined elsewhere
            visualize_kl_heatmap(
                kl_in_out,
                attr,
                binning_config[attr]['bin_names'],
                os.path.join(attr_dir, f'kl_input_output_heatmap.png')
            )
        else:
            LOGGER.warning(f"Skipping KL heatmap visualization for {attr} as no KL divergence was calculated.")


        results[attr]['kl_input_output'] = kl_in_out
        results[attr]['input_distributions'] = input_colors_by_attr.get(attr, {}) # Safe access
        results[attr]['output_distributions'] = output_colors_by_attr.get(attr, {}) # Safe access

    return results


class ShapeAnalyzer:
    """Class to analyze shape data across multiple attribute dimensions."""

    def __init__(
        self,
        global_workspace,
        device: torch.device,
        shape_names: List[str] = ["diamond", "egg", "triangle"],
        channels: List[str] = ["H", "R", "G", "B"],
        color: bool = True,
        output_dir: str = ".",
        seed=0
    ):
        """
        Initialize the shape analyzer.
        """
        self.global_workspace = global_workspace
        self.device = device
        self.shape_names = shape_names
        self.color = color
        self.output_dir = output_dir
        self.channels = channels
        self.seed = seed

        os.makedirs(self.output_dir, exist_ok=True)
        self.rgb_colors, self.hls_colors = generate_fixed_colors(100)#, seed=self.seed)
        self.binning_config = default_binning_config

    def _process_csv(
        self,
        csv_path: str,
        attributes: List[str],
        use_fixed_reference: bool = False,
        reference_color_idx: int = 0
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """
        Load the CSV file, preprocess all samples, and process them through the global workspace.
        """
        df = pd.read_csv(csv_path)
        df["location"] = df["location"].apply(ast.literal_eval)
        if use_fixed_reference:
            fixed_color = self.rgb_colors[reference_color_idx]
            df["fixed_color"] = [fixed_color] * len(df)
        preprocessed_samples = preprocess_dataset(
            df,
            attributes,
            self.shape_names,
            self.color,
            self.rgb_colors,
            self.device,
            fixed_reference=use_fixed_reference
        )
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
        binning_config=None
    ) -> Dict[str, Any]:
        """
        Analyze a dataset of shape images.
        """
        if analysis_attributes is None:
            analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        if binning_config is None:
            binning_config = self.binning_config
        if seed is None:
            seed = self.seed

        results = {attr: {} for attr in analysis_attributes}
        df, preprocessed_samples, processed_samples = self._process_csv(csv_path, analysis_attributes, use_fixed_reference=False)


        input_colors_by_attr, output_colors_by_attr, examples_by_attr = initialize_binning_structures(
            analysis_attributes,
            binning_config,
            self.channels
        )
        bin_processed_samples(
            preprocessed_samples=preprocessed_samples,
            processed_samples=processed_samples,
            analysis_attributes=analysis_attributes,
            binning_config=binning_config,
            input_colors_by_attr=input_colors_by_attr,
            output_colors_by_attr=output_colors_by_attr,
            examples_by_attr=examples_by_attr,
            channels=self.channels,
            display_examples=display_examples
        )

        results = process_analysis_attributes(
            analysis_attributes,
            self.output_dir,
            self.color,
            self.channels,
            binning_config,
            input_colors_by_attr,
            output_colors_by_attr,
            examples_by_attr,
            results
        )

        return results

    def compare_color_distributions_across_shapes(
        self,
        csv_path: str,
        shape_names: List[str] = None,
        display_distributions: bool = True,
        display_ks_test: bool = True,
        display_kl_divergence: bool = True
    ) -> Dict[str, Any]:
        """
        Compare reconstructed color distributions across different shapes using KS tests and KL divergence.
        """
        output_dir = os.path.join(self.output_dir, "shape_color_comparison")
        os.makedirs(output_dir, exist_ok=True)
        if shape_names is None:
            shape_names = self.shape_names
        else:
            for shape_name in shape_names:
                if shape_name not in self.shape_names:
                    raise ValueError(f"Shape '{shape_name}' not in the analyzer's shape list: {self.shape_names}")

        df, _, processed_samples = self._process_csv(csv_path, attributes=['shape'], use_fixed_reference=False)
        shape_color_distributions = {shape_name: {ch: [] for ch in self.channels} for shape_name in shape_names}

        for index, row in df.iterrows():
            shape_label = self.shape_names[int(row['class'])]  # Assuming 'class' column represents shape index
            if shape_label in shape_names:
                processed_sample = processed_samples[index]
                for ch in self.channels:
                    # Use safe extraction to avoid missing keys
                    value = safe_extract_channel(processed_sample, ch, index)
                    shape_color_distributions[shape_label][ch].append(value)

        visualize_color_distributions_by_attribute(
            shape_color_distributions,
            "shape",
            shape_names,
            os.path.join(output_dir, f'shape_color_distributions.png')
        )

        # KS test calculations
        ks_test_results = {}
        # KL divergence calculations
        kl_divergence_results = {}
        
        shape_pairs = []
        for i in range(len(shape_names)):
            for j in range(i + 1, len(shape_names)):
                shape_pairs.append((shape_names[i], shape_names[j]))

        for shape1, shape2 in shape_pairs:
            ks_test_results[(shape1, shape2)] = {}
            kl_divergence_results[(shape1, shape2)] = {}
            average_dist1 = np.mean([shape_color_distributions[shape1][ch] for ch in self.channels], axis = 0)
            average_dist2 = np.mean([shape_color_distributions[shape2][ch] for ch in self.channels], axis = 0)
            ks_stat, ks_pval = ks_2samp(average_dist1, average_dist2)
            LOGGER.info(f"KS test for shapes {shape1} vs {shape2}: stat={ks_stat:.3f}, p={ks_pval:.3f}")
            print("========================================================================================")
            print(f"KS test for averaged channels shapes {shape1} vs {shape2}: stat={ks_stat:.3f}, p={ks_pval:.3f}")

            for ch in self.channels:

                dist1 = np.array([x for x in shape_color_distributions[shape1][ch] if not np.isnan(x)])
                dist2 = np.array([x for x in shape_color_distributions[shape2][ch] if not np.isnan(x)])
                
                if len(dist1) > 1 and len(dist2) > 1:
                    # Bin the distributions into histograms
                    num_bins = 50
                    # Create histograms and normalize them to create proper probability distributions
                    hist1, bin_edges = np.histogram(dist1, bins=num_bins, range=(0, 255), density=True)
                    hist2, _ = np.histogram(dist2, bins=num_bins, range=(0, 255), density=True)
                    
                    # Get the bin centers for each bin
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Create arrays with bin values repeated according to their frequencies
                    # This allows us to use ks_2samp which expects sample data, not histograms
                    binned_dist1 = np.repeat(bin_centers, np.round(hist1 * 1000).astype(int))
                    binned_dist2 = np.repeat(bin_centers, np.round(hist2 * 1000).astype(int))
                    
                    # Normalize for KS test (KS test assumes values between 0-1)
                    binned_dist1_norm = binned_dist1 / 255.0
                    binned_dist2_norm = binned_dist2 / 255.0
                    
                    # KS test on binned distributions
                    ks_stat, ks_pval = ks_2samp(binned_dist1_norm, binned_dist2_norm)
                    ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': ks_stat, 'p_value': ks_pval}
                    LOGGER.info(f"KS test for shapes {shape1} vs {shape2}, channel {ch}: stat={ks_stat:.3f}, p={ks_pval:.3f}")
                    
                    # KL divergence calculations (in both directions since KL is asymmetric)
                    kl_1_to_2 = kl_divergence(dist1, dist2)
                    kl_2_to_1 = kl_divergence(dist2, dist1)
                    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2 
                    kl_divergence_results[(shape1, shape2)][ch] = {
                        'kl_1_to_2': kl_1_to_2,
                        'kl_2_to_1': kl_2_to_1,
                        'kl_symmetric': (kl_1_to_2 + kl_2_to_1) / 2  # Symmetric version (Jensen-Shannon inspired)
                    }
                    LOGGER.info(f"Symmetric KL divergence for shapes {shape1} <-> {shape2}, channel {ch}: {symmetric_kl:.3f}")
                    # LOGGER.info(f"KL divergence for shapes {shape2} -> {shape1}, channel {ch}: {kl_2_to_1:.3f}")
                else:
                    ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': None, 'p_value': None, 'error': 'Insufficient data'}
                    kl_divergence_results[(shape1, shape2)][ch] = {'kl_1_to_2': None, 'kl_2_to_1': None, 'kl_symmetric': None, 'error': 'Insufficient data'}
                    LOGGER.warning(f"Insufficient data to compare shapes {shape1} and {shape2} for channel {ch}")

        # Display KS test heatmap
        if display_ks_test and shape_pairs:
            ks_pval_matrix = np.zeros((len(shape_names), len(shape_names)))
            for shape1, shape2 in shape_pairs:
                row_idx = shape_names.index(shape1)
                col_idx = shape_names.index(shape2)
                pair_p_values = [ks_test_results[(shape1, shape2)][ch]['p_value']
                                for ch in self.channels
                                if ks_test_results[(shape1, shape2)][ch]['p_value'] is not None]
                if pair_p_values:
                    avg_p_value = np.mean(pair_p_values)
                    ks_pval_matrix[row_idx, col_idx] = avg_p_value
                    ks_pval_matrix[col_idx, row_idx] = avg_p_value  # Symmetric matrix

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(ks_pval_matrix, cmap='viridis', vmin=0, vmax=0.05)
            ax.set_xticks(np.arange(len(shape_names)))
            ax.set_yticks(np.arange(len(shape_names)))
            ax.set_xticklabels(shape_names)
            ax.set_yticklabels(shape_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(shape_names)):
                for j in range(len(shape_names)):
                    if i < j and (shape_names[i], shape_names[j]) in ks_test_results:
                        pair_values = [ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value']
                                    for ch in self.channels
                                    if ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value'] is not None]
                        avg_p_val_text = f"{np.mean(pair_values):.3f}" if pair_values else "N/A"
                        text_color = "w" if np.mean(pair_values) < 0.05 else "black" if pair_values else "black"
                        ax.text(j, i, avg_p_val_text, ha="center", va="center", color=text_color)
                    elif i > j:
                        pair_values = [ks_test_results[(shape_names[j], shape_names[i])][ch]['ks_statistic']
                                    for ch in self.channels
                                    if ks_test_results[(shape_names[j], shape_names[i])][ch]['ks_statistic'] is not None]
                        avg_ks_stat_text = f"{np.mean(pair_values):.2f}" if pair_values else "N/A"
                        ax.text(j, i, avg_ks_stat_text, ha="center", va="center", color="black")
            ax.set_title("KS Test p-values (Upper Triangle) and KS Statistics (Lower Triangle) for Color Distribution Comparison Across Shapes")
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shape_color_ks_test_heatmap.png'))
            plt.close()
            
        # Display KL divergence heatmap
        if display_kl_divergence and shape_pairs:
            # Create a symmetric KL divergence matrix
            kl_matrix = np.zeros((len(shape_names), len(shape_names)))
            
            for shape1, shape2 in shape_pairs:
                row_idx = shape_names.index(shape1)
                col_idx = shape_names.index(shape2)
                
                symmetric_kl_values = [kl_divergence_results[(shape1, shape2)][ch]['kl_symmetric']
                                    for ch in self.channels
                                    if kl_divergence_results[(shape1, shape2)][ch]['kl_symmetric'] is not None]
                
                if symmetric_kl_values:
                    avg_kl = np.mean(symmetric_kl_values)
                    kl_matrix[row_idx, col_idx] = avg_kl
                    kl_matrix[col_idx, row_idx] = avg_kl  # Make it symmetric
            
            # Plot KL divergence heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            # Custom color map to highlight differences (higher KL values = more different distributions)
            max_kl = np.max(kl_matrix) if np.max(kl_matrix) > 0 else 1.0
            im = ax.imshow(kl_matrix, cmap='viridis', vmin=0, vmax=max_kl)
            
            ax.set_xticks(np.arange(len(shape_names)))
            ax.set_yticks(np.arange(len(shape_names)))
            ax.set_xticklabels(shape_names)
            ax.set_yticklabels(shape_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add KL values to the cells
            for i in range(len(shape_names)):
                for j in range(len(shape_names)):
                    if i != j:  # Skip the diagonal
                        kl_value = kl_matrix[i, j]
                        if kl_value > 0:
                            text_color = "w" if kl_value > max_kl/2 else "black"
                            ax.text(j, i, f"{kl_value:.2f}", ha="center", va="center", color=text_color)
            
            ax.set_title("KL Divergence Between Shape Color Distributions (Averaged Across Channels)")
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shape_color_kl_divergence_heatmap.png'))
            plt.close()
            
            # Additional visualization: directional KL divergence (asymmetric)
            # For each channel, create a separate heatmap
            for ch_idx, ch in enumerate(self.channels):
                kl_dir_matrix = np.zeros((len(shape_names), len(shape_names)))
                
                for i, shape1 in enumerate(shape_names):
                    for j, shape2 in enumerate(shape_names):
                        if i != j:
                            if i < j:  # Upper triangle
                                kl_value = kl_divergence_results.get((shape1, shape2), {}).get(ch, {}).get('kl_1_to_2')
                            else:  # Lower triangle
                                kl_value = kl_divergence_results.get((shape2, shape1), {}).get(ch, {}).get('kl_2_to_1')
                                
                            if kl_value is not None:
                                kl_dir_matrix[i, j] = kl_value
                
                fig, ax = plt.subplots(figsize=(8, 6))
                max_kl_ch = np.max(kl_dir_matrix) if np.max(kl_dir_matrix) > 0 else 1.0
                im = ax.imshow(kl_dir_matrix, cmap='viridis', vmin=0, vmax=max_kl_ch)
                
                ax.set_xticks(np.arange(len(shape_names)))
                ax.set_yticks(np.arange(len(shape_names)))
                ax.set_xticklabels(shape_names)
                ax.set_yticklabels(shape_names)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add KL values to the cells
                for i in range(len(shape_names)):
                    for j in range(len(shape_names)):
                        if i != j:  # Skip the diagonal
                            kl_value = kl_dir_matrix[i, j]
                            if kl_value > 0:
                                text_color = "w" if kl_value > max_kl_ch/2 else "black"
                                ax.text(j, i, f"{kl_value:.2f}", ha="center", va="center", color=text_color)
                
                ax.set_title(f"Directional KL Divergence for Channel {ch} (from row to column)")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shape_color_kl_divergence_{ch}_heatmap.png'))
                plt.close()
                
        # Add KL divergence results to summary file
        summary_path = os.path.join(output_dir, 'shape_color_comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Shape Color Distribution Comparison Summary\n")
            for (shape1, shape2), results_dict in ks_test_results.items():
                f.write(f"\nComparison between {shape1} and {shape2}:\n")
                f.write("KS Test Results:\n")
                for ch, values in results_dict.items():
                    if 'error' in values:
                        f.write(f"  Channel {ch}: {values['error']}\n")
                    else:
                        f.write(f"  Channel {ch}: KS Statistic = {values['ks_statistic']:.3f}, p-value = {values['p_value']:.3f}")
                        if values['p_value'] < 0.05:
                            f.write(" (Significant difference)")
                        f.write("\n")
                
                f.write("\nKL Divergence Results:\n")
                for ch, values in kl_divergence_results[(shape1, shape2)].items():
                    if 'error' in values:
                        f.write(f"  Channel {ch}: {values['error']}\n")
                    else:
                        f.write(f"  Channel {ch}: {shape1} -> {shape2} = {values['kl_1_to_2']:.3f}, "
                                f"{shape2} -> {shape1} = {values['kl_2_to_1']:.3f}, "
                                f"Symmetric = {values['kl_symmetric']:.3f}\n")

        return {
            "shape_color_distributions": shape_color_distributions,
            "ks_test_results": ks_test_results,
            "kl_divergence_results": kl_divergence_results,
            "output_dir": output_dir
        }