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
from SSD_utils import generate_fixed_colors, normalize_position, normalize_size, normalize_rotation, kl_divergence, bin_attribute, segment_shape, extract_shape_color#, process_dataset
from SSD_utils import * # TO BE MODIFIED 

import pickle
default_binning_config = {
        'shape': {
            'n_bins': 3,
            'range': None,  # Not applicable for categorical data
            'bin_names': ['diamond', 'egg', 'triangle']
        },
        'rotation': {
            'n_bins': 4,
            'range': (0, 2*np.pi),
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
            'bin_names': ['Top', 'Bottom']  # For the corners at margin and imsize-margin
        }
    }



def visualize_examples_by_attribute(
    examples: Dict[str, Dict[int, List[np.ndarray]]],
    attribute_name: str,
    bin_names: List[str],
    output_path: str,
    max_per_bin: int = 5
) -> None:
    """
    Create a visualization of examples grouped by attribute bins.

    Args:
        examples: Dictionary mapping attribute values to example images
        attribute_name: Name of the attribute being visualized
        bin_names: Names for each bin
        output_path: Path to save the visualization
        max_per_bin: Maximum number of examples to show per bin
    """
    # Filter out bins with no examples
    valid_bins = []
    valid_bin_names = []
    for bin_idx, bin_name in enumerate(bin_names):
        if bin_name in examples and len(examples[bin_name]) > 0:
            valid_bins.append(bin_idx)
            valid_bin_names.append(bin_name)
    
    n_valid_bins = len(valid_bins)
    
    # If no valid bins, just return
    if n_valid_bins == 0:
        print(f"Warning: No valid examples for {attribute_name}")
        return
    
    # Determine number of examples to show per bin (limited by available examples)
    examples_per_bin = {
        bin_idx: min(max_per_bin, len(examples.get(bin_names[bin_idx], [])))
        for bin_idx in valid_bins
    }

    max_examples = max(list(examples_per_bin.values()))  # Avoid empty figure

    fig, axes = plt.subplots(n_valid_bins, max_examples, figsize=(max_examples*2, n_valid_bins*2))
    if n_valid_bins == 1:
        axes = [axes]  # Handle 1D array case

    for out_idx, bin_idx in enumerate(valid_bins):
        bin_name = bin_names[bin_idx]
        for ex_idx in range(max_examples):
            ax = axes[out_idx][ex_idx] if max_examples > 1 else axes[out_idx]

            if bin_name in examples and ex_idx < len(examples[bin_name]):
                ax.imshow(examples[bin_name][ex_idx])
                if ex_idx == 0:
                    ax.set_ylabel(bin_name)

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"Example images by {attribute_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_color_distributions_by_attribute(
    data: Dict[str, Dict[str, List[float]]],
    attribute_name: str,
    bin_names: List[str],
    output_path: str
) -> None:
    """
    Create a visualization of color distributions grouped by attribute bins.

    Args:
        data: Dictionary mapping bins to color channel data
        attribute_name: Name of the attribute being visualized
        bin_names: Names for each bin
        output_path: Path to save the visualization
    """
    channels = ["R", "G", "B"]
    
    # Filter out bins with no data
    valid_bins = []
    valid_bin_names = []
    for bin_idx, bin_name in enumerate(bin_names):
        if bin_name in data and any(channel in data[bin_name] and len(data[bin_name][channel]) > 0 
                                   for channel in channels):
            valid_bins.append(bin_idx)
            valid_bin_names.append(bin_name)
    
    n_valid_bins = len(valid_bins)
    
    # If no valid bins, just return
    if n_valid_bins == 0:
        print(f"Warning: No valid data for {attribute_name} color distributions")
        return

    fig, axes = plt.subplots(n_valid_bins, len(channels), figsize=(len(channels)*3, n_valid_bins*2))
    if n_valid_bins == 1:
        axes = [axes]  # Handle 1D array case

    for out_idx, bin_idx in enumerate(valid_bins):
        bin_name = bin_names[bin_idx]
        for ch_idx, channel in enumerate(channels):
            ax = axes[out_idx][ch_idx]

            if bin_name in data and channel in data[bin_name] and len(data[bin_name][channel]) > 0:
                # Filter out NaN values
                filtered_data = [x for x in data[bin_name][channel] if not np.isnan(x)]
                if len(filtered_data) > 0:
                    ax.hist(filtered_data, bins=20, alpha=0.7)
                    if ch_idx == 0:
                        ax.set_ylabel(bin_name)
                    ax.set_title(f"Channel {channel}")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')

            ax.set_xlim(0, 1)

    plt.suptitle(f"Color distributions by {attribute_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def visualize_input_output_distributions(
    input_data: Dict[str, List[float]],
    output_data: Dict[str, List[float]],
    bin_name: str,
    attribute_name: str,
    output_path: str
) -> None:
    """
    Create a visualization comparing input and output color distributions for a specific bin.

    Args:
        input_data: Dictionary mapping color channels to input distribution data
        output_data: Dictionary mapping color channels to output distribution data
        bin_name: Name of the bin being visualized
        attribute_name: Name of the attribute being visualized
        output_path: Path to save the visualization
    """
    channels = ["R", "G", "B"]
    
    # Check if there's any valid data
    has_valid_data = False
    for channel in channels:
        input_valid = channel in input_data and len([x for x in input_data[channel] if not np.isnan(x)]) > 0
        output_valid = channel in output_data and len([x for x in output_data[channel] if not np.isnan(x)]) > 0
        if input_valid or output_valid:
            has_valid_data = True
            break
    
    # If no valid data, just return
    if not has_valid_data:
        print(f"Warning: No valid data for {attribute_name} - {bin_name}")
        return
    
    fig, axes = plt.subplots(1, len(channels), figsize=(len(channels)*4, 3))

    for ch_idx, channel in enumerate(channels):
        ax = axes[ch_idx]

        # Plot input distribution
        if channel in input_data and len(input_data[channel]) > 0:
            # Filter out NaN values
            filtered_input = np.array([x for x in input_data[channel] if not np.isnan(x)])
            if len(filtered_input) > 0:
                hist_in, bin_edges = np.histogram(filtered_input, bins=20, range=(0, 1), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.bar(bin_centers, hist_in, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, label="Input", color='blue')

        # Plot output distribution
        if channel in output_data and len(output_data[channel]) > 0:
            # Filter out NaN values
            filtered_output = np.array([x for x in output_data[channel] if not np.isnan(x)])
            if len(filtered_output) > 0:
                hist_out, bin_edges = np.histogram(filtered_output, bins=20, range=(0, 1), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.bar(bin_centers, hist_out, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, label="Output", color='red')

        ax.set_title(f"Channel {channel}")
        ax.set_xlim(0, 1)
        ax.legend()

    plt.suptitle(f"{attribute_name} - {bin_name}: Input vs Output Color Distributions")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def calculate_kl_input_output(
    input_data: Dict[str, List[float]],
    output_data: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Calculate KL divergence between input and output distributions for each color channel.

    Args:
        input_data: Dictionary mapping color channels to input distribution data
        output_data: Dictionary mapping color channels to output distribution data

    Returns:
        Dictionary mapping channels to KL divergence values
    """
    channels = ["R", "G", "B"]
    results = {}

    for channel in channels:
        if (channel in input_data and channel in output_data and
            len(input_data[channel]) > 0 and len(output_data[channel]) > 0):
            results[channel] = kl_divergence(
                input_data[channel],
                output_data[channel]
            )
        else:
            results[channel] = float('nan')

    return results


def visualize_kl_heatmap(
    kl_values: Dict[str, Dict[str, float]],
    attribute_name: str,
    bin_names: List[str],
    output_path: str
) -> None:
    """
    Create a heatmap visualization of KL divergence between input and output distributions.

    Args:
        kl_values: Dictionary mapping bins to channel KL values
        attribute_name: Name of the attribute being visualized
        bin_names: Names for each bin
        output_path: Path to save the visualization
    """
    channels = ["R", "G", "B"]
    
    # Filter out bins with no data
    valid_bins = []
    valid_bin_names = []
    for bin_idx, bin_name in enumerate(bin_names):
        if bin_name in kl_values and any(channel in kl_values[bin_name] and not np.isnan(kl_values[bin_name][channel]) 
                                         for channel in channels):
            valid_bins.append(bin_idx)
            valid_bin_names.append(bin_name)
    
    n_valid_bins = len(valid_bins)
    
    # If no valid data, just return
    if n_valid_bins == 0:
        print(f"Warning: No valid data for {attribute_name} KL divergence heatmap")
        return
    
    # Create a matrix for the heatmap with only valid bins
    kl_matrix = np.full((n_valid_bins, len(channels)), np.nan)

    # Fill the matrix with KL values
    for out_idx, bin_idx in enumerate(valid_bins):
        bin_name = bin_names[bin_idx]
        if bin_name in kl_values:
            for ch_idx, channel in enumerate(channels):
                if channel in kl_values[bin_name] and not np.isnan(kl_values[bin_name][channel]):
                    kl_matrix[out_idx, ch_idx] = kl_values[bin_name][channel]

    # Create the figure and heatmap
    fig, ax = plt.subplots(figsize=(8, n_valid_bins * 0.8 + 2))
    
    # Filter out NaN values for calculating the color scale
    valid_data = kl_matrix[~np.isnan(kl_matrix)]
    if len(valid_data) > 0:
        vmin, vmax = np.min(valid_data), np.max(valid_data)
        im = ax.imshow(kl_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        # This shouldn't happen after our previous checks, but just in case
        im = ax.imshow(np.zeros((n_valid_bins, len(channels))), cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("KL Divergence")

    # Add bin and channel labels
    ax.set_xticks(np.arange(len(channels)))
    ax.set_yticks(np.arange(n_valid_bins))
    ax.set_xticklabels(channels)
    ax.set_yticklabels(valid_bin_names)

    # Add title and labels
    ax.set_title(f"KL Divergence: Input vs Output Color Distributions for {attribute_name}")

    # Add text annotations in each cell
    for i in range(n_valid_bins):
        for j in range(len(channels)):
            if not np.isnan(kl_matrix[i, j]):
                # Use mean of valid data for color threshold
                mean_val = np.nanmean(valid_data) if len(valid_data) > 0 else 0
                text = ax.text(j, i, f"{kl_matrix[i, j]:.2f}",
                              ha="center", va="center", 
                              color="w" if kl_matrix[i, j] > mean_val/2 else "black")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


# class ShapeAnalyzer:
#     """Class to analyze shape data across multiple attribute dimensions."""

#     def __init__(
#         self,
#         global_workspace,
#         device: torch.device,
#         shape_names: List[str] = None,
#         color: bool = True,
#         output_dir: str = ".", 
#         seed = 0
#     ):
#         """
#         Initialize the shape analyzer.

#         Args:
#             global_workspace: Global workspace model
#             device: Torch device for computation
#             shape_names: Names of shape classes
#             color: Whether to include color information
#             output_dir: Directory to save outputs
#         """
#         self.global_workspace = global_workspace
#         self.device = device
#         self.shape_names = shape_names or ["diamond", "egg", "triangle"]
#         self.color = color
#         self.output_dir = output_dir
#         self.channels = ["R", "G", "B"]
#         self.seed = seed

#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)

#         # Initialize random colors with consistent seed
#         self.rgb_colors, _ = generate_fixed_colors(100, seed=0)

#         # Attribute binning configuration
#         self.binning_config = default_binning_config

#     def bin_data_point(self, row: pd.Series) -> Dict[str, int]:
#         """
#         Assign a data point to bins for each attribute.

#         Args:
#             row: Pandas Series with shape attributes

#         Returns:
#             Dictionary mapping attribute names to bin indices
#         """
#         bins = {}

#         # Shape class is already an index
#         bins['shape'] = int(row['class'])

#         # Bin rotation
#         rotation = float(row['rotation'])
#         bins['rotation'] = bin_attribute(
#             rotation,
#             self.binning_config['rotation']['n_bins'],
#             self.binning_config['rotation']['range']
#         )

#         # Bin size
#         size = float(row['size'])
#         bins['size'] = bin_attribute(
#             size,
#             self.binning_config['size']['n_bins'],
#             self.binning_config['size']['range']
#         )

#         # Bin position
#         pos = row['location']
#         bins['position_x'] = bin_attribute(
#             pos[0],
#             self.binning_config['position_x']['n_bins'],
#             self.binning_config['position_x']['range']
#         )
#         bins['position_y'] = bin_attribute(
#             pos[1],
#             self.binning_config['position_y']['n_bins'],
#             self.binning_config['position_y']['range']
#         )

#         return bins
    
#     def analyze_dataset(
#         self,
#         csv_path: str,
#         analysis_attributes: List[str] = None,
#         display_examples: bool = True, 
#         seed = 0,
#         binning_config = None
#     ) -> Dict[str, Any]:
#         """
#         Analyze a dataset of shape images through the global workspace model.
#         Reorganized into three clear steps:
#         1. Preprocess data
#         2. Process through Global Workspace
#         3. Bin and analyze results
        
#         Args:
#             csv_path: Path to CSV file containing shape metadata
#             analysis_attributes: List of attributes to analyze
#             display_examples: Whether to generate visualizations
#             seed: Random seed for reproducibility
#             binning_config: Configuration for binning (uses default_binning_config if None)
            
#         Returns:
#             Dictionary containing analysis results
#         """
#         # Set default analysis attributes if none provided
#         if analysis_attributes is None:
#             analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        
#         # Use default binning config if none provided
#         if binning_config is None:
#             binning_config = default_binning_config
            
#         # Initialize results structure
#         results = {attr: {} for attr in analysis_attributes}
        
#         # Initialize color mapping
#         rgb_colors, _ = generate_fixed_colors(100, seed=self.seed)
        
#         # =============================================
#         # STEP 1: PREPROCESS DATA
#         # =============================================
        
#         # Load and parse dataset
#         df = pd.read_csv(csv_path)
#         df["location"] = df["location"].apply(ast.literal_eval)
        
#         # Prepare the samples without binning
#         preprocessed_samples = preprocess_dataset(
#             df, 
#             analysis_attributes, 
#             self.shape_names, 
#             self.color, 
#             rgb_colors, 
#             self.device
#         )
        
#         # =============================================
#         # STEP 2: PROCESS THROUGH GLOBAL WORKSPACE
#         # =============================================
        
#         # Process all samples through the Global Workspace model
#         processed_samples = process_through_global_workspace(
#             self.global_workspace,
#             preprocessed_samples,
#             self.device,
#             self.channels
#         )

#         mean_reconstructed_colors = processed_samples[-1]


#         #save processed samples list of dictionary
#         with open('mean_reconstructed_colors.pkl', 'wb') as f:
#             pickle.dump(mean_reconstructed_colors, f)
        
        
#         # =============================================
#         # STEP 3: BIN DATA AND ANALYZE RESULTS
#         # =============================================
        
#         # Initialize data structures for binned results
#         input_colors_by_attr, output_colors_by_attr, examples_by_attr = initialize_binning_structures(
#             analysis_attributes,
#             binning_config,
#             self.channels
#         )
        
#         # Bin the processed samples
#         bin_processed_samples(
#             preprocessed_samples,
#             processed_samples,
#             analysis_attributes,
#             binning_config,
#             input_colors_by_attr,
#             output_colors_by_attr,
#             examples_by_attr,
#             self.channels,
#             display_examples
#         )
        
#         # Analyze bins for each attribute
#         for attr in analysis_attributes:
#             # Create output directory
#             attr_dir = os.path.join(self.output_dir, f"{attr}{'_nocolor' if not self.color else ''}")
#             os.makedirs(attr_dir, exist_ok=True)
            
#             # Visualize examples for each bin
#             if display_examples:
#                 visualize_examples_by_attribute(
#                     examples_by_attr[attr],
#                     attr,
#                     binning_config[attr]['bin_names'],
#                     os.path.join(attr_dir, f'examples_by_{attr}.png')
#                 )
            
#             # Calculate KL divergence between input and output for each bin
#             kl_in_out = {}
            
#             for bin_idx, bin_name in enumerate(binning_config[attr]['bin_names']):
#                 # Check if bin has data
#                 has_data = False
#                 for ch in self.channels:
#                     input_values = [x for x in input_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)]
#                     output_values = [x for x in output_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)]
#                     if len(input_values) > 1 and len(output_values) > 1:
#                         has_data = True
#                         break
                
#                 if not has_data:
#                     print(f"Skipping bin {bin_name} for {attr} - no valid data")
#                     continue
                
#                 # Visualize input vs output distributions for each bin
#                 if display_examples:
#                     visualize_input_output_distributions(
#                         input_colors_by_attr[attr][bin_name],
#                         output_colors_by_attr[attr][bin_name],
#                         bin_name,
#                         attr,
#                         os.path.join(attr_dir, f'input_vs_output_{bin_name.replace("/", "_")}.png')
#                     )
                    
#                 bin_kl = {}
#                 ks_stat = {}
#                 ks_pval = {}
#                 for ch in self.channels:
#                     # Filter arrays for valid data
#                     input_values = np.array([x for x in input_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)])
#                     output_values = np.array([x for x in output_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)])
                    
#                     if len(input_values) > 1 and len(output_values) > 1:
#                         bin_kl[ch] = kl_divergence(input_values, output_values)
#                         print(f"Case : {self.color}, KL Divergence between input and output distributions for bin {bin_name} and channel {ch}: {bin_kl[ch]}")
#                         ks_stat[ch], ks_pval[ch] = ks_2samp(input_values, output_values)
#                         print(f"Case: {self.color}, K-S test between input and output distributions for bin {bin_name} and channel {ch}: {ks_stat[ch]}, {ks_pval[ch]}")
                
#                 if bin_kl:  # Only add if at least one channel has data
#                     kl_in_out[bin_name] = bin_kl
            
#             # Create heatmap visualization of KL divergence values
#             if display_examples:
#                 visualize_kl_heatmap(
#                     kl_in_out,
#                     attr,
#                     binning_config[attr]['bin_names'],
#                     os.path.join(attr_dir, f'kl_input_output_heatmap.png')
#                 )
                
#             # Store results
#             results[attr]['kl_input_output'] = kl_in_out
#             results[attr]['input_distributions'] = input_colors_by_attr[attr]
#             results[attr]['output_distributions'] = output_colors_by_attr[attr]
        
#         return results


# def evaluate_dataset(
#     csv_path: str,
#     global_workspace,
#     device: torch.device,
#     analysis_attributes: List[str] = None,
#     color: bool = True,
#     display_examples: bool = True,
#     output_dir: str = "."
# ) -> Dict[str, Any]:
#     """
#     Evaluate a dataset of shape images with multidimensional analysis.

#     This function uses the ShapeAnalyzer class to analyze a dataset of shape images,
#     grouping them by different attributes and comparing the color distributions 
#     between input and output for each bin.

#     Args:
#         csv_path: Path to CSV file containing shape metadata.
#         global_workspace: Global workspace model to be evaluated.
#         device: Torch device to use for computation.
#         analysis_attributes: List of attributes to analyze (e.g., ['shape', 'rotation']).
#                            If None, defaults to all available attributes.
#         color: Whether to include color information in the analysis.
#         display_examples: Whether to generate visualizations of example images and color distributions.
#         output_dir: Directory to save the analysis results and visualizations.

#     Returns:
#         A dictionary containing the analysis results, including input-output distributions
#         and KL divergence for color distributions for each bin.
#     """
#     analyzer = ShapeAnalyzer(
#         global_workspace=global_workspace,
#         device=device,
#         color=color,
#         output_dir=output_dir
#     )
#     results = analyzer.analyze_dataset(
#         csv_path=csv_path,
#         analysis_attributes=analysis_attributes,
#         display_examples=display_examples
#     )
#     return results


class ShapeAnalyzer:
    """Class to analyze shape data across multiple attribute dimensions."""
    
    def __init__(
        self,
        global_workspace,
        device: torch.device,
        shape_names: List[str] = None,
        color: bool = True,
        output_dir: str = ".",
        seed=0
    ):
        """
        Initialize the shape analyzer.
        """
        self.global_workspace = global_workspace
        self.device = device
        self.shape_names = shape_names or ["diamond", "egg", "triangle"]
        self.color = color
        self.output_dir = output_dir
        self.channels = ["R", "G", "B"]
        self.seed = seed

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize random colors with a consistent seed
        self.rgb_colors, _ = generate_fixed_colors(100, seed=self.seed)

        # Attribute binning configuration
        self.binning_config = default_binning_config

    def _process_csv(
        self,
        csv_path: str,
        attributes: List[str],
        use_fixed_reference: bool = False,
        reference_color_idx: int = 0
    ) -> Tuple[pd.DataFrame, List[Any]]:
        """
        Load the CSV file, preprocess all samples, and process them through the global workspace.
        If use_fixed_reference is True, then the preprocessing uses a fixed reference color.
        """
        df = pd.read_csv(csv_path)
        df["location"] = df["location"].apply(ast.literal_eval)
        
        if use_fixed_reference:
            # For consistency testing, we override the color info by adding a fixed color column.
            fixed_color = self.rgb_colors[reference_color_idx]
            df["fixed_color"] = [fixed_color] * len(df)
        
        # Preprocess the dataset.
        # (Assume that the preprocessing function can accept an optional flag to use the fixed reference.)
        preprocessed_samples = preprocess_dataset(
            df,
            attributes,
            self.shape_names,
            self.color,
            self.rgb_colors,
            self.device,
            fixed_reference=use_fixed_reference
        )
        
        # Process all samples through the global workspace in one pass.
        processed_samples = process_through_global_workspace(
            self.global_workspace,
            preprocessed_samples,
            self.device,
            self.channels
        )
        
        return df, processed_samples

    def analyze_dataset(
        self,
        csv_path: str,
        analysis_attributes: List[str] = None,
        display_examples: bool = True,
        seed=None,
        binning_config=None
    ) -> Dict[str, Any]:
        """
        Analyze a dataset of shape images. This function now uses a single pass
        through the global workspace (via _process_csv) to obtain processed samples.
        """
        if analysis_attributes is None:
            analysis_attributes = ['shape', 'rotation', 'size', 'position_x', 'position_y']
        if binning_config is None:
            binning_config = self.binning_config
        if seed is None:
            seed = self.seed

        results = {attr: {} for attr in analysis_attributes}
        
        # Single pass: load, preprocess, and process all samples.
        df, processed_samples = self._process_csv(csv_path, analysis_attributes, use_fixed_reference=False)
        
        # Initialize structures for binning.
        input_colors_by_attr, output_colors_by_attr, examples_by_attr = initialize_binning_structures(
            analysis_attributes,
            binning_config,
            self.channels
        )
        
        # Bin the processed samples (assumes that bin_processed_samples works with the processed samples list).
        bin_processed_samples(
            preprocessed_samples=None,  # Not needed now because processed_samples are cached.
            processed_samples=processed_samples,
            analysis_attributes=analysis_attributes,
            binning_config=binning_config,
            input_colors_by_attr=input_colors_by_attr,
            output_colors_by_attr=output_colors_by_attr,
            examples_by_attr=examples_by_attr,
            channels=self.channels,
            display_examples=display_examples
        )
        
        # Analyze each attribute’s bins.
        for attr in analysis_attributes:
            attr_dir = os.path.join(self.output_dir, f"{attr}{'_nocolor' if not self.color else ''}")
            os.makedirs(attr_dir, exist_ok=True)
            
            if display_examples:
                visualize_examples_by_attribute(
                    examples_by_attr[attr],
                    attr,
                    binning_config[attr]['bin_names'],
                    os.path.join(attr_dir, f'examples_by_{attr}.png')
                )
            
            kl_in_out = {}
            for bin_idx, bin_name in enumerate(binning_config[attr]['bin_names']):
                has_data = False
                for ch in self.channels:
                    input_values = [x for x in input_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)]
                    output_values = [x for x in output_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)]
                    if len(input_values) > 1 and len(output_values) > 1:
                        has_data = True
                        break
                if not has_data:
                    print(f"Skipping bin {bin_name} for {attr} - no valid data")
                    continue

                if display_examples:
                    visualize_input_output_distributions(
                        input_colors_by_attr[attr][bin_name],
                        output_colors_by_attr[attr][bin_name],
                        bin_name,
                        attr,
                        os.path.join(attr_dir, f'input_vs_output_{bin_name.replace("/", "_")}.png')
                    )
                
                bin_kl = {}
                ks_stat = {}
                ks_pval = {}
                for ch in self.channels:
                    input_values = np.array([x for x in input_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)])
                    output_values = np.array([x for x in output_colors_by_attr[attr][bin_name][ch] if not np.isnan(x)])
                    if len(input_values) > 1 and len(output_values) > 1:
                        bin_kl[ch] = kl_divergence(input_values, output_values)
                        print(f"KL divergence for bin {bin_name}, channel {ch}: {bin_kl[ch]}")
                        ks_stat[ch], ks_pval[ch] = ks_2samp(input_values, output_values)
                        print(f"K-S test for bin {bin_name}, channel {ch}: stat={ks_stat[ch]}, p={ks_pval[ch]}")
                if bin_kl:
                    kl_in_out[bin_name] = bin_kl
            
            if display_examples:
                visualize_kl_heatmap(
                    kl_in_out,
                    attr,
                    binning_config[attr]['bin_names'],
                    os.path.join(attr_dir, f'kl_input_output_heatmap.png')
                )
            
            results[attr]['kl_input_output'] = kl_in_out
            results[attr]['input_distributions'] = input_colors_by_attr[attr]
            results[attr]['output_distributions'] = output_colors_by_attr[attr]
        
        return results

    def test_color_consistency_across_bins(
        self,
        csv_path: str,
        target_attribute: str,
        reference_color_idx: int = 0,
        display_examples: bool = True
    ) -> Dict[str, Any]:
        """
        Test if the distribution of reconstructed colors varies significantly
        between different bins for a fixed input color. This method reuses the
        global workspace pass by calling _process_csv with fixed_reference=True.
        """
        if target_attribute not in self.binning_config:
            raise ValueError(f"Invalid attribute: {target_attribute}. Valid options: {list(self.binning_config.keys())}")
        
        output_dir = os.path.join(self.output_dir, f"consistency_{target_attribute}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the CSV once with fixed reference color.
        df, processed_samples = self._process_csv(
            csv_path,
            attributes=list(self.binning_config.keys()),
            use_fixed_reference=True,
            reference_color_idx=reference_color_idx
        )
        
        # Filter rows that use the reference color.
        df_filtered = df[df["color_index"] == reference_color_idx]
        if len(df_filtered) == 0:
            print(f"No examples found with color_index={reference_color_idx}")
            return {}
        
        ref_color_rgb = self.rgb_colors[reference_color_idx]
        print(f"Reference color: RGB={ref_color_rgb}")
        
        bin_names = self.binning_config[target_attribute]['bin_names']
        output_distributions = {bn: {ch: [] for ch in self.channels} for bn in bin_names}
        bin_examples = {bn: [] for bn in bin_names}
        
        # Use the cached processed samples (assuming the order is preserved).
        for idx in df_filtered.index:
            row = df_filtered.loc[idx]
            # Determine bin index based on target_attribute.
            if target_attribute == 'shape':
                bin_idx = int(row['class'])
            elif target_attribute == 'rotation':
                bin_idx = bin_attribute(float(row['rotation']),
                                        self.binning_config['rotation']['n_bins'],
                                        self.binning_config['rotation']['range'])
            elif target_attribute == 'size':
                bin_idx = bin_attribute(float(row['size']),
                                        self.binning_config['size']['n_bins'],
                                        self.binning_config['size']['range'])
            elif target_attribute == 'position_x':
                bin_idx = bin_attribute(row['location'][0],
                                        self.binning_config['position_x']['n_bins'],
                                        self.binning_config['position_x']['range'])
            elif target_attribute == 'position_y':
                bin_idx = bin_attribute(row['location'][1],
                                        self.binning_config['position_y']['n_bins'],
                                        self.binning_config['position_y']['range'])
            bn = bin_names[bin_idx]
            
            # Use the precomputed global workspace output.
            processed_sample = processed_samples[idx]
            for ch in self.channels:
                output_distributions[bn][ch].append(processed_sample[ch])
            bin_examples[bn].append(processed_sample)
        
        if display_examples:
            visualize_examples_by_attribute(
                bin_examples,
                target_attribute,
                bin_names,
                os.path.join(output_dir, f'examples_by_{target_attribute}.png')
            )
            visualize_color_distributions_by_attribute(
                output_distributions,
                target_attribute,
                bin_names,
                os.path.join(output_dir, f'color_distributions_by_{target_attribute}.png')
            )
        
        # Compare each bin with the reference bin.
        bin_comparisons = {}
        max_samples = 0
        ref_bin = None
        for bn in bin_names:
            n_samples = len([x for x in output_distributions[bn]["R"] if not np.isnan(x)])
            if n_samples > max_samples:
                max_samples = n_samples
                ref_bin = bn
        
        if ref_bin is None:
            print("No bin with valid data")
            return {}
        
        print(f"Reference bin for comparisons: {ref_bin}")
        significant_differences = []
        
        for bn in bin_names:
            if bn == ref_bin:
                continue
            bin_comparisons[bn] = {}
            for ch in self.channels:
                ref_vals = np.array([x for x in output_distributions[ref_bin][ch] if not np.isnan(x)])
                bin_vals = np.array([x for x in output_distributions[bn][ch] if not np.isnan(x)])
                if len(ref_vals) > 1 and len(bin_vals) > 1:
                    kl_val = kl_divergence(ref_vals, bin_vals)
                    bin_comparisons[bn][f"{ch}_kl"] = kl_val
                    ks_stat, ks_pval = ks_2samp(ref_vals, bin_vals)
                    bin_comparisons[bn][f"{ch}_ks_stat"] = ks_stat
                    bin_comparisons[bn][f"{ch}_ks_pval"] = ks_pval
                    print(f"Bin {bn} vs {ref_bin}, channel {ch}: KL={kl_val}, KS={ks_stat}, p={ks_pval}")
                    if ks_pval < 0.05:
                        significant_differences.append((bn, ch, ks_pval))
        
        # (Visualization of comparison heatmaps is similar to before.)
        if display_examples:
            valid_bins = [b for b in bin_names if b != ref_bin and b in bin_comparisons]
            if valid_bins:
                kl_matrix = np.zeros((len(valid_bins), len(self.channels)))
                ks_stat_matrix = np.zeros((len(valid_bins), len(self.channels)))
                ks_pval_matrix = np.zeros((len(valid_bins), len(self.channels)))
                for i, bn in enumerate(valid_bins):
                    for j, ch in enumerate(self.channels):
                        if f"{ch}_kl" in bin_comparisons[bn]:
                            kl_matrix[i, j] = bin_comparisons[bn][f"{ch}_kl"]
                            ks_stat_matrix[i, j] = bin_comparisons[bn][f"{ch}_ks_stat"]
                            ks_pval_matrix[i, j] = bin_comparisons[bn][f"{ch}_ks_pval"]
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                im0 = axes[0].imshow(kl_matrix, cmap='viridis')
                im1 = axes[1].imshow(ks_stat_matrix, cmap='viridis')
                im2 = axes[2].imshow(ks_pval_matrix, cmap='viridis', vmin=0, vmax=0.05)
                for i in range(len(valid_bins)):
                    for j in range(len(self.channels)):
                        axes[0].text(j, i, f"{kl_matrix[i, j]:.2f}", ha="center", va="center")
                        axes[1].text(j, i, f"{ks_stat_matrix[i, j]:.2f}", ha="center", va="center")
                        axes[2].text(j, i, f"{ks_pval_matrix[i, j]:.3f}", ha="center", va="center")
                for ax, title, data in zip(axes, ["KL Divergence", "KS Statistic", "KS p-value"],
                                             [kl_matrix, ks_stat_matrix, ks_pval_matrix]):
                    ax.set_xticks(np.arange(len(self.channels)))
                    ax.set_yticks(np.arange(len(valid_bins)))
                    ax.set_xticklabels(self.channels)
                    ax.set_yticklabels(valid_bins)
                    ax.set_title(title)
                    plt.colorbar(ax.imshow(data, cmap='viridis'), ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'bin_comparison_stats.png'))
                plt.close()
        
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(f"Color consistency analysis for attribute: {target_attribute}\n")
            f.write(f"Reference color: RGB={ref_color_rgb}\n")
            f.write(f"Reference bin: {ref_bin}\n\n")
            if significant_differences:
                f.write("Significant differences detected (p < 0.05):\n")
                for bn, ch, pval in significant_differences:
                    f.write(f"  Bin {bn}, channel {ch}: p-value = {pval:.4f}\n")
            else:
                f.write("No significant differences detected between bins.\n")
        
        return {
            "reference_bin": ref_bin,
            "reference_color": ref_color_rgb,
            "comparisons": bin_comparisons,
            "output_distributions": output_distributions,
            "significant_differences": significant_differences
        }


    def compare_color_distributions_across_shapes(
            self,
            csv_path: str,
            shape_names: List[str] = None,
            display_distributions: bool = True,
            display_ks_test: bool = True
        ) -> Dict[str, Any]:
            """
            Compares reconstructed color distributions across different shapes using KS tests.
            """
            output_dir = os.path.join(self.output_dir, "shape_color_comparison")
            os.makedirs(output_dir, exist_ok=True)

            if shape_names is None:
                shape_names = self.shape_names
            else:
                for shape_name in shape_names:
                    if shape_name not in self.shape_names:
                        raise ValueError(f"Shape '{shape_name}' not in the analyzer's shape list: {self.shape_names}")

            df, processed_samples = self._process_csv(csv_path, attributes=['shape'], use_fixed_reference=False)

            shape_color_distributions = {shape_name: {ch: [] for ch in self.channels} for shape_name in shape_names}

            for index, row in df.iterrows():
                shape_label = self.shape_names[int(row['class'])] # Assuming 'class' column represents shape index
                if shape_label in shape_names:
                    processed_sample = processed_samples[index]
                    for ch in self.channels:
                        shape_color_distributions[shape_label][ch].append(processed_sample[ch])

            if display_distributions:
                visualize_color_distributions_by_attribute(
                    shape_color_distributions,
                    "shape",
                    shape_names,
                    os.path.join(output_dir, f'shape_color_distributions.png')
                )

            ks_test_results = {}
            shape_pairs = []
            for i in range(len(shape_names)):
                for j in range(i + 1, len(shape_names)):
                    shape_pairs.append((shape_names[i], shape_names[j]))

            for shape1, shape2 in shape_pairs:
                ks_test_results[(shape1, shape2)] = {}
                for ch in self.channels:
                    dist1 = np.array([x for x in shape_color_distributions[shape1][ch] if not np.isnan(x)])
                    dist2 = np.array([x for x in shape_color_distributions[shape2][ch] if not np.isnan(x)])
                    if len(dist1) > 1 and len(dist2) > 1:
                        ks_stat, ks_pval = ks_2samp(dist1, dist2)
                        ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': ks_stat, 'p_value': ks_pval}
                        print(f"KS test for shapes {shape1} vs {shape2}, channel {ch}: stat={ks_stat:.3f}, p={ks_pval:.3f}")
                    else:
                        ks_test_results[(shape1, shape2)][ch] = {'ks_statistic': None, 'p_value': None, 'error': 'Insufficient data'}
                        print(f"Insufficient data to compare shapes {shape1} and {shape2} for channel {ch}")

            if display_ks_test and shape_pairs:
                ks_pval_matrix = np.zeros((len(shape_names), len(shape_names)))
                valid_pairs_idx = []
                for pair_idx, (shape1, shape2) in enumerate(shape_pairs):
                    row_idx = shape_names.index(shape1)
                    col_idx = shape_names.index(shape2)
                    pair_p_values = [ks_test_results[(shape1, shape2)][ch]['p_value'] for ch in self.channels if ks_test_results[(shape1, shape2)][ch]['p_value'] is not None]
                    if pair_p_values:
                        avg_p_value = np.mean(pair_p_values) # Average p-value across channels for visualization
                        ks_pval_matrix[row_idx, col_idx] = avg_p_value
                        ks_pval_matrix[col_idx, row_idx] = avg_p_value # Symmetric matrix for visualization
                        valid_pairs_idx.append(pair_idx)


                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(ks_pval_matrix, cmap='viridis', vmin=0, vmax=0.05) # Focus on significant p-values
                ax.set_xticks(np.arange(len(shape_names)))
                ax.set_yticks(np.arange(len(shape_names)))
                ax.set_xticklabels(shape_names)
                ax.set_yticklabels(shape_names)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                for i in range(len(shape_names)):
                    for j in range(len(shape_names)):
                        if i < j and (shape_names[i], shape_names[j]) in ks_test_results:
                            avg_p_val_text = f"{np.mean([ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value'] for ch in self.channels if ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value'] is not None]):.3f}" if any(ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value'] is not None for ch in self.channels) else "N/A"
                            text = ax.text(j, i, avg_p_val_text, ha="center", va="center", color="w" if np.mean([ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value'] for ch in self.channels if ks_test_results[(shape_names[i], shape_names[j])][ch]['p_value'] is not None]) < 0.05 else "black") # white if significant
                        elif i > j:
                            avg_ks_stat_text = f"{np.mean([ks_test_results[(shape_names[j], shape_names[i])][ch]['ks_statistic'] for ch in self.channels if ks_test_results[(shape_names[j], shape_names[i])][ch]['ks_statistic'] is not None]):.2f}" if any(ks_test_results[(shape_names[j], shape_names[i])][ch]['ks_statistic'] is not None for ch in self.channels) else "N/A"
                            text = ax.text(j, i, avg_ks_stat_text, ha="center", va="center", color="black")


                ax.set_title("KS Test p-values (Upper Triangle) and KS Statistics (Lower Triangle) for Color Distribution Comparison Across Shapes")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shape_color_ks_test_heatmap.png'))
                plt.close()


            summary_path = os.path.join(output_dir, 'shape_color_comparison_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("Shape Color Distribution Comparison Summary\n")
                for (shape1, shape2), results in ks_test_results.items():
                    f.write(f"\nComparison between {shape1} and {shape2}:\n")
                    for ch, values in results.items():
                        if 'error' in values:
                            f.write(f"  Channel {ch}: {values['error']}\n")
                        else:
                            f.write(f"  Channel {ch}: KS Statistic = {values['ks_statistic']:.3f}, p-value = {values['p_value']:.3f}")
                            if values['p_value'] < 0.05:
                                f.write(" (Significant difference)")
                            f.write("\n")

            return {
                "shape_color_distributions": shape_color_distributions,
                "ks_test_results": ks_test_results,
                "output_dir": output_dir
            }


# def evaluate_dataset(
#     csv_path: str,
#     global_workspace,
#     device: torch.device,
#     analysis_attributes: List[str] = None,
#     color: bool = True,
#     display_examples: bool = True,
#     output_dir: str = "."
# ) -> Dict[str, Any]:
#     """
#     Evaluate a dataset of shape images using the ShapeAnalyzer.
#     """
#     analyzer = ShapeAnalyzer(
#         global_workspace=global_workspace,
#         device=device,
#         color=color,
#         output_dir=output_dir
#     )
#     results = analyzer.analyze_dataset(
#         csv_path=csv_path,
#         analysis_attributes=analysis_attributes,
#         display_examples=display_examples
#     )
#     return results


# def test_color_consistency_across_bins(
#     csv_path: str,
#     global_workspace,
#     device: torch.device,
#     target_attribute: str,
#     reference_color_idx: int = 0,
#     color: bool = True,
#     output_dir: str = "."
# ) -> Dict[str, Any]:
#     """
#     Test color consistency across bins using a single global workspace pass.
#     """
#     analyzer = ShapeAnalyzer(
#         global_workspace=global_workspace,
#         device=device,
#         color=color,
#         output_dir=output_dir
#     )
#     return analyzer.test_color_consistency_across_bins(
#         csv_path=csv_path,
#         target_attribute=target_attribute,
#         reference_color_idx=reference_color_idx,
#         display_examples=True
#     )
