# utils.py
import os
import math
import io
import csv
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import patches
import cv2
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp
import colorsys
from tqdm import tqdm
import torchvision.transforms.functional as TF

from shimmer_ssd import LOGGER # Assuming LOGGER is configured elsewhere

# Assuming these are available or defined within shimmer_ssd/simple_shapes_dataset
try:
    from simple_shapes_dataset.cli import generate_image, get_transformed_coordinates
except ImportError:
    LOGGER.error("Could not import from simple_shapes_dataset. Ensure it's installed.")
    # Define dummy functions if needed for static analysis, but code will fail at runtime
    def generate_image(*args, **kwargs): raise NotImplementedError
    def get_transformed_coordinates(*args, **kwargs): raise NotImplementedError

# --- Configuration ---
DEFAULT_BINNING_CONFIG = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle'] # Must match shape_names order
    },
    'rotation': {
        'n_bins': 4,
        'range': (0, 2 * np.pi),
        'bin_names': ['0°', '90°', '180°', '270°']
    },
    'size': {
        'n_bins': 4,
        'range': (7, 14),
        'bin_names': ['Very Small', 'Small', 'Medium', 'Large']
    },
    'position_x': {
        'n_bins': 3, # Adjust if center is included: Left, Center, Right? Or just Left/Right?
        'range': (0, 32), # Assuming imsize 32
        # Example names, adjust based on actual position bins if needed
        'bin_names': ['Left', 'Center', 'Right']
    },
    'position_y': {
        'n_bins': 3,
        'range': (0, 32),
        'bin_names': ['Bottom', 'Center', 'Top']
    }
    # Add 'color' binning if needed (e.g., based on hue ranges)
}


# --- Color Utilities ---

def generate_fixed_colors(n_colors: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generates a fixed set of RGB colors distributed across the hue spectrum."""
    # Simple HSV generation for now, can be replaced with a more sophisticated method
    rgb_colors = []
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    for h in hues:
        # Use fixed saturation and value for brighter colors
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
        rgb_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    LOGGER.info(f"Generated {len(rgb_colors)} fixed RGB colors.")
    # Returning None for HLS as it wasn't used in the refactored analyzer
    return np.array(rgb_colors), None

def rgb_to_hsv(rgb_values_0_255: Union[np.ndarray, Tuple[int, int, int], None]) -> Tuple[float, float, float]:
    """
    Convert RGB values (0-255) to HSV values.
    Returns H in range [0, 360), S and V in range [0, 1].
    Returns (nan, nan, nan) if input is None or conversion fails.
    """
    if rgb_values_0_255 is None:
        return np.nan, np.nan, np.nan
    try:
        # Normalize RGB to [0, 1] range
        r, g, b = [x / 255.0 for x in rgb_values_0_255]
        # colorsys expects 0-1 range
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # Convert H to degrees [0, 360)
        h_deg = h * 360
        return h_deg, s, v
    except Exception as e:
        LOGGER.warning(f"Could not convert RGB {rgb_values_0_255} to HSV: {e}")
        # Attempt to return V if possible (for grayscale)
        try: v = max(r, g, b)
        except: v = np.nan
        return np.nan, np.nan, v # Return NaN for H, S if conversion fails


# --- Normalization Utilities ---

def normalize_size(size: int, min_val: int = 7, max_val: int = 14) -> float:
    """Normalize size to [0, 1] range."""
    if max_val == min_val: return 0.5 # Avoid division by zero
    return (size - min_val) / (max_val - min_val)

def normalize_rotation(rotation: float) -> float:
    """Normalize rotation (0 to 2pi) to [0, 1] range."""
    return rotation / (2 * np.pi)

def normalize_position(pos_coord: float, max_val: int = 32) -> float:
    """Normalize position coordinate (0 to max_val) to [0, 1] range."""
    return pos_coord / max_val

# --- Image Processing Utilities ---

def segment_shape(image_np: np.ndarray, threshold: float = 0.1) -> Optional[np.ndarray]:
    """
    Simple threshold-based segmentation to create a mask of the shape.
    Assumes image_np is in [0, 1] range (C, H, W) or (H, W, C).
    Assumes shape is brighter than background.
    Returns a binary mask (H, W) or None if fails.
    """
    try:
        if image_np.ndim == 3:
            # Convert to grayscale if needed (average over channels or use luminance)
            if image_np.shape[0] == 3: # C, H, W
                 gray_img = np.mean(image_np, axis=0)
            elif image_np.shape[2] == 3: # H, W, C
                 gray_img = np.mean(image_np, axis=2)
            else: # Assuming single channel H, W
                 gray_img = image_np
        elif image_np.ndim == 2:
            gray_img = image_np
        else:
            LOGGER.error(f"Unsupported image dimension for segmentation: {image_np.ndim}")
            return None

        # Thresholding
        mask = (gray_img > threshold).astype(np.uint8)
        return mask
    except Exception as e:
        LOGGER.error(f"Error during segmentation: {e}")
        return None


def extract_shape_color(image_np: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Extracts the average color from the masked region of the image.
    Assumes image_np is (H, W, C) with values in [0, 1].
    Returns average RGB color [0, 1] as numpy array, or None if fails.
    """
    if mask is None or image_np is None:
        return None
    if image_np.ndim != 3 or image_np.shape[2] != 3:
         LOGGER.error(f"Image must be HWC for color extraction, got shape {image_np.shape}")
         return None
    if image_np.shape[:2] != mask.shape:
         LOGGER.error(f"Image shape {image_np.shape[:2]} and mask shape {mask.shape} mismatch.")
         return None

    try:
        # Ensure mask is boolean
        mask_bool = mask > 0
        if np.sum(mask_bool) == 0:
             LOGGER.warning("Empty mask provided for color extraction.")
             return None # Or return background color?

        # Calculate mean color only where mask is True
        masked_pixels = image_np[mask_bool]
        avg_color = np.mean(masked_pixels, axis=0)

        # Ensure color is within [0, 1]
        avg_color = np.clip(avg_color, 0.0, 1.0)
        return avg_color # Shape (3,) with values in [0, 1]

    except Exception as e:
        LOGGER.error(f"Error during color extraction: {e}")
        return None


def generate_image_tensor(
    cls: int,
    pos: np.ndarray,
    size: int,
    rotation: float,
    color_rgb_0_255: Union[np.ndarray, Tuple[int, int, int]], # Expect RGB 0-255
    imsize: int,
    device: torch.device
) -> Optional[torch.Tensor]:
    """
    Generates an image using matplotlib based on attributes and returns it as a PyTorch tensor.
    Returns None if generation fails.
    """
    dpi = 100
    fig = None # Initialize fig to None
    try:
        # Normalize color to 0-1 for matplotlib
        color_rgb_0_1 = np.array(color_rgb_0_255) / 255.0

        fig, ax = plt.subplots(figsize=(imsize/dpi, imsize/dpi), dpi=dpi, frameon=False)
        ax.set_axis_off()
        ax.set_xlim(0, imsize)
        ax.set_ylim(0, imsize)
        ax.set_aspect('equal', adjustable='box')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # Remove padding

        # --- Call the existing generate_image function ---
        generate_image(ax, cls, pos, size, rotation, color_rgb_0_1, imsize) # Pass 0-1 color

        buf = io.BytesIO()
        # Save without extra whitespace
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Close figure to save memory
        fig = None # Ensure fig is cleared after close
        buf.seek(0)

        pil_img = Image.open(buf).convert('RGB')
        # Ensure image is exactly imsize x imsize
        if pil_img.size != (imsize, imsize):
            pil_img = pil_img.resize((imsize, imsize), Image.Resampling.LANCZOS)

        img_tensor = TF.to_tensor(pil_img) # Converts to (C, H, W) and scales to [0, 1]
        return img_tensor.to(device)

    except Exception as e:
        LOGGER.error(f"Failed to generate image tensor for class {cls}, pos {pos}: {e}")
        if fig is not None:
            plt.close(fig) # Attempt to close figure even if error occurred
        return None


# --- Dataset Generation ---

def generate_dataset(output_dir: str, imsize: int = 32, min_scale: int = 7, max_scale: int = 14,
                     n_sizes: int = 4, n_colors: int = 100, seed: int = 0,
                     rotations_array: Optional[np.ndarray] = None,
                     positions_array: Optional[np.ndarray] = None,
                     shapes: List[int] = [0, 1, 2]) -> None:
    """Generates the Simple Shapes Dataset images and attributes CSV."""
    np.random.seed(seed)
    if rotations_array is None:
        rotations_array = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    if positions_array is None:
        margin = max_scale // 2 + 2 # Ensure margin allows center placement + buffer
        positions_array = np.array([
            [margin, margin], [margin, imsize - margin], # Adjusted indices
            [imsize - margin, margin], [imsize - margin, imsize - margin],
            [imsize//2, imsize//2] # Center position
        ])
        LOGGER.info(f"Using default positions: {positions_array.tolist()}")

    sizes = np.linspace(min_scale, max_scale, n_sizes, dtype=int)
    rgb_colors, _ = generate_fixed_colors(n_colors) # Generate fixed colors

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_file = out_dir / "attributes.csv"
    columns = ["filename", "class", "color_index", "size", "rotation", "location_x", "location_y"]
    total_expected = len(shapes) * n_colors * len(sizes) * len(rotations_array) * len(positions_array)
    count = 0
    LOGGER.info(f"Starting dataset generation for approx {total_expected} images in {out_dir}...")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        # Use tqdm for progress bar
        pbar = tqdm(total=total_expected, desc="Generating Dataset")
        for cls in shapes:
            for color_idx in range(n_colors):
                color_rgb_0_255 = rgb_colors[color_idx]
                for size in sizes:
                    for rotation in rotations_array:
                        for pos in positions_array:
                            filename = f"image_{count:06d}.png"
                            filepath = out_dir / filename
                            # Generate image tensor first to check for errors
                            img_tensor = generate_image_tensor(
                                cls, pos, size, rotation, color_rgb_0_255, imsize, torch.device('cpu') # Gen on CPU
                            )
                            if img_tensor is not None:
                                # Save the generated tensor as an image
                                TF.to_pil_image(img_tensor).save(filepath)
                                # Write row with separate x, y
                                writer.writerow([filename, cls, color_idx, size, rotation, pos[0], pos[1]])
                                count += 1
                            else:
                                LOGGER.warning(f"Skipped saving image {count} due to generation error.")
                            pbar.update(1)
        pbar.close()

    LOGGER.info(f"\nDataset generation complete. {count} images created in {output_dir}")


# --- Data Preprocessing for Model ---

def preprocess_dataset(
    df: pd.DataFrame,
    attributes_for_vector: List[str], # Attributes model expects in attr_vector
    shape_names: List[str],
    color_attr_in_model: bool, # True if model expects color in attr_vector
    rgb_colors_list: np.ndarray, # Full list of RGB 0-255 colors by index
    device: torch.device,
    visual_encoder: torch.nn.Module, # The VAE encoder (e.g., vae.encoder)
    imsize: int = 32,
) -> List[Dict[str, Any]]:
    """
    Preprocesses samples from DataFrame: generates images, encodes them with VAE encoder,
    and creates attribute vectors based on model requirements.
    """
    preprocessed_samples = []
    # Consolidate location columns if needed
    if 'location_x' in df.columns and 'location_y' in df.columns and 'location' not in df.columns:
         df['location'] = df.apply(lambda row: [row['location_x'], row['location_y']], axis=1)
    elif 'location' in df.columns and isinstance(df["location"].iloc[0], str):
         # Handle stringified lists like '[10, 10]'
         try: df["location"] = df["location"].apply(ast.literal_eval)
         except (ValueError, SyntaxError) as e:
              raise ValueError(f"Failed to parse 'location' column. Ensure it contains valid list representations. Error: {e}")

    required_cols = ['class', 'location', 'size', 'rotation', 'color_index']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame missing required columns: {missing}. Found: {df.columns.tolist()}")

    visual_encoder.eval() # Ensure encoder is in eval mode

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing (Generating & Encoding)"):
            try:
                cls = int(row['class'])
                # Ensure location is a numpy array
                loc = np.array(row['location'], dtype=np.float32)
                size = int(row['size'])
                rotation = float(row['rotation'])
                color_idx = int(row['color_index'])

                if not (0 <= color_idx < len(rgb_colors_list)):
                     LOGGER.warning(f"Skipping sample {index}: Invalid color_index {color_idx}")
                     continue
                rgb_color_0_255 = rgb_colors_list[color_idx]

                # --- Generate image ---
                image_tensor = generate_image_tensor(
                    cls, loc, size, rotation, rgb_color_0_255, imsize, device
                ) # (C, H, W), [0, 1]

                if image_tensor is None:
                     LOGGER.warning(f"Skipping sample {index} due to image generation error.")
                     continue

                # --- Encode image: encode returns (mu, logvar) ---
                # Ensure tensor has batch dimension
                mean, logvar = visual_encoder(image_tensor.unsqueeze(0))
                initial_v_latent = mean # Use mean as the latent representation

                # --- Create attribute vector ---
                # One-hot for shape class (if needed by attr domain - check domain impl.)
                # one_hot = torch.zeros(len(shape_names), device=device)
                # one_hot[cls] = 1.0

                # Normalize attributes
                norm_size = normalize_size(size) # Uses default range
                norm_rot = normalize_rotation(rotation)
                norm_pos_x = normalize_position(loc[0], max_val=imsize)
                norm_pos_y = normalize_position(loc[1], max_val=imsize)

                attr_list = []
                # Order matters - should match the attribute domain's expectation
                if 'size' in attributes_for_vector: attr_list.append(norm_size)
                if 'rotation' in attributes_for_vector: attr_list.append(norm_rot)
                if 'position_x' in attributes_for_vector: attr_list.append(norm_pos_x)
                if 'position_y' in attributes_for_vector: attr_list.append(norm_pos_y)

                # Add normalized color if required by the model
                if color_attr_in_model and 'color' in attributes_for_vector:
                    h, s, v = rgb_to_hsv(rgb_color_0_255) # H(0-360), S(0-1), V(0-1)
                    norm_h = h / 360.0 if not np.isnan(h) else 0.0
                    norm_s = s if not np.isnan(s) else 0.0
                    norm_v = v if not np.isnan(v) else 0.0
                    # Assuming order H, S, V if 'color' group is requested
                    attr_list.extend([norm_h, norm_s, norm_v])
                elif color_attr_in_model and 'color' not in attributes_for_vector:
                     LOGGER.warning("Model expects color attribute, but 'color' not in attributes_for_vector list.")
                elif not color_attr_in_model and 'color' in attributes_for_vector:
                     LOGGER.warning("Model does NOT expect color attribute, but 'color' IS in attributes_for_vector list.")


                attr_vector = torch.tensor(attr_list, dtype=torch.float32, device=device)

                # --- Store results ---
                sample_data = {
                    'index': index,
                    'ground_truth': {
                        'class': cls, 'location': loc, 'size': size, 'rotation': rotation,
                        'color_index': color_idx, 'rgb_color_0_255': tuple(rgb_color_0_255), # GT color
                        'image_tensor': image_tensor # Store the generated tensor (on device)
                    },
                    'model_inputs': {
                        # 'one_hot': one_hot.unsqueeze(0), # Include if attr domain uses it
                        'attr_vector': attr_vector.unsqueeze(0), # Add batch dim
                        'v_latent': initial_v_latent, # Already has batch dim from encoder
                    }
                }
                preprocessed_samples.append(sample_data)

            except Exception as e:
                LOGGER.error(f"Error preprocessing sample index {index}: {e}", exc_info=True) # Log traceback
                continue # Skip faulty sample

    return preprocessed_samples


# --- Model Processing Logic ---

def process_through_global_workspace(
    global_workspace: Any, # Should be GlobalWorkspace2Domains instance
    preprocessed_samples: List[Dict],
    device: torch.device,
) -> List[Dict]:
    """
    Processes samples through the Global Workspace for three paths:
    1. Translation: Attribute -> GW -> Visual
    2. Half-Cycle: Visual -> GW -> Visual
    3. Full-Cycle: Visual -> GW -> Attribute -> GW -> Visual
    Decodes images and extracts colors for each path.
    """
    processed_results = []
    global_workspace.eval() # Ensure model is in evaluation mode

    # --- Get necessary model components (robust access) ---
    visual_domain = global_workspace.domain_mods.get("v_latents")
    attr_domain = global_workspace.domain_mods.get("attr")
    gw_mod = getattr(global_workspace, "gw_mod", None)

    if not all([visual_domain, attr_domain, gw_mod]):
        raise ValueError("Could not find required modules ('v_latents', 'attr', 'gw_mod') in global_workspace.")
    if not hasattr(visual_domain, 'decode_images'):
        raise AttributeError("Visual domain module ('v_latents') must have a 'decode_images' method.")
    if not hasattr(attr_domain, 'encode'):
         raise AttributeError("Attribute domain module ('attr') must have an 'encode' method.")
         # Add check for attr_domain.decode if needed later

    with torch.no_grad():
        for sample in tqdm(preprocessed_samples, desc="Processing through GW"):
            sample_idx = sample.get('index', 'N/A') # For logging
            result_dict = {'original_sample': sample}

            # --- Get Inputs ---
            try:
                model_inputs = sample['model_inputs']
                attr_vector = model_inputs['attr_vector'].to(device) # Shape (1, attr_dim)
                initial_v_latent = model_inputs['v_latent'].to(device) # Shape (1, latent_dim)
                batch_size = initial_v_latent.size(0)
                if batch_size != 1: # This logic assumes batch size 1
                    LOGGER.warning(f"Sample {sample_idx}: Unexpected batch size {batch_size}.")
                    # Handle potential issues or skip
                    continue

                # Create presence tensors (assuming batch size 1)
                presence_v = {"v_latents": torch.ones(batch_size, device=device)}
                presence_attr = {"attr": torch.ones(batch_size, device=device)}

            except KeyError as e:
                LOGGER.warning(f"Skipping sample {sample_idx}: Missing key in 'model_inputs': {e}")
                continue
            except Exception as e:
                 LOGGER.error(f"Error processing inputs for sample {sample_idx}: {e}")
                 continue

            # --- Helper to decode, segment, extract color ---
            def decode_segment_extract(v_latent_tensor: Optional[torch.Tensor]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int,int,int]]]:
                """Decodes latent, converts to numpy, segments, extracts color."""
                if v_latent_tensor is None:
                    return None, None, None
                try:
                    # Decode image(s)
                    img_tensor_list = visual_domain.decode_images(v_latent_tensor)
                    # Assuming decode_images returns list or tensor
                    img_tensor = img_tensor_list[0] if isinstance(img_tensor_list, list) else img_tensor_list
                    if img_tensor.ndim == 4: img_tensor = img_tensor.squeeze(0) # Remove batch dim

                    # Convert to numpy HWC [0, 1] for processing
                    img_np_hwc = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np_hwc = np.clip(img_np_hwc, 0.0, 1.0) # Ensure range

                    # Segment
                    mask = segment_shape(img_np_hwc) # Expects HWC [0,1]

                    # Extract Color
                    color_norm_rgb = extract_shape_color(img_np_hwc, mask) # Expects HWC [0,1], mask

                    # Convert color to 0-255 tuple
                    if color_norm_rgb is not None and not np.isnan(color_norm_rgb).any():
                        color_0_255 = tuple((color_norm_rgb * 255).astype(np.uint8))
                    else:
                         color_0_255 = None

                    # Return HWC numpy image, mask, and RGB 0-255 tuple
                    return img_np_hwc, mask, color_0_255

                except Exception as e:
                    LOGGER.error(f"Error during decode/segment/extract for sample {sample_idx}: {e}", exc_info=True)
                    return None, None, None

            # --- Path 1: Translation (Attribute -> GW -> Visual) ---
            translated_v_latent = None
            try:
                # Encode attribute vector using attribute domain module
                attr_latent_domain = attr_domain.encode(attr_vector) # Domain module handles encoding
                # Encode attribute latent using GW encoder
                attr_gw_latent = gw_mod.encode({"attr": attr_latent_domain})
                # Fuse using only attribute presence
                gw_latent_from_attr = gw_mod.fuse(attr_gw_latent, presence_attr)
                # Decode from GW to domains, get visual latent
                decoded_latents_attr = gw_mod.decode(gw_latent_from_attr)
                translated_v_latent = decoded_latents_attr.get("v_latents")

                img, mask, color = decode_segment_extract(translated_v_latent)
                result_dict['translated_image'] = img # HWC numpy [0,1]
                result_dict['translated_mask'] = mask # HW numpy binary
                result_dict['translated_shape_color'] = color # RGB 0-255 tuple or None
            except Exception as e:
                LOGGER.error(f"Error in Translation path for sample {sample_idx}: {e}", exc_info=True)
                result_dict.update({'translated_image': None, 'translated_mask': None, 'translated_shape_color': None})

            # --- Path 2: Half-Cycle (Visual -> GW -> Visual) ---
            half_cycle_v_latent = None
            gw_latent_from_v = None # Store for full-cycle
            try:
                # Encode visual latent using GW encoder
                # Visual domain 'encode' might just pass through latents or wrap them
                # Assuming gw_mod.encode handles the input correctly if it's already latent
                v_gw_latent_initial = gw_mod.encode({"v_latents": initial_v_latent})
                # Fuse using only visual presence
                gw_latent_from_v = gw_mod.fuse(v_gw_latent_initial, presence_v)
                # Decode from GW to domains, get visual latent
                decoded_latents_v = gw_mod.decode(gw_latent_from_v)
                half_cycle_v_latent = decoded_latents_v.get("v_latents")

                img, mask, color = decode_segment_extract(half_cycle_v_latent)
                result_dict['half_cycle_image'] = img
                result_dict['half_cycle_mask'] = mask
                result_dict['half_cycle_shape_color'] = color
            except Exception as e:
                LOGGER.error(f"Error in Half-Cycle path for sample {sample_idx}: {e}", exc_info=True)
                result_dict.update({'half_cycle_image': None, 'half_cycle_mask': None, 'half_cycle_shape_color': None})
                gw_latent_from_v = None # Nullify if error occurred


            # --- Path 3: Full-Cycle (Visual -> GW -> Attribute -> GW -> Visual) ---
            full_cycle_v_latent = None
            try:
                if gw_latent_from_v is not None: # Check if Path 2 fusion succeeded
                    # Decode from GW (from V) to get intermediate Attribute latent
                    intermediate_decoded_latents = gw_mod.decode(gw_latent_from_v)
                    intermediate_attr_latent = intermediate_decoded_latents.get("attr")

                    if intermediate_attr_latent is not None:
                        # Encode intermediate attribute latent using GW encoder
                        intermediate_attr_gw_latent = gw_mod.encode({"attr": intermediate_attr_latent})
                        # Fuse using only attribute presence
                        gw_latent_intermediate = gw_mod.fuse(intermediate_attr_gw_latent, presence_attr)
                        # Decode from GW to domains, get final visual latent
                        final_decoded_latents = gw_mod.decode(gw_latent_intermediate)
                        full_cycle_v_latent = final_decoded_latents.get("v_latents")

                        img, mask, color = decode_segment_extract(full_cycle_v_latent)
                        result_dict['full_cycle_image'] = img
                        result_dict['full_cycle_mask'] = mask
                        result_dict['full_cycle_shape_color'] = color
                    else:
                        # If intermediate attribute latent is None, full cycle fails
                        LOGGER.warning(f"Full-Cycle path failed for sample {sample_idx}: Intermediate attribute latent was None.")
                        result_dict.update({'full_cycle_image': None, 'full_cycle_mask': None, 'full_cycle_shape_color': None})
                else:
                    # If half-cycle failed, full-cycle cannot proceed
                    LOGGER.warning(f"Full-Cycle path skipped for sample {sample_idx} because Half-Cycle path failed.")
                    result_dict.update({'full_cycle_image': None, 'full_cycle_mask': None, 'full_cycle_shape_color': None})

            except Exception as e:
                LOGGER.error(f"Error in Full-Cycle path for sample {sample_idx}: {e}", exc_info=True)
                result_dict.update({'full_cycle_image': None, 'full_cycle_mask': None, 'full_cycle_shape_color': None})

            processed_results.append(result_dict)

    return processed_results


# --- Binning Logic ---

def bin_attribute(value: Any, attribute_name: str, binning_config: Dict) -> Optional[str]:
    """Determines the bin name for a given attribute value based on the config."""
    if attribute_name not in binning_config:
        LOGGER.warning(f"No binning config found for attribute: {attribute_name}")
        return None

    config = binning_config[attribute_name]
    n_bins = config.get('n_bins')
    bin_range = config.get('range')
    bin_names = config.get('bin_names')

    if bin_names and len(bin_names) != n_bins:
        LOGGER.error(f"Mismatch between n_bins ({n_bins}) and number of bin_names ({len(bin_names)}) for {attribute_name}")
        return None

    if bin_range: # Numeric attribute with defined range
        min_val, max_val = bin_range
        if not isinstance(value, (int, float, np.number)):
             LOGGER.warning(f"Numeric value expected for {attribute_name}, got {type(value)}. Cannot bin.")
             return None
        if value < min_val or value > max_val:
            # Handle out-of-range values if needed, or just ignore
            # LOGGER.debug(f"Value {value} for {attribute_name} is outside range {bin_range}. Skipping binning.")
            return None # Or assign to edge bins?

        # Calculate bin index
        bin_width = (max_val - min_val) / n_bins
        # Handle edge case where value equals max_val
        bin_index = min(int((value - min_val) // bin_width), n_bins - 1) if bin_width > 0 else 0

    elif attribute_name == 'shape': # Categorical attribute (shape index)
        if not isinstance(value, (int, np.integer)):
             LOGGER.warning(f"Integer class index expected for shape, got {type(value)}. Cannot bin.")
             return None
        bin_index = value # Assume value is the index

    elif attribute_name == 'rotation': # Special handling for rotation bins
        # Map rotation value (radians) to predefined bins [0, pi/2, pi, 3pi/2] approx
        # This depends heavily on the exact values in rotations_array used in generation
        pi = np.pi
        if np.isclose(value, 0): bin_index = 0
        elif np.isclose(value, pi/2): bin_index = 1
        elif np.isclose(value, pi): bin_index = 2
        elif np.isclose(value, 3*pi/2): bin_index = 3
        else:
            LOGGER.warning(f"Rotation value {value} doesn't match expected bin values. Using range-based binning.")
            # Fallback to range-based binning if exact match fails
            min_val, max_val = 0, 2 * pi
            bin_width = (max_val - min_val) / n_bins
            bin_index = min(int((value - min_val) // bin_width), n_bins - 1) if bin_width > 0 else 0

    else: # Other categorical or unhandled numeric
        LOGGER.warning(f"Binning logic not defined for attribute {attribute_name} without a range. Skipping.")
        return None

    # Return the corresponding bin name
    if bin_names and 0 <= bin_index < len(bin_names):
        return bin_names[bin_index]
    else:
        # Fallback to generic bin name if names are missing or index is invalid
        return f"Bin_{bin_index}"


def initialize_h_binning_structures(
    analysis_attributes: List[str],
    binning_config: Dict,
    shape_names: List[str] # Needed if 'shape' is analyzed
) -> Tuple[Dict, Dict]:
    """
    Initialize data structures for binning samples by attribute for H channel.
    Stores Ground Truth H and H from the three processing paths.
    Also prepares lists for example images.
    """
    colors_by_attr = {} # Stores lists of Hue values {attr: {bin: {path: [hues]}}}
    examples_by_attr = {} # Stores lists of example dicts {attr: {bin: [example_dicts]}}

    # Use provided shape_names for shape bins if 'shape' is analyzed
    if 'shape' in analysis_attributes and 'shape' in binning_config:
         # Ensure config matches provided names
         if len(binning_config['shape'].get('bin_names', [])) != len(shape_names):
              LOGGER.warning(f"Bin config shape names {binning_config['shape'].get('bin_names')} mismatch with provided names {shape_names}. Using provided names.")
         binning_config['shape']['bin_names'] = shape_names
         binning_config['shape']['n_bins'] = len(shape_names)

    for attr in analysis_attributes:
        if attr not in binning_config:
             LOGGER.warning(f"No binning config found for analysis attribute: {attr}. Skipping.")
             continue

        attr_bins = {}
        example_bins = {}
        bin_names = binning_config[attr].get('bin_names')
        if not bin_names:
             LOGGER.warning(f"No bin names found for attribute: {attr}. Generating generic names.")
             n_bins = binning_config[attr].get('n_bins', 1)
             bin_names = [f"Bin_{i}" for i in range(n_bins)]
             # Update config for consistency if needed later
             binning_config[attr]['bin_names'] = bin_names


        for bin_name in bin_names:
            # Structure to hold lists of Hue values for each path
            attr_bins[bin_name] = {
                'H_gt': [],
                'H_translated': [],
                'H_half_cycle': [],
                'H_full_cycle': [],
            }
            example_bins[bin_name] = [] # List to store example dicts

        colors_by_attr[attr] = attr_bins
        examples_by_attr[attr] = example_bins

    return colors_by_attr, examples_by_attr


def bin_h_processed_samples(
    preprocessed_samples: List[Dict], # Contains ground truth info
    processed_samples: List[Dict],    # Contains processed results
    analysis_attributes: List[str],
    binning_config: Dict,
    colors_by_attr: Dict,             # Structure initialized by initialize_h_binning_structures
    examples_by_attr: Dict,           # Structure initialized by initialize_h_binning_structures
    max_examples_per_bin: int = 5,
) -> None:
    """
    Bins samples based on ground truth attributes and collects Hue values
    (ground truth, translated, half-cycle, full-cycle) and example images/data.
    Modifies colors_by_attr and examples_by_attr dictionaries in place.
    """
    if len(preprocessed_samples) != len(processed_samples):
        LOGGER.error("Mismatch between preprocessed ({len(preprocessed_samples)}) and processed ({len(processed_samples)}) sample counts!")
        return # Or raise error

    for i, proc_sample in enumerate(processed_samples):
        # Find corresponding preprocessed sample (safer than relying on index if lists got reordered)
        # Assuming 'index' field matches DataFrame index
        pre_sample_idx = proc_sample.get('original_sample', {}).get('index')
        if pre_sample_idx is None:
            LOGGER.warning(f"Processed sample {i} has no original index. Skipping.")
            continue
        # Find the preprocessed sample with the matching index
        pre_sample = next((p for p in preprocessed_samples if p.get('index') == pre_sample_idx), None)
        if pre_sample is None:
             LOGGER.warning(f"Could not find preprocessed sample for index {pre_sample_idx}. Skipping.")
             continue

        gt = pre_sample['ground_truth']

        for attr in analysis_attributes:
            if attr not in gt and attr not in ['position_x', 'position_y']:
                # Allow splitting location into position_x/y
                LOGGER.debug(f"Attribute '{attr}' not directly in ground truth keys for sample {pre_sample_idx}. Skipping.")
                continue

            try:
                # Determine the value to bin based on the attribute name
                if attr == 'position_x':
                    value_to_bin = gt['location'][0] if 'location' in gt and len(gt['location']) > 0 else None
                elif attr == 'position_y':
                    value_to_bin = gt['location'][1] if 'location' in gt and len(gt['location']) > 1 else None
                elif attr == 'shape':
                    value_to_bin = gt['class'] # Use the class index for shape binning
                else:
                    value_to_bin = gt.get(attr) # Use value directly (e.g., size, rotation)

                if value_to_bin is None:
                     LOGGER.debug(f"Value for attribute '{attr}' is None for sample {pre_sample_idx}. Skipping binning.")
                     continue

                # Get the bin name for this attribute value
                bin_name = bin_attribute(value_to_bin, attr, binning_config)

                if bin_name is not None and attr in colors_by_attr and bin_name in colors_by_attr[attr]:
                     target_color_bin = colors_by_attr[attr][bin_name]
                     target_example_bin = examples_by_attr[attr][bin_name]

                     # --- Collect Hue Values ---
                     # Ground Truth Hue
                     h_gt, _, _ = rgb_to_hsv(gt.get('rgb_color_0_255'))
                     if not np.isnan(h_gt): target_color_bin['H_gt'].append(h_gt)

                     # Hue from different paths
                     for path in ['translated', 'half_cycle', 'full_cycle']:
                         rgb_path = proc_sample.get(f'{path}_shape_color') # Expects RGB 0-255 tuple
                         if rgb_path is not None:
                             h_path, _, _ = rgb_to_hsv(rgb_path)
                             if not np.isnan(h_path):
                                 target_color_bin[f'H_{path}'].append(h_path)

                     # --- Collect Examples (if needed and space available) ---
                     if len(target_example_bin) < max_examples_per_bin:
                         # Store relevant data for visualization
                         example_data = {
                             'index': pre_sample_idx,
                             # GT image tensor already on device, move to CPU for storage if needed later
                             'gt_img_tensor': gt['image_tensor'].cpu(),
                             'gt_color': gt.get('rgb_color_0_255'),
                             # Processed images are numpy HWC [0,1], keep as is
                             'trans_img': proc_sample.get('translated_image'),
                             'trans_color': proc_sample.get('translated_shape_color'),
                             'half_img': proc_sample.get('half_cycle_image'),
                             'half_color': proc_sample.get('half_cycle_shape_color'),
                             'full_img': proc_sample.get('full_cycle_image'),
                             'full_color': proc_sample.get('full_cycle_shape_color'),
                         }
                         target_example_bin.append(example_data)

            except Exception as e:
                LOGGER.error(f"Error binning sample {pre_sample_idx} for attribute '{attr}': {e}", exc_info=True)
                continue # Continue to next attribute or sample


# --- Statistical Utilities ---

def kl_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 50, smoothing: float = 1e-6) -> Optional[float]:
    """
    Calculate KL divergence KL(P || Q) between two distributions P and Q
    represented by samples. Handles potential NaN values and empty arrays.
    Uses histogram estimation.
    """
    # Remove NaN values
    p = p[~np.isnan(p)]
    q = q[~np.isnan(q)]

    if len(p) == 0 or len(q) == 0:
        LOGGER.warning("KL divergence calculation skipped: one or both sample sets are empty after NaN removal.")
        return None # Cannot compute KL divergence if either distribution is empty

    # Define common bins based on the combined range of data
    combined = np.concatenate((p, q))
    min_val, max_val = np.min(combined), np.max(combined)
    if min_val == max_val: # Handle case where all values are the same
        # If both p and q have the same single value, KL is 0.
        # If one is empty, handled above. If different single values, KL is inf.
        # Let's return 0 if means are close enough, otherwise indicate difference.
        # A simpler approach for constant values: if means match, KL=0, else KL=inf (or large number).
        # For now, let's just check if the single values match.
        if len(np.unique(p)) == 1 and len(np.unique(q)) == 1 and np.isclose(p[0], q[0]):
             return 0.0
        else:
             # Cannot compute reliable histogram KL for constant but different values
             LOGGER.warning("KL divergence calculation skipped: data contains constant but different values.")
             return None # Or return float('inf')?

    # Create histograms
    p_hist, bin_edges = np.histogram(p, bins=n_bins, range=(min_val, max_val), density=False)
    q_hist, _ = np.histogram(q, bins=n_bins, range=(min_val, max_val), density=False)

    # Convert counts to probabilities and add smoothing
    p_prob = (p_hist + smoothing) / (np.sum(p_hist) + n_bins * smoothing)
    q_prob = (q_hist + smoothing) / (np.sum(q_hist) + n_bins * smoothing)

    # Calculate KL divergence
    # KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
    kl_div = np.sum(p_prob * (np.log(p_prob) - np.log(q_prob)))

    # Handle potential numerical issues leading to negative KL
    if kl_div < 0 and np.isclose(kl_div, 0):
        return 0.0
    elif kl_div < 0:
        LOGGER.warning(f"Calculated KL divergence is negative ({kl_div}). This might indicate numerical instability. Clamping to 0.")
        return 0.0

    return float(kl_div)

# --- Visualization Utilities (Placeholders - Define these based on SSD_visualize_functions) ---

def visualize_color_distributions_by_attribute(color_data, attribute_name, bin_names, output_path, channels, num_bins):
    LOGGER.info(f"Placeholder: Visualize color distributions for {attribute_name} saved to {output_path}")
    # Implementation would loop through bins, channels and plot histograms/density plots
    pass

def visualize_input_output_distributions(input_colors, output_colors, bin_name, attr, output_path, channels):
    LOGGER.info(f"Placeholder: Visualize input vs output for {attr} - {bin_name} saved to {output_path}")
    # Implementation would plot input and output distributions for specified channels side-by-side or overlaid
    pass

def visualize_kl_heatmap(kl_data, attr, bin_names, output_path, channels):
    LOGGER.info(f"Placeholder: Visualize KL heatmap for {attr} saved to {output_path}")
    # Implementation would create a heatmap showing KL divergence for each channel across bins
    pass

def visualize_examples_by_attribute(examples_by_bin, attr, bin_names, output_path):
    LOGGER.info(f"Placeholder: Visualize examples for {attr} saved to {output_path}")
    # Implementation would create a grid of images showing GT and reconstructed examples for each bin
    pass

def visualize_ks_test_heatmap(ks_test_results, shape_names, output_path_dir, comparison_path):
     LOGGER.info(f"Placeholder: Visualize KS test heatmap saved to {output_path_dir}")
     pass

def visualize_kl_divergence_heatmap(kl_divergence_results, shape_names, output_path_dir, comparison_path, symmetric=True):
    LOGGER.info(f"Placeholder: Visualize KL divergence heatmap saved to {output_path_dir}")
    pass