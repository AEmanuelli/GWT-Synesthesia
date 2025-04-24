import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Dict
import warnings
from tqdm import tqdm
import colorsys
import os 
import pickle

from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.stats import ks_2samp
def preprocess_dataset(
    df: Any,
    analysis_attributes: List[str],
    shape_names: List[str],
    color_mode: bool,
    rgb_colors: np.ndarray,
    device: torch.device,
    fixed_reference: bool = False,
    reference_color_idx: int = 0,
    im_dir: str = "./evaluation_set"
) -> List[dict]:

    """
    Preprocesses the dataset by extracting and normalizing all required parameters
    without performing binning.
    
    Args:
        df: Pandas DataFrame containing the dataset.
        analysis_attributes: List of attributes to analyze.
        shape_names: List of shape names.
        color_mode: Whether color information should be used.
        rgb_colors: RGB color mapping as a numpy array.
        device: Torch device.
        fixed_reference: If True, overrides each sample's color with the reference.
        reference_color_idx: Index of the reference color to use if fixed_reference is True.
        
    Returns:
        List of preprocessed samples with all metadata.
    """
    from PIL import Image
    preprocessed_samples = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing samples"):
        sample = {}
        
        # Extract shape parameters
        cls = int(row["class"])
        pos = row["location"]
        size_px = float(row["size"])
        rotation = float(row["rotation"])
        
        # Normalize shape parameters
        pos_norm = normalize_position(pos)
        size_norm = normalize_size(size_px)
        rotx, roty = normalize_rotation(rotation)
        
        # Store original attribute values for later binning
        sample['shape'] = shape_names[cls]
        sample['position_x'] = pos[0]
        sample['position_y'] = pos[1]
        sample['size'] = size_px
        sample['rotation'] = rotation
        
        # Initialize attribute vector with shape parameters
        attr_vector = torch.tensor(
            [[pos_norm[0], pos_norm[1], size_norm, rotx, roty]],
            dtype=torch.float32
        )
        
        # Process color information: override with fixed reference if requested
        if fixed_reference:
            color_rgb = rgb_colors[reference_color_idx]
        else:
            color_idx = int(row["color_index"])
            color_rgb = rgb_colors[color_idx]
        
        # Store original normalized color (0 to 1 range) for comparison
        original_color = [
            color_rgb[0] / 255.0,
            color_rgb[1] / 255.0,
            color_rgb[2] / 255.0
        ]
        sample['original_color'] = color_rgb
        
        # Normalize RGB values to [-1, 1] range for the model
        color_norm = [
            color_rgb[0] / 127.5 - 1,
            color_rgb[1] / 127.5 - 1,
            color_rgb[2] / 127.5 - 1
        ]
        
        # Add color information to the attribute vector if color mode is enabled
        if color_mode:
            attr_vector = torch.tensor(
                [[pos_norm[0], pos_norm[1], size_norm, rotx, roty,
                  color_norm[0], color_norm[1], color_norm[2]]],
                dtype=torch.float32
            )
        
        # Create one-hot class encoding
        one_hot = F.one_hot(torch.tensor([cls]), num_classes=len(shape_names)).float()
        
        # Store model inputs
        sample['model_inputs'] = {
            'one_hot': one_hot.to(device),
            'attr_vector': attr_vector.to(device)
        }
        
        # Store original row index for reference
        sample['row_idx'] = idx

        # load the corresponding image that is stored right next to the csv file
        image_path = im_dir + "/" + f"image_{idx:05d}.png"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            sample['visual_ground_truth'] = image
        if image is None:
            warnings.warn(f"Image not found at {image_path}; skipping sample.")
            continue
        
        
        
        
        preprocessed_samples.append(sample)
    
    return preprocessed_samples

def generate_fixed_colors(
    n_samples: int, max_hue: int = 359
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate colors with a linear distribution across the hue circle.
    """
    # Create evenly spaced hue values, 180 is the max value for hue, be
    hue_values = np.linspace(0, max_hue, n_samples, endpoint=False).astype(np.uint8)
    
    lightness_values = np.full(n_samples, 180, dtype=np.uint8)
    # Create evenly spaced saturation values (full saturation)
    saturation_values = np.full(n_samples, 230, dtype=np.uint8)
    
    # Create the HLS array
    hls = np.zeros((1, n_samples, 3), dtype=np.uint8)
    hls[0, :, 0] = hue_values
    hls[0, :, 1] = lightness_values
    hls[0, :, 2] = saturation_values
    
    # Convert to RGB
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]
    
    return rgb.astype(int), hls[0].astype(int)

def generate_fixed_colors_original(
    n_samples: int, min_lightness: int = 0, max_lightness: int = 256, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    assert 0 <= max_lightness <= 256
    hls = np.random.randint(
        [0, min_lightness, 0],
        [181, max_lightness, 256],
        size=(1, n_samples, 3),
        dtype=np.uint8,
    )
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]  # type: ignore
    return rgb.astype(int), hls[0].astype(int)


def compare_color_distribution(colors, title, is_hls=False, n_samples=100, seed=0):
    """
    Generate colors and display histograms for each channel (RGB or HLS).
    
    Args:
        colors: Array of color values to plot
        title: Title for the saved plot
        is_hls: If True, interpret colors as HLS; otherwise as RGB
        n_samples: Number of colors to generate
        seed: Random seed for reproducibility
    """
    import matplotlib.pyplot as plt
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set channel names and colors based on the color space
    if is_hls:
        channels = ['Hue', 'Lightness', 'Saturation']
        plot_colors = ['purple', 'gray', 'orange']
        xlims = [(0, 180), (0, 255), (0, 255)]  # HLS ranges
    else:
        channels = ['Red', 'Green', 'Blue']
        plot_colors = ['red', 'green', 'blue']
        xlims = [(0, 255), (0, 255), (0, 255)]  # RGB ranges
    
    # Plot histogram for each channel
    for i in range(3):
        channel_values = colors[:, i]
        axes[i].hist(channel_values, bins=180, color=plot_colors[i], alpha=0.7)
        axes[i].set_title(f'{channels[i]} Channel Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim(*xlims[i])
        
    plt.tight_layout()
    type_text = "HLS" if is_hls else "RGB"
    plt.suptitle(f'{type_text} Color Channel Distributions', y=1.02, fontsize=16)
    plt.savefig(title)
    plt.show()

n_samples = 100
seed = 0
# Generate colors
rgb_colors, hls_colors = generate_fixed_colors_original(n_samples=n_samples)

# Compare color distributions for HLS values
compare_color_distribution(hls_colors, title='hls_distribution', is_hls=True)




def normalize_position(pos: List[float]) -> List[float]:
    """
    Normalize position coordinates from pixel space to [-1, 1] range.
    
    Args:
        pos: Position in pixel coordinates [x, y].
        
    Returns:
        Normalized position in the range [-1, 1].
    """
    return [(pos[0] - 7) / 18 * 2 - 1, (pos[1] - 7) / 18 * 2 - 1]


def normalize_size(size: float) -> float:
    """
    Normalize size from pixel space to [-1, 1] range.
    
    Args:
        size: Size in pixels.
        
    Returns:
        Normalized size in the range [-1, 1].
    """
    s = (size - 7) / 7
    return s * 2 - 1


def normalize_rotation(rot: float) -> Tuple[float, float]:
    """
    Convert rotation angle to cosine and sine components.
    
    Args:
        rot: Rotation angle in radians.
        
    Returns:
        Tuple of (cos(rot), sin(rot)).
    """
    return math.cos(rot), math.sin(rot)


def bin_attribute(value: float, bins: int, attr_range: Tuple[float, float]) -> int:
    """
    Assign a value to a bin index based on the specified range and number of bins.
    
    Args:
        value: The value to bin.
        bins: Number of bins.
        attr_range: (min, max) tuple defining the attribute range.
        
    Returns:
        Bin index from 0 to bins-1.
    """
    min_val, max_val = attr_range
    normalized = (value - min_val) / (max_val - min_val)
    bin_idx = int(normalized * bins + 0.25)
    if bin_idx == bins:
        bin_idx = bins - 1
    return bin_idx


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        p: First distribution as a numpy array.
        q: Second distribution as a numpy array.
        bins: Number of bins for histogram discretization.
        
    Returns:
        KL divergence value.
    """
    if len(p) == 0 or len(q) == 0:
        return float('nan')
    
    # Filter out NaN values
    p = np.array([x for x in p if not np.isnan(x)])
    q = np.array([x for x in q if not np.isnan(x)])
    
    if len(p) == 0 or len(q) == 0:
        return float('nan')

    hist_p, bin_edges = np.histogram(p, bins=bins, range=(0, 255), density=True)
    hist_q, _ = np.histogram(q, bins=bins, range=(0, 255), density=True)

    epsilon = 1e-10
    hist_p = hist_p + epsilon
    hist_q = hist_q + epsilon

    hist_p = hist_p / np.sum(hist_p)
    hist_q = hist_q / np.sum(hist_q)

    return np.sum(hist_p * np.log(hist_p / hist_q))


def segment_shape(image: np.ndarray) -> np.ndarray:
    """
    Segment a shape from the background using Otsu thresholding.
    
    Args:
        image: RGB image array.
        
    Returns:
        Binary mask (bool array) where True indicates shape pixels.
    """
    gray_image = np.mean(image, axis=2)
    gray_uint8 = (gray_image * 255).astype(np.uint8)
    _, mask = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0


def extract_shape_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract the average color of a shape using its mask.
    Args:
        image: RGB image array.
        mask: Binary mask where True indicates shape pixels.
    Returns:
        Array of mean RGB values for the shape.
    """
    # Check if mask is valid
    if mask is None or not isinstance(mask, np.ndarray):
        warnings.warn("Invalid mask; using full image for color extraction")
        return np.mean(image, axis=(0, 1))
        
    # Check if mask has enough pixels
    if np.sum(mask) < 10:
        warnings.warn("Empty mask detected; using full image for color extraction")
        return np.mean(image, axis=(0, 1))
        
    mean_color = np.zeros(3)
    for c in range(3):
        pixels = image[:, :, c][mask]
        if len(pixels) > 0:
            mean_color[c] = np.mean(pixels)
        else:
            mean_color[c] = 0.0
    
    # brightest_image_color = np.max(image, axis=(0, 1))
            
    return mean_color



def process_through_global_workspace(
    global_workspace: Any,
    preprocessed_samples: List[Dict],
    device: torch.device,
    debug: bool = False,
    reverb_n : int = 1,
) -> List[Dict]:
    """
    Processes samples through the Global Workspace, focusing on visual transformations.
    Calculates:
    1. Translation: Attribute -> GW -> Visual Image
    2. Half-Cycle: Visual -> GW -> Visual Image
    3. Full-Cycle: Visual -> GW -> Attribute -> GW -> Visual Image

    Args:
        global_workspace: The global workspace model instance.
        preprocessed_samples: List of preprocessed samples. Each sample dict
                              is expected to contain sample['model_inputs']
                              with keys like 'attr_vector', 'one_hot', and 'v_latent'.
        device: Torch device.
        debug: If True, generates random outputs instead of processing through model.

    Returns:
        List of processed samples, each containing the original sample info
        and the decoded images (and their extracted colors) for the three paths.
    """
    processed_results = []
    global_workspace.eval()  # Make sure model is in evaluation mode

    with torch.no_grad():  # Disable gradient calculations for inference
        for sample in tqdm(preprocessed_samples, desc="Processing samples through GW model"):
            if debug:
                processed_results.append(_generate_debug_output(sample))
                continue

            # Prepare inputs
            attr_inputs, visual_ground_truth_tensor = _prepare_inputs(sample, device)
            
            # Get visual latent vector from ground truth image
            visual_module = global_workspace.domain_mods["v_latents"]
            v_latent_vector = visual_module.visual_module.encode(visual_ground_truth_tensor)
            
            # Process through different pathways
            translated_image_np, translated_mask, translated_shape_color = _process_translation_path(
                global_workspace, attr_inputs, device, n=reverb_n)
                
            half_cycle_image_np, half_cycle_mask, half_cycle_shape_color = _process_half_cycle_path(
                global_workspace, v_latent_vector, device)
                
            full_cycle_image_np, full_cycle_mask, full_cycle_shape_color = _process_full_cycle_path(
                global_workspace, v_latent_vector, device, n=reverb_n)
            
            
            processed_results.append({
                'original_sample': sample,  # Keep track of the original data
                # Translation Results
                'translated_image': translated_image_np,
                'translated_mask': translated_mask,
                'translated_shape_color': translated_shape_color,  # RGB color
                # Half-Cycle Results
                'half_cycle_image': half_cycle_image_np,
                'half_cycle_mask': half_cycle_mask,
                'half_cycle_shape_color': half_cycle_shape_color,  # RGB color
                # Full-Cycle Results
                'full_cycle_image': full_cycle_image_np,
                'full_cycle_mask': full_cycle_mask,
                'full_cycle_shape_color': full_cycle_shape_color,  # RGB color
            })
            
    return processed_results


def _generate_debug_output(sample: Dict) -> Dict:
    """
    Generate random outputs for debugging purposes.
    
    Args:
        sample: The original sample.
        
    Returns:
        Dictionary with random outputs for debugging.
    """
    import numpy as np
    # Generate random outputs for debugging
    random_image = lambda: np.random.random((32, 32, 3))
    random_mask = lambda: np.random.random((32, 32)) > 0.5
    random_color = lambda: np.random.randint(0, 256, size=3)
    
    return {
        'original_sample': sample,  # Keep track of the original data
        # Translation Results
        'translated_image': random_image(),
        'translated_mask': random_mask(),
        'translated_shape_color': random_color(),  # RGB color
        # Half-Cycle Results
        'half_cycle_image': random_image(),
        'half_cycle_mask': random_mask(),
        'half_cycle_shape_color': random_color(),  # RGB color
        # Full-Cycle Results
        'full_cycle_image': random_image(),
        'full_cycle_mask': random_mask(),
        'full_cycle_shape_color': random_color(),  # RGB color
    }


def _prepare_inputs(sample: Dict, device: torch.device) -> Tuple[List, torch.Tensor]:
    """
    Prepare attribute and visual inputs from the sample.
    
    Args:
        sample: The sample dictionary.
        device: Torch device.
        
    Returns:
        Tuple containing attribute inputs and visual ground truth tensor.
    """
    # Attribute inputs
    one_hot = sample['model_inputs']['one_hot'].to(device)
    attr_vector = sample['model_inputs']['attr_vector'].to(device)
    attr_inputs = [one_hot, attr_vector]
    
    # Visual inputs
    if 'visual_ground_truth' not in sample:
        raise ValueError(f"Sample missing 'visual_ground_truth'. Sample keys: {sample.keys()}")
    
    img = sample["visual_ground_truth"]
    visual_ground_truth_tensor = to_tensor(img)[:3].unsqueeze(0).to(device)
    
    return attr_inputs, visual_ground_truth_tensor




def _process_half_cycle_path(global_workspace: Any, v_latent_vector: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the Half-Cycle path: Visual -> GW -> Visual.
    
    Args:
        global_workspace: The global workspace model.
        v_latent_vector: Visual latent vector.
        device: Torch device.
        
    Returns:
        Tuple containing (image, mask, shape_color) for the half-cycle path.
    """
    # Encode Visual Latent -> GW
    v_gw_latent_pre_fusion = global_workspace.gw_mod.encode({"v_latents": v_latent_vector})
    
    # Fuse in GW
    gw_latent_from_v = global_workspace.gw_mod.fuse(
        v_gw_latent_pre_fusion,
        {"v_latents": torch.ones(v_gw_latent_pre_fusion["v_latents"].size(0)).to(device)}
    )
    
    # Decode GW -> Visual Latent
    half_cycle_v_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["v_latents"]
    
    # Decode Visual Latent -> Image
    half_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(half_cycle_v_latent)[0]
    half_cycle_image_np = half_cycle_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    half_cycle_mask = segment_shape(half_cycle_image_np)
    half_cycle_shape_color = extract_shape_color(half_cycle_image_np, half_cycle_mask) * 255
    
    return half_cycle_image_np, half_cycle_mask, half_cycle_shape_color

from shimmer.modules import SingleDomainSelection, FixedSharedSelection, DynamicQueryAttention
def _process_full_cycle_path(global_workspace: Any, v_latent_vector: torch.Tensor, 
                            device: torch.device, n: int = 10, selection_module = SingleDomainSelection()
                            )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    

    """
    Process the Full-Cycle path: Visual -> GW -> Attribute -> GW -> Visual.
    Perform the cycle n times.
    
    Args:
        global_workspace: The global workspace model.
        v_latent_vector: Visual latent vector.
        device: Torch device.
        n: Number of cycles to perform (default: 1).
        
    Returns:
        Tuple containing (image, mask, shape_color) for the full-cycle path.
    """
    selection_module.to(device)
    # Start with the original visual latent vector
    current_v_latent = v_latent_vector
    
    # Perform n cycles
    for _ in range(n):
        # Encode Visual Latent -> GW
        gw_latent_from_v = global_workspace.gw_mod.encode_and_fuse({"v_latents": current_v_latent}, selection_module=selection_module)
        
        # Decode GW -> Attribute Latent
        intermediate_attr_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["attr"]
        attr_latent = global_workspace.domain_mods["attr"].decode(intermediate_attr_latent)
        
        # Encode Attribute Latent -> GW
        intermediate_attr_domain = global_workspace.encode_domain(attr_latent, "attr")
        gw_latent_intermediate = global_workspace.gw_mod.encode_and_fuse({"attr": intermediate_attr_domain}, selection_module=selection_module)
        
        # Decode GW -> Visual Latent
        current_v_latent = global_workspace.gw_mod.decode(gw_latent_intermediate)["v_latents"]
    
    # Decode final Visual Latent -> Image
    full_cycle_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(current_v_latent)[0]
    full_cycle_image_np = full_cycle_image_tensor.permute(1, 2, 0).cpu().detach().numpy()
    full_cycle_mask = segment_shape(full_cycle_image_np)
    full_cycle_shape_color = extract_shape_color(full_cycle_image_np, full_cycle_mask) * 255
    
    return full_cycle_image_np, full_cycle_mask, full_cycle_shape_color

def _process_translation_path(global_workspace: Any, attr_inputs: List, device: torch.device, n: int = 10, selection_module = SingleDomainSelection()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the Translation path: Attribute -> GW -> Visual.
    Perform the cycle n times.
    
    Args:
        global_workspace: The global workspace model.
        attr_inputs: List of attribute inputs.
        device: Torch device.
        n: Number of cycles to perform (default: 1).
        selection_module: Selection module for domain attention (default: SingleDomainSelection()).
    
    Returns:
        Tuple containing (image, mask, shape_color) for the translation path.
    """
    selection_module.to(device)
    
    # Encode initial Attribute -> GW
    current_attr_input = attr_inputs
    
    # Perform n cycles
    for _ in range(n):
        # Encode Attribute -> GW
        intermediate_attr_domain = global_workspace.encode_domain(current_attr_input, "attr")
        attr_gw_latent = global_workspace.gw_mod.encode_and_fuse(
            {"attr": intermediate_attr_domain}, 
            selection_module=selection_module
        )
        
        # Decode GW -> Visual Latent
        v_latent = global_workspace.gw_mod.decode(attr_gw_latent)["v_latents"]
        
        # For additional cycles, we'd need to go back to attribute domain
        if _ < n - 1:
            # Encode Visual Latent -> GW
            gw_latent_from_v = global_workspace.gw_mod.encode_and_fuse(
                {"v_latents": v_latent}, 
                selection_module=selection_module
            )
            
            # Decode GW -> Attribute Latent
            intermediate_attr_latent = global_workspace.gw_mod.decode(gw_latent_from_v)["attr"]
            current_attr_input = global_workspace.domain_mods["attr"].decode(intermediate_attr_latent)
    
    # Decode final Visual Latent -> Image
    translated_image_tensor = global_workspace.domain_mods["v_latents"].decode_images(v_latent)[0]
    translated_image_np = translated_image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    translated_mask = segment_shape(translated_image_np)
    translated_shape_color = extract_shape_color(translated_image_np, translated_mask) * 255
    
    return translated_image_np, translated_mask, translated_shape_color


def safe_extract_channel(processed_sample: dict, channel: str, sample_index: int) -> float:
    """
    Safely extract a channel value from a processed sample.
    If not found at the top level, attempt to extract from 'shape_color'.
    
    Args:
        processed_sample: A dictionary containing processed sample data.
        channel: The color channel to extract (e.g., 'R', 'G', or 'B').
        sample_index: Index of the sample for logging purposes.
        
    Returns:
        The extracted channel value or np.nan if extraction fails.
    """

    shape_color = processed_sample['shape_color']
        
    # Handle numpy arrays
    if isinstance(shape_color, np.ndarray):
        channel_indices = {"R": 0, "G": 1, "B": 2}
        idx = channel_indices.get(channel)
        if idx is not None and idx < shape_color.shape[0]:
            return float(shape_color[idx])
    
    # Handle lists/tuples
    elif isinstance(shape_color, (list, tuple)):
        channel_indices = {"R": 0, "G": 1, "B": 2}
        idx = channel_indices.get(channel)
        if idx is not None and idx < len(shape_color):
            return float(shape_color[idx])
    
    # Debug information
    if 'shape_color' in processed_sample:
        warnings.warn(f"Shape color data type: {type(processed_sample['shape_color'])}")
        warnings.warn(f"Shape color value: {processed_sample['shape_color']}")
    else:
        warnings.warn(f"No shape_color key in processed sample at index {sample_index}")
        
    warnings.warn(f"Could not extract {channel} channel data from processed sample at index {sample_index}")
    return np.nan


def rgb_to_hls(rgb_values):
    """
    Convert RGB values (0-255) to HSV values
    Returns H in range [0,360), S and V in range [0,1]
    """
    # Normalize RGB to [0,1] range
    r, g, b = [x/255.0 for x in rgb_values]
    h, s, v = colorsys.rgb_to_hls(r, g, b)
    # Convert H to degrees [0,360)
    h = h * 360
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return h, r, g, b


def initialize_h_binning_structures(
    analysis_attributes,
    binning_config, 
    rgb = False
):
    """
    Initialize data structures for binning samples by attribute for H channel.
    Creates separate dictionaries for each processing path.
    """
    input_colors_by_attr = {}
    translated_colors_by_attr = {}
    half_cycle_colors_by_attr = {}
    full_cycle_colors_by_attr = {}
    examples_by_attr = {}

    for attr in analysis_attributes:
        input_bins = {}
        translated_bins = {}
        half_cycle_bins = {}
        full_cycle_bins = {}
        example_bins = {}
        
        if not rgb:
            for bin_name in binning_config[attr]['bin_names']:
                input_bins[bin_name] = {'H': []}
                translated_bins[bin_name] = {'H': []}
                half_cycle_bins[bin_name] = {'H': []}
                full_cycle_bins[bin_name] = {'H': []}
                example_bins[bin_name] = []
        else : 
            for bin_name in binning_config[attr]['bin_names']:
                input_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                translated_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                half_cycle_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                full_cycle_bins[bin_name] = {'H': [], 'R': [], 'G': [], 'B': []}
                example_bins[bin_name] = []

        input_colors_by_attr[attr] = input_bins
        translated_colors_by_attr[attr] = translated_bins
        half_cycle_colors_by_attr[attr] = half_cycle_bins
        full_cycle_colors_by_attr[attr] = full_cycle_bins
        examples_by_attr[attr] = example_bins

    return input_colors_by_attr, translated_colors_by_attr, half_cycle_colors_by_attr, full_cycle_colors_by_attr, examples_by_attr

def bin_h_processed_samples_with_paths(
            preprocessed_samples,
            processed_samples,
            analysis_attributes,
            binning_config,
            input_colors_by_attr,
            translated_colors_by_attr,
            half_cycle_colors_by_attr,
            full_cycle_colors_by_attr,
            examples_by_attr,
            display_examples=True, 
            rgb=False
        ):
        """
        Bins processed samples by attribute values and extracts Hue channel values for all paths.
        This is an improved version of bin_h_processed_samples that tracks all paths separately.
        
        Args:
            preprocessed_samples: List of preprocessed samples
            processed_samples: List of processed samples from process_through_global_workspace
            analysis_attributes: List of attributes to analyze
            binning_config: Configuration for binning
            input_colors_by_attr: Dict to store input color distributions
            output_colors_by_attr: Dict to store output color distributions (translated path)
            half_cycle_colors_by_attr: Dict to store half-cycle color distributions
            full_cycle_colors_by_attr: Dict to store full-cycle color distributions
            examples_by_attr: Dict to store example images by attribute and bin
            display_examples: Whether to store example images
        """
        for idx, (preproc, processed) in enumerate(zip(preprocessed_samples, processed_samples)):
            # Extract attribute values directly from preprocessed samples
            shape = preproc.get('model_inputs').get('one_hot').argmax().item()
            rotation = preproc.get('rotation')
            size = preproc.get('size')
            position_x = preproc.get('position_x')
            position_y = preproc.get('position_y')

            # Extract RGB channels from colors
            input_color = preproc.get('original_color')
            input_h, input_r, input_g, input_b = rgb_to_hls(input_color)
            # Get output colors from all paths
            translated_h, translated_r, translated_g, translated_b = rgb_to_hls(processed['translated_shape_color'])
            half_cycle_h, half_cycle_r, half_cycle_g, half_cycle_b = rgb_to_hls(processed['half_cycle_shape_color'])
            full_cycle_h, full_cycle_r, full_cycle_g, full_cycle_b = rgb_to_hls(processed['full_cycle_shape_color'])
            
            
            
            for attr in analysis_attributes:
                if attr == 'shape' and shape is not None:
                    # Shape is categorical, so use name directly
                    bin_name = binning_config['shape']['bin_names'][shape]
                    bin_attr_value = shape
                elif attr == 'rotation' and rotation is not None:
                    # Use raw rotation value for binning
                    bin_idx = bin_attribute(rotation, binning_config['rotation']['n_bins'], 
                                        binning_config['rotation']['range'])
                    bin_name = binning_config['rotation']['bin_names'][bin_idx]
                    bin_attr_value = rotation
                    
                elif attr == 'size' and size is not None:
                    # Use raw size value for binning
                    bin_idx = int(size)-7
                    bin_name = binning_config['size']['bin_names'][bin_idx]
                    bin_attr_value = size
                elif attr == 'position_x' and position_x is not None:
                    # Use raw x-position for binning
                    bin_idx = bin_attribute(position_x, binning_config['position_x']['n_bins'], 
                                        binning_config['position_x']['range'])
                    bin_name = binning_config['position_x']['bin_names'][bin_idx]
                    bin_attr_value = position_x

                    print(f"Position X: {position_x}, Bin Name: {bin_name}")
                    print(f"Bin Index: {bin_idx}, Bin Names: {binning_config['position_x']['bin_names']}") 
                    print(f"Bin Range: {binning_config['position_x']['range']}")
                    
                elif attr == 'position_y' and position_y is not None:
                    # Use raw y-position for binning
                    bin_idx = bin_attribute(position_y, binning_config['position_y']['n_bins'], 
                                        binning_config['position_y']['range'])
                    bin_name = binning_config['position_y']['bin_names'][bin_idx]
                    bin_attr_value = position_y
                else:
                    continue
                
                # Add hue values to the appropriate bins for each path
                if bin_name in input_colors_by_attr[attr]:
                    input_colors_by_attr[attr][bin_name]['H'].append(input_h)
                    translated_colors_by_attr[attr][bin_name]['H'].append(translated_h)
                    half_cycle_colors_by_attr[attr][bin_name]['H'].append(half_cycle_h)
                    full_cycle_colors_by_attr[attr][bin_name]['H'].append(full_cycle_h)

                    if rgb:
                        input_colors_by_attr[attr][bin_name]['R'].append(input_r)
                        input_colors_by_attr[attr][bin_name]['G'].append(input_g)
                        input_colors_by_attr[attr][bin_name]['B'].append(input_b)
                        
                        translated_colors_by_attr[attr][bin_name]['R'].append(translated_r)
                        translated_colors_by_attr[attr][bin_name]['G'].append(translated_g)
                        translated_colors_by_attr[attr][bin_name]['B'].append(translated_b)

                        half_cycle_colors_by_attr[attr][bin_name]['R'].append(half_cycle_r)
                        half_cycle_colors_by_attr[attr][bin_name]['G'].append(half_cycle_g)
                        half_cycle_colors_by_attr[attr][bin_name]['B'].append(half_cycle_b)

                        full_cycle_colors_by_attr[attr][bin_name]['R'].append(full_cycle_r)
                        full_cycle_colors_by_attr[attr][bin_name]['G'].append(full_cycle_g)
                        full_cycle_colors_by_attr[attr][bin_name]['B'].append(full_cycle_b)
                    
                    # Store example if needed
                    if display_examples and len(examples_by_attr[attr][bin_name]) < 5:
                        examples_by_attr[attr][bin_name].append({
                            'input_image': preproc.get('visual_ground_truth'),
                            'translated_image': processed.get('translated_image'),
                            'half_cycle_image': processed.get('half_cycle_image'),
                            'full_cycle_image': processed.get('full_cycle_image'),
                            'attr_value': bin_attr_value
                        })


def save_binned_results(output_dir, input_colors_by_attr, translated_colors_by_attr, 
                        half_cycle_colors_by_attr, full_cycle_colors_by_attr, 
                        examples_by_attr, binning_config, analysis_attributes):
    """
    Save binned results to disk for later reuse.
    
    Args:
        output_dir: Directory to save results to
        input_colors_by_attr: Dict of input color distributions
        translated_colors_by_attr: Dict of translated color distributions
        half_cycle_colors_by_attr: Dict of half-cycle color distributions
        full_cycle_colors_by_attr: Dict of full-cycle color distributions
        examples_by_attr: Dict of example images by attribute and bin
        binning_config: Configuration used for binning
        analysis_attributes: List of attributes that were analyzed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dictionaries containing the binning results
    result_data = {
        'input_colors_by_attr': input_colors_by_attr,
        'translated_colors_by_attr': translated_colors_by_attr,
        'half_cycle_colors_by_attr': half_cycle_colors_by_attr,
        'full_cycle_colors_by_attr': full_cycle_colors_by_attr,
        'examples_by_attr': examples_by_attr,
        'binning_config': binning_config,
        'analysis_attributes': analysis_attributes
    }
    
    save_path = os.path.join(output_dir, 'binned_results.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(result_data, f)
    

def load_binned_results(output_dir):
    """
    Load previously saved binned results from disk.
    
    Args:
        output_dir: Directory where results were saved
        
    Returns:
        Tuple of (input_colors_by_attr, translated_colors_by_attr, 
                 half_cycle_colors_by_attr, full_cycle_colors_by_attr, 
                 examples_by_attr, binning_config, analysis_attributes)
        Returns None if no saved results are found
    """
    save_path = os.path.join(output_dir, 'binned_results.pkl')
    
    if not os.path.exists(save_path):
        print(f"No saved binned results found at {save_path}")
        return None
    
    
    

    with open(save_path, 'rb') as f:
        result_data = pickle.load(f)
    
    
    
    # Extract and return the individual components
    return (
        result_data['input_colors_by_attr'],
        result_data['translated_colors_by_attr'],
        result_data['half_cycle_colors_by_attr'],
        result_data['full_cycle_colors_by_attr'],
        result_data['examples_by_attr'],
        result_data['binning_config'],
        result_data['analysis_attributes']
    )

def comparison_metrics(values1, values2, num_bins):

        # KL(Bin1 || Bin2) - Order matters!
        kl_h_12 = kl_divergence(values1, values2)
        # KL(Bin2 || Bin1)
        kl_h_21 = kl_divergence(values2, values1)
        # Symmetric KL (Average)
        kl_h_sym = (kl_h_12 + kl_h_21) / 2.0 if np.isfinite(kl_h_12) and np.isfinite(kl_h_21) else np.inf

        # Determine the range for histograms
        min_val = min(np.min(values1) if len(values1) > 0 else 0, np.min(values2) if len(values2) > 0 else 0)
        max_val = max(np.max(values1) if len(values1) > 0 else 255, np.max(values2) if len(values2) > 0 else 255)
        
        # Create histograms and normalize to get probability distributions
        hist1, bin_edges = np.histogram(values1, bins=num_bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(values2, bins=num_bins, range=(min_val, max_val), density=True)
        
        # Get bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create arrays with bin values repeated according to frequencies
        # Multiply by 1000 and round to get integer counts
        binned_dist1 = np.repeat(bin_centers, np.round(hist1 * 1000).astype(int))
        binned_dist2 = np.repeat(bin_centers, np.round(hist2 * 1000).astype(int))
        
        # Normalize values to 0-1 range for KS test
        range_size = max_val - min_val
        if range_size > 0:
            binned_dist1_norm = (binned_dist1 - min_val) / range_size
            binned_dist2_norm = (binned_dist2 - min_val) / range_size
            
            # Perform KS test on the binned distributions
            ks_result = ks_2samp(binned_dist1_norm, binned_dist2_norm)
            ks_stat = ks_result.statistic
            ks_pval = ks_result.pvalue
        else:
            LOGGER.warning(f"Cannot normalize distributions - all values are the same: {min_val}")
            ks_stat = 0.0 if np.array_equal(values1, values2) else 1.0
            ks_pval = 1.0 if np.array_equal(values1, values2) else 0.0

        return kl_h_12, kl_h_21, kl_h_sym, ks_stat, ks_pval



binning_config_6144 = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 16,
        'range': (0, 2 * np.pi),
        'bin_names': [f'{i}*2pi/16' for i in range(1, 17)]
    },
    'size': {
        'n_bins': 8,
        'range': (7, 14),
        'bin_names': [f"{i}" for i in range(0, 8)]
    },
    'position_x': {
        'n_bins': 4,
        'range': (7, 25), # Assuming 32x32 images
        'bin_names': ['Left', 'Middle-Left', 'Middle-Right', 'Right']
    },
    'position_y': {
        'n_bins': 4,
        'range': (7, 25), # Assuming 32x32 images
        'bin_names': ['Bottom', 'Low-Middle', 'High-Middle', 'Top']
    }
}


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
