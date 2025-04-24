from SSD_utils import _process_full_cycle_path
from SSD_eval_regularity import eval_regularity, load_global_workspace
from SSD_utils import _prepare_inputs
from PIL import Image
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig
from pathlib import Path
from torchvision.transforms.functional import to_tensor
import numpy as np
from typing import Any, Tuple, List
from SSD_H_evaluation_functions import segment_shape, extract_shape_color
import torch
from shimmer.modules import SingleDomainSelection
import matplotlib.pyplot as plt
from tqdm import tqdm
from SSD_utils import preprocess_dataset
import pandas as pd
from SSD_utils import generate_fixed_colors, _prepare_inputs
import ast
import os
from shimmer.modules.selection import SelectionBase
from shimmer.types import LatentsDomainGroupT

rgb_colors, _  = generate_fixed_colors(100)
##################################   ATTRIBUTES   ################################
csv_path = "evaluation_set/attributes.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(csv_path)
df["location"] = df["location"].apply(ast.literal_eval)

preprocessed_samples = preprocess_dataset(df, device=device, analysis_attributes=['shape', 'rotation', 'size', 'position_x', 'position_y'], 
    shape_names=["diamond", "egg", "triangle"], color_mode=False, rgb_colors=rgb_colors)

###########################################################################""


class SingleDomainSelection_(SelectionBase):

    def __init__(self, domain: str):
        super().__init__()
        self.domain = domain

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        if self.domain=='attr':
            return {'attr': torch.tensor([1.], device="cuda")}
        else:
            return {'v_latents': torch.tensor([1.], device="cuda")}


def process_translation_path(global_workspace: Any, attr_inputs: List, device: torch.device, n: int = 1, selection_module = SingleDomainSelection()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            selection_module=selection_module_attr
        )
        
        # Decode GW -> Visual Latent
        v_latent = global_workspace.gw_mod.decode(attr_gw_latent)["v_latents"]
        
        # For additional cycles, we'd need to go back to attribute domain
        if _ < n - 1:
            # Encode Visual Latent -> GW
            gw_latent_from_v = global_workspace.gw_mod.encode_and_fuse(
                {"v_latents": v_latent}, 
                selection_module=selection_module_v
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



# Args:
my_hparams = {
    "temperature": 0.1,
    "alpha": 0.1,
}
checkpoint_base_params = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params/Removing_colors/Base_params/version_colorTrue_None/checkpoints/epoch=1202.ckpt"
checkpoint_high_cycle_300k = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/high_cycle/Removing_colors/high_cycle/version_colorTrue_None/checkpoints/last.ckpt"

checkpoint = checkpoint_high_cycle_300k
# Configure the model
config = {
    "domains": [
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.v_latents,
            checkpoint_path=Path("checkpoints/domain_v.ckpt"),
        ),
        LoadedDomainConfig(
            domain_type=DomainModuleVariant.attr_legacy_no_color,
            checkpoint_path=Path("checkpoints/domain_attr.ckpt"),
            args=my_hparams,
        ),
    ],
    "global_workspace": {
        "latent_dim": 12,
        "encoders": {"hidden_dim": 256, "n_layers": 2},
        "decoders": {"hidden_dim": 256, "n_layers": 2},
    },
}

# Load the global workspace
global_workspace = load_global_workspace(
    gw_checkpoint=checkpoint,
    config=config
)

# Create selection module
selection_module_attr = SingleDomainSelection_(domain='attr')
selection_module_v = SingleDomainSelection_(domain='v')

for sample in tqdm(preprocessed_samples, desc="Processing samples through GW model"):

    attr_inputs, visual_ground_truth_tensor = _prepare_inputs(sample, device)


    # Get visual latent vector from ground truth image
    visual_module = global_workspace.domain_mods["v_latents"]
    v_latent_vector = visual_module.visual_module.encode(visual_ground_truth_tensor)

    # Run the pipeline for different values of n
    n_values = [i for i in range(1, 100)]
    # Create a directory to save the images
    output_dir = Path("reconstructed_cycle_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the original image  
    # image.save(output_dir / "original_image.png")


    # for n in n_values:
    #     print(f"Processing with n = {n}")
    #     recons, _, _ = _process_full_cycle_path(
    #         global_workspace=global_workspace,
    #         v_latent_vector=v_latent_vector, 
    #         device="cuda", 
    #         n=n
    #     )
        
    #     # Convert to appropriate format for saving
    #     if recons.max() <= 1.0:
    #         recons = (recons * 255).astype(np.uint8)
            
    #     # If recons has shape (C,H,W), transpose to (H,W,C)
    #     if recons.shape[0] == 3 and len(recons.shape) == 3:
    #         recons = recons.transpose(1, 2, 0)
            
    #     # Save the reconstructed image
    #     recons_img = Image.fromarray(recons)
    #     recons_img.save(output_dir / f"recons_n{n}.png")
        


    # print("Processing complete. All images saved.")

    image = sample["visual_ground_truth"]
    reconstructed_colors = []
    image_np = np.array(image)
    original_rgb = extract_shape_color(image=image_np, mask = segment_shape(image=image))

    for n in n_values:
        # Reload the saved shape color or recalculate if needed
        translated, mask , shape_color = process_translation_path(
                    global_workspace=global_workspace,
                    attr_inputs=attr_inputs, 
                    device="cuda", 
                    n=n
                )

        reconstructed_colors.append(shape_color)
    # Convert list to numpy array
    reconstructed_colors = np.array(reconstructed_colors)

    # Plot RGB component changes
    plt.figure(figsize=(12, 8))
    plt.plot(n_values, reconstructed_colors[:, 0], 'r-o', label='R Component')
    plt.plot(n_values, reconstructed_colors[:, 1], 'g-o', label='G Component')
    plt.plot(n_values, reconstructed_colors[:, 2], 'b-o', label='B Component')

    # Add horizontal lines for original RGB values
    plt.axhline(y=original_rgb[0], color='r', linestyle='--', alpha=0.5, label='Original R')
    plt.axhline(y=original_rgb[1], color='g', linestyle='--', alpha=0.5, label='Original G')
    plt.axhline(y=original_rgb[2], color='b', linestyle='--', alpha=0.5, label='Original B')

    plt.xlabel('Number of Cycles (n)', fontsize=12)
    plt.ylabel('RGB Component Value', fontsize=12)
    plt.title('RGB Component Change with Increasing Cycles', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values[::5], n_values[::5], fontsize=10)
    plt.legend()
    plt.tight_layout()

    # Save the RGB component plot
    plt.savefig("rgb_components_vs_cycles.png", dpi=300)
    print("RGB component plot saved as rgb_components_vs_cycles.png")

    # Create a visualization of the actual colors
    plt.figure(figsize=(12, 3))
    ax = plt.subplot(111)

    # Create rectangles for each color
    width = 1.0 / (len(n_values) + 1)
    # Original color first
    rect = plt.Rectangle((0, 0), width, 1, color=tuple(original_rgb/255))
    ax.add_patch(rect)
    plt.text(width/2, 0.5, "Original", ha='center', va='center')

    # Add reconstructed colors
    for i, n in enumerate(n_values):
        color = tuple(reconstructed_colors[i]/255)
        rect = plt.Rectangle((width * (i + 1), 0), width, 1, color=color)
        ax.add_patch(rect)
        if n%5==0:
            plt.text(width * (i + 1.5), 0.5, f"{n}", ha='center', va='center')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()

    # Save the color visualization
    plt.savefig("color_visualization.png", dpi=300)
    print("Color visualization saved as color_visualization.png")
    


























binning_config_6144 = {
    'shape': {
        'n_bins': 3,
        'range': None,
        'bin_names': ['diamond', 'egg', 'triangle']
    },
    'rotation': {
        'n_bins': 16,
        'range': (0, 2 * np.pi),
        'bin_names': [f'{i}/16*2pi' for i in range(16)]
    },
    'size': {
        'n_bins': 8,
        'range': (7, 14),
        'bin_names': [f"{i}" for i in range(1, 8)]
    },
    'position_x': {
        'n_bins': 4,
        'range': (0, 32), # Assuming 32x32 images
        'bin_names': ['Left', 'Middle-Left', 'Middle-Right', 'Right']
    },
    'position_y': {
        'n_bins': 4,
        'range': (0, 32), # Assuming 32x32 images
        'bin_names': ['Bottom', 'Low-Middle', 'High-Middle', 'Top']
    }
}