from SSD_utils import _process_full_cycle_path
from SSD_eval_regularity import eval_regularity, load_global_workspace
from SSD_utils import _prepare_inputs
from PIL import Image
from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig
from pathlib import Path
from torchvision.transforms.functional import to_tensor
import numpy as np
from typing import Any, Tuple
from SSD_H_evaluation_functions import segment_shape, extract_shape_color
import torch
from shimmer.modules import SingleDomainSelection
import matplotlib.pyplot as plt
from tqdm import tqdm

# Args:
my_hparams = {
    "temperature": 0.1,
    "alpha": 0.1,
}
checkpoint_base_params = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/Base_params/Removing_colors/Base_params/version_colorTrue_None/checkpoints/epoch=1202.ckpt"
checkpoint_high_cycle_300k = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/high_cycle/Removing_colors/high_cycle/version_colorTrue_None/checkpoints/last.ckpt"
checkpoint_low_temperature_01_300k_low_lr = "/home/alexis/Desktop/checkpoints/training_logs/Removing_colors/low_temperature_0.1_low_lr/checkpoints/epoch=1233.ckpt"
checkpoint_base_params_final_scheduler = "/home/alexis/Desktop/checkpoints/training_logs/DEBUG/Final_scheduler/checkpoints/epoch=1229.ckpt"


checkpoint = checkpoint_base_params_final_scheduler
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
selection_module = SingleDomainSelection()


# Load the image

for i in tqdm(range(0, 19200, 192)):
    image_path = f"evaluation_set/image_{i:05d}.png"
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_np = np.array(image)
# image_path = "evaluation_set_180/image_00242.png"
# with Image.open(image_path) as image:
#     image = image.convert("RGB")

# Load the checkpoint


    # Prepare inputs
    visual_ground_truth_tensor = to_tensor(image)[:3].unsqueeze(0).to("cuda")

    # Get visual latent vector from ground truth image
    visual_module = global_workspace.domain_mods["v_latents"]
    v_latent_vector = visual_module.visual_module.encode(visual_ground_truth_tensor)

    # Run the pipeline for different values of n
    n_values = [i for i in range(1, 100)]
    # Create a directory to save the images
    output_dir = Path("reconstructed_full_cycle_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the original image  
    image.save(output_dir / "original_image.png")


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

    reconstructed_images = [(image_np).astype(np.uint8)]
    reconstructed_colors = []
    
    original_rgb = extract_shape_color(image=image_np, mask = segment_shape(image=image))
    
    for n in n_values:
        # Reload the saved shape color or recalculate if needed
        recons_image, _, shape_color = _process_full_cycle_path(
                    global_workspace=global_workspace,
                    v_latent_vector=v_latent_vector, 
                    device="cuda", 
                    n=n
                )
        shape_color = shape_color
        reconstructed_images.append(recons_image)
        reconstructed_colors.append(shape_color)

    # Make a video of the list of reconstructed images : 
    import cv2
    out = cv2.VideoWriter(f'reverberation_reconstruction_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (reconstructed_images[0].shape[1], reconstructed_images[0].shape[0]))
    for img in reconstructed_images:
        # Ensure image is in proper shape (H, W, C)
        if img.shape[0] == 3 and len(img.shape) == 3:  # If in (C, H, W) format
            img = img.transpose(1, 2, 0)
        
        # Scale to [0, 255] if in [0, 1] range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        out.write(img_bgr)
    out.release()


       
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