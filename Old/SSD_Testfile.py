
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
from scipy.stats import entropy
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid

from simple_shapes_dataset import SimpleShapesDataModule, get_default_domains
from simple_shapes_dataset.cli import generate_image

# Additional imports for shape generation and evaluation
import os
import numpy as np
import cv2
import matplotlib.path as mpath
from matplotlib import patches
import csv
import argparse
import math
import io
import ast
from PIL import Image
import pandas as pd
from scipy.stats import ks_2samp

def generate_color(n_colors, seed=42, max_val=256):
    """
    Generate a set of unique RGB colors.
    
    Args:
        n_colors (int): Number of colors to generate
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        max_val (int, optional): Maximum color value. Defaults to 256.
    
    Returns:
        tuple: A tuple containing:
            - numpy array of RGB colors
            - list of color names (not implemented in this version)
    """
    np.random.seed(seed)
    
    # Generate unique colors by sampling in RGB space
    colors = np.random.randint(0, max_val, size=(n_colors, 3))
    
    # Optional: Ensure some color diversity by avoiding very similar colors
    unique_colors = []
    for color in colors:
        # Check if the color is sufficiently different from existing colors
        if not any(np.allclose(color, unique_color, atol=30) for unique_color in unique_colors):
            unique_colors.append(color)
        
        if len(unique_colors) == n_colors:
            break
    
    # If we couldn't generate enough unique colors, use the original random colors
    if len(unique_colors) < n_colors:
        unique_colors = colors[:n_colors]
    
    return np.array(unique_colors), [f"Color {i}" for i in range(n_colors)]


def load_global_workspace(gw_checkpoint: Path, config: dict) -> GlobalWorkspace2Domains:
    domain_modules, gw_encoders, gw_decoders = load_pretrained_domains(
        config["domains"],
        config["global_workspace"]["latent_dim"],
        config["global_workspace"]["encoders"]["hidden_dim"],
        config["global_workspace"]["encoders"]["n_layers"],
        config["global_workspace"]["decoders"]["hidden_dim"],
        config["global_workspace"]["decoders"]["n_layers"],
    )
    global_workspace = GlobalWorkspace2Domains.load_from_checkpoint(
        gw_checkpoint,
        domain_mods=domain_modules,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
    )
    return global_workspace


# # Les fonctions de transformation et création de patchs restent inchangées.
# def get_transformed_coordinates(coordinates: np.ndarray, origin: np.ndarray, scale: float, rotation: float) -> np.ndarray:
#     center = np.array([[0.5, 0.5]])
#     rotation_m = np.array([
#         [np.cos(rotation), -np.sin(rotation)],
#         [np.sin(rotation),  np.cos(rotation)]
#     ])
#     rotated_coordinates = (coordinates - center) @ rotation_m.T
#     return origin + scale * rotated_coordinates


# def get_image_array(cls, location, size, rotation, color, imsize=32):
#     dpi = 1
#     fig, ax = plt.subplots(figsize=(imsize/dpi, imsize/dpi), dpi=dpi)
#     generate_image(ax, cls, location, size, rotation, color, imsize)
#     plt.tight_layout(pad=0)
#     fig.canvas.draw()
#     width, height = fig.canvas.get_width_height()
#     # Utilisation de tostring_argb pour obtenir 4 canaux
#     data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
#     data = data.reshape((height, width, 4))
#     # On retire le canal alpha (le premier) pour obtenir RGB
#     rgb_data = data[:, :, 1:]
#     plt.close(fig)
#     return rgb_data.astype(np.float32) / 255.0  # Normalisation en [0,1]


# # --- Fonctions de normalisation des attributs ---
# def normalize_position(pos, imsize=32, margin=7):
#     # On suppose que la position est en pixels et que la marge minimale est 7.
#     return [(p - margin) / (imsize - 2 * margin) * 2 - 1 for p in pos]

# def normalize_size(size, min_scale=7, max_scale=14):
#     s = (size - min_scale) / (max_scale - min_scale)
#     return s * 2 - 1

# def normalize_rotation(rot):
#     return math.cos(rot), math.sin(rot)

# # --- Calcul d'un histogramme couleur ---
# def compute_color_histogram(image, bins=16):
#     # image : tableau HxWx3, valeurs dans [0,1]
#     hist_total = np.zeros((3, bins))
#     for channel in range(3):
#         hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 1))
#         hist = hist.astype(float)
#         if hist.sum() > 0:
#             hist /= hist.sum()
#         hist_total[channel] = hist
#     return hist_total

# # --- Attribution des images à un bin de rotation ---
# def get_rotation_bin(rot, n_bins=4):
#     # rot en radians, dans [0, 2π]
#     return int(rot / (2 * math.pi / n_bins)) % n_bins

# # --- Évaluation par divergence KL ---
# def evaluate_kl_divergence(csv_path, global_workspace, device, imsize=32, min_scale=7, max_scale=14):
#     import math
#     import ast
#     from scipy.stats import entropy, ks_2samp
#     import torch.nn.functional as F
#     import torch
#     import numpy as np
#     import pandas as pd
#     from tqdm import tqdm

#     df = pd.read_csv(csv_path)
#     df["location"] = df["location"].apply(ast.literal_eval)
    
#     # Nombre de bins pour la rotation
#     n_rot_bins = 4
#     # Dictionnaires pour accumuler les histogrammes moyens par bin (pour divergence KL)
#     orig_hist_sum = {i: np.zeros((3, 16)) for i in range(n_rot_bins)}
#     gw_hist_sum = {i: np.zeros((3, 16)) for i in range(n_rot_bins)}
#     count_per_bin = {i: 0 for i in range(n_rot_bins)}
    
#     # Dictionnaires pour accumuler toutes les valeurs de pixels pour le test KS
#     orig_pixels = {i: {ch: [] for ch in range(3)} for i in range(n_rot_bins)}
#     gw_pixels   = {i: {ch: [] for ch in range(3)} for i in range(n_rot_bins)}
    
#     # Pour re-générer la couleur : on suppose 100 couleurs et fixe la seed
#     n_colors = 100
#     np.random.seed(0)
#     rgb_colors, _ = generate_color(n_colors, 46, 256)
    
#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         cls = int(row["class"])
#         location = row["location"]
#         size = float(row["size"])
#         rotation = float(row["rotation"])
#         color_idx = int(row["color_index"])
#         color = rgb_colors[color_idx]
        
#         # Génère l'image originale à partir des attributs
#         orig_img = get_image_array(cls, location, size, rotation, color, imsize)
        
#         # Prépare les attributs normalisés pour le Global Workspace
#         pos_norm = normalize_position(location, imsize=imsize, margin=7)
#         size_norm = normalize_size(size, min_scale, max_scale)
#         rotx, roty = normalize_rotation(rotation)
#         attr_vector = torch.tensor([[pos_norm[0], pos_norm[1], size_norm, rotx, roty]], dtype=torch.float32)
#         one_hot = F.one_hot(torch.tensor([cls]), num_classes=3).float()
#         samples = [one_hot.to(device), attr_vector.to(device)]
        
#         # Passage par le Global Workspace pour obtenir l'image construite
#         attr_gw_latent = global_workspace.gw_mod.encode({"attr": global_workspace.encode_domain(samples, "attr")})
#         gw_latent = global_workspace.gw_mod.fuse(attr_gw_latent, {"attr": torch.ones(attr_gw_latent["attr"].size(0)).to(device)})
#         decoded_latents = global_workspace.gw_mod.decode(gw_latent)["v_latents"]
#         decoded_images_tensor = global_workspace.domain_mods["v_latents"].decode_images(decoded_latents)[0]
#         gw_img = decoded_images_tensor.permute(1, 2, 0).detach().cpu().numpy()
        
#         # Calcul des histogrammes couleur sur 16 bins pour chaque canal
#         hist_orig = compute_color_histogram(orig_img, bins=16)
#         hist_gw = compute_color_histogram(gw_img, bins=16)
        
#         # Détermination du bin de rotation (rotation ∈ [0,2π])
#         bin_index = int(rotation / (2 * math.pi / n_rot_bins)) % n_rot_bins
#         orig_hist_sum[bin_index] += hist_orig
#         gw_hist_sum[bin_index] += hist_gw
#         count_per_bin[bin_index] += 1
        
#         # Accumulation des valeurs de pixels pour KS : on stocke tous les pixels par canal
#         for ch in range(3):
#             orig_pixels[bin_index][ch].extend(orig_img[:, :, ch].flatten().tolist())
#             gw_pixels[bin_index][ch].extend(gw_img[:, :, ch].flatten().tolist())
    
#     # Calcul de la divergence KL moyenne par bin de rotation (moyenne sur les 3 canaux)
#     kl_divergences = {}
#     for bin_index in range(n_rot_bins):
#         if count_per_bin[bin_index] > 0:
#             avg_orig = orig_hist_sum[bin_index] / count_per_bin[bin_index]
#             avg_gw = gw_hist_sum[bin_index] / count_per_bin[bin_index]
#             kl_channels = []
#             for ch in range(3):
#                 eps = 1e-10
#                 p = avg_orig[ch] + eps
#                 q = avg_gw[ch] + eps
#                 p /= p.sum()
#                 q /= q.sum()
#                 kl = entropy(p, q)
#                 kl_channels.append(kl)
#             kl_divergences[bin_index] = np.mean(kl_channels)
#         else:
#             kl_divergences[bin_index] = None
#     for bin_index, kl_val in kl_divergences.items():
#         print(f"Bin de rotation {bin_index}: divergence KL moyenne = {kl_val:.4f}")
    
#     # Tests de Kolmogorov–Smirnov pour chaque canal dans chaque bin
#     ks_results = {}
#     for bin_index in range(n_rot_bins):
#         ks_results[bin_index] = {}
#         for ch in range(3):
#             stat, p_value = ks_2samp(orig_pixels[bin_index][ch], gw_pixels[bin_index][ch])
#             ks_results[bin_index][ch] = (stat, p_value)
#             print(f"Bin de rotation {bin_index}, Canal {ch}: KS stat = {stat:.4f}, p = {p_value:.4e}")
    
#     return kl_divergences, ks_results



# # --- Exemple d'utilisation ---
# if __name__ == "__main__":
#     # Chemin vers le CSV généré (par exemple lors de la génération du SSD)
#     csv_path = Path("./evaluation_set/attributes.csv")
    
#     # Chargement du Global Workspace (remplacez les chemins et config par les vôtres)
#     gw_checkpoint = Path("checkpoints/gw/version_None/epoch=120.ckpt")
#     config = {
#         "domains": [
#             LoadedDomainConfig(domain_type=DomainModuleVariant.v_latents, checkpoint_path=Path("./checkpoints/domain_v.ckpt")),
#             LoadedDomainConfig(domain_type=DomainModuleVariant.attr_legacy, checkpoint_path=Path("./checkpoints/domain_attr.ckpt")),
#         ],
#         "global_workspace": {
#             "latent_dim": 12,
#             "encoders": {"hidden_dim": 32, "n_layers": 1},
#             "decoders": {"hidden_dim": 32, "n_layers": 1},
#         },
#     }
#     global_workspace = load_global_workspace(gw_checkpoint, config)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     global_workspace.to(device)
    
#     # Calcul et affichage de la divergence KL moyenne par bin de rotation
#     kl_divs = evaluate_kl_divergence(csv_path, global_workspace, device, imsize=32, min_scale=7, max_scale=14)

def get_image_array(cls, location, size, rotation, color, imsize=32):
    """
    Génère l'image sous forme de tableau numpy en normalisant les valeurs entre 0 et 1.
    On utilise tostring_argb pour récupérer 4 canaux puis on enlève l'alpha.
    """
    import matplotlib.pyplot as plt
    dpi = 1
    fig, ax = plt.subplots(figsize=(imsize/dpi, imsize/dpi), dpi=dpi)
    generate_image(ax, cls, location, size, rotation, color, imsize)
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    data = data.reshape((height, width, 4))
    # Conserver uniquement les canaux R, G et B (en enlevant le canal alpha)
    rgb_data = data[:, :, 1:]
    plt.close(fig)
    return rgb_data.astype(np.float32) / 255.0

def compute_color_histogram(image, bins=16):
    """
    Calcule l'histogramme couleur pour chacun des 3 canaux d'une image (valeurs normalisées dans [0,1]).
    Retourne un tableau de forme (3, bins) avec des distributions normalisées.
    """
    hist_total = np.zeros((3, bins))
    for ch in range(3):
        hist, _ = np.histogram(image[:, :, ch], bins=bins, range=(0, 1))
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()
        hist_total[ch] = hist
    return hist_total

def normalize_position(pos, imsize=32, margin=7):
    # Normalise la position en pixels en [-1, 1]
    return [(pos[0] - margin) / (imsize - 2 * margin) * 2 - 1, (pos[1] - margin) / (imsize - 2 * margin) * 2 - 1]

def normalize_size(size, min_scale=7, max_scale=14):
    s = (size - min_scale) / (max_scale - min_scale)
    return s * 2 - 1

def normalize_rotation(rot):
    return math.cos(rot), math.sin(rot)

def generate_fixed_colors(n_samples: int, min_lightness: int = 46, max_lightness: int = 256):
    """
    Génère n_samples couleurs en HLS puis les convertit en RGB.
    """
    import cv2
    hls = np.random.randint(
        [0, min_lightness, 0],
        [181, max_lightness, 256],
        size=(1, n_samples, 3),
        dtype=np.uint8,
    )
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]
    return rgb.astype(int), hls[0].astype(int)

# --- Fonction d'évaluation par rotation ---
def evaluate_by_rotation(csv_path: str, global_workspace, device, imsize=32, min_scale=7, max_scale=14, n_bins=4, colors_flag = True):
    """
    Pour chaque exemple du dataset (défini dans csv_path), cette fonction :
      - Génère l'image originale à partir des attributs.
      - Passe les attributs dans le Global Workspace pour obtenir l'image reconstruite.
      - Calcule un histogramme couleur (16 bins par canal) pour chaque image.
      - Regroupe les exemples selon un bin de rotation (l'intervalle [0,2π] est découpé en n_bins).
      - Accumule les histogrammes pour chaque bin, puis calcule la divergence KL moyenne par canal.
      - Accumule aussi l'ensemble des valeurs de pixels par canal pour effectuer un test KS.
    """
    df = pd.read_csv(csv_path)
    df["location"] = df["location"].apply(ast.literal_eval)

    # Dictionnaires pour accumuler les histogrammes et valeurs de pixels par bin de rotation
    rot_hist_orig = {i: np.zeros((3, 16)) for i in range(n_bins)}
    rot_hist_gw   = {i: np.zeros((3, 16)) for i in range(n_bins)}
    counts = {i: 0 for i in range(n_bins)}
    pixels_orig = {i: {ch: [] for ch in range(3)} for i in range(n_bins)}
    pixels_gw   = {i: {ch: [] for ch in range(3)} for i in range(n_bins)}

    # On recrée les couleurs utilisées lors de la génération
    n_colors = 100
    np.random.seed(0)
    rgb_colors, _ = generate_fixed_colors(n_colors, min_lightness=46, max_lightness=256)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cls = int(row["class"])
        location = row["location"]
        size = float(row["size"])
        rotation = float(row["rotation"])
        color_idx = int(row["color_index"])
        color = rgb_colors[color_idx]

        # Génération de l'image originale
        orig_img = get_image_array(cls, location, size, rotation, color, imsize)

        # Préparation des attributs normalisés pour le Global Workspace
        pos_norm = normalize_position(location, imsize=imsize, margin=7)
        size_norm = normalize_size(size, min_scale, max_scale)
        rotx, roty = normalize_rotation(rotation)
        attr_vector = torch.tensor([[pos_norm[0], pos_norm[1], size_norm, rotx, roty]], dtype=torch.float32)
        
        case = "no_color"
        if colors_flag : 
            case = "color"
            # Get RGB color from color index
            color_idx = int(row["color_index"])
            np.random.seed(0)  # Same seed as dataset generation
            rgb_colors, _ = generate_fixed_colors(100)  # Assumes n_colors=100
            color_rgb = rgb_colors[color_idx]
            # Normalize RGB values to [-1, 1] range
            color_norm = [color_rgb[0]/127.5 - 1, color_rgb[1]/127.5 - 1, color_rgb[2]/127.5 - 1]
            # Update attr_vector to include RGB values
            attr_vector = torch.tensor([[pos_norm[0], pos_norm[1], size_norm, rotx, roty, 
                                       color_norm[0], color_norm[1], color_norm[2]]], 
                                       dtype=torch.float32)
        one_hot = F.one_hot(torch.tensor([cls]), num_classes=3).float()
        samples = [one_hot.to(device), attr_vector.to(device)]


        # Passage par le Global Workspace
        attr_gw_latent = global_workspace.gw_mod.encode({"attr": global_workspace.encode_domain(samples, "attr")})
        gw_latent = global_workspace.gw_mod.fuse(attr_gw_latent, {"attr": torch.ones(attr_gw_latent["attr"].size(0)).to(device)})
        decoded_latents = global_workspace.gw_mod.decode(gw_latent)["v_latents"]
        decoded_images_tensor = global_workspace.domain_mods["v_latents"].decode_images(decoded_latents)[0]
        gw_img = decoded_images_tensor.permute(1, 2, 0).detach().cpu().numpy()

        # Calcul des histogrammes couleur pour l'image originale et reconstruite
        hist_orig = compute_color_histogram(orig_img, bins=16)
        hist_gw = compute_color_histogram(gw_img, bins=16)

        # Détermination du bin de rotation (rotation dans [0,2π])
        bin_idx = int(rotation / (2 * math.pi / n_bins)) % n_bins

        rot_hist_orig[bin_idx] += hist_orig
        rot_hist_gw[bin_idx]   += hist_gw
        counts[bin_idx] += 1

        # Accumulation des valeurs de pixels pour le test KS (par canal)
        for ch in range(3):
            pixels_orig[bin_idx][ch].extend(orig_img[:, :, ch].flatten().tolist())
            pixels_gw[bin_idx][ch].extend(gw_img[:, :, ch].flatten().tolist())

    # Calcul de la divergence KL moyenne par bin de rotation
    kl_divergence = {}
    for i in range(n_bins):
        if counts[i] > 0:
            avg_hist_orig = rot_hist_orig[i] / counts[i]
            avg_hist_gw = rot_hist_gw[i] / counts[i]
            kl_channels = []
            for ch in range(3):
                eps = 1e-10
                p = avg_hist_orig[ch] + eps
                q = avg_hist_gw[ch] + eps
                p /= p.sum()
                q /= q.sum()
                kl = entropy(p, q)
                kl_channels.append(kl)
            kl_divergence[i] = np.mean(kl_channels)
        else:
            kl_divergence[i] = None
        print(f"Bin de rotation {i}: divergence KL moyenne = {kl_divergence[i]:.4f}")

    # Réalisation des tests KS pour chaque canal dans chaque bin
    ks_results = {}
    for i in range(n_bins):
        ks_results[i] = {}
        for ch in range(3):
            stat, p_value = ks_2samp(pixels_orig[i][ch], pixels_gw[i][ch])
            ks_results[i][ch] = (stat, p_value)
            print(f"Bin de rotation {i}, Canal {ch}: KS stat = {stat:.4f}, p = {p_value:.4e}")

    return kl_divergence, ks_results

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Chemin vers le CSV généré (par exemple lors de la génération du SSD)
    csv_path = Path("./evaluation_set/attributes.csv")
    
    # Chargement du Global Workspace (remplacer par vos chemins et config réels)
    gw_checkpoint = Path("checkpoints/gw-attr-v-all-paired-data.ckpt")
    # La configuration et le chargement du GW dépendent de votre implémentation.
    # Ici, nous utilisons une fonction de chargement stub.
    config = {
        "domains": [
            LoadedDomainConfig(domain_type=DomainModuleVariant.v_latents, checkpoint_path=Path("checkpoints/domain_v.ckpt")),
            LoadedDomainConfig(domain_type=DomainModuleVariant.attr, checkpoint_path=Path("checkpoints/domain_attr.ckpt")),
        ],
        "global_workspace": {
            "latent_dim": 12,
            "encoders": {"hidden_dim": 32, "n_layers": 3},
            "decoders": {"hidden_dim": 32, "n_layers": 3},
        },
    }
    global_workspace = load_global_workspace(gw_checkpoint, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_workspace.to(device)
    
    kl_divs, ks_res = evaluate_by_rotation(str(csv_path), global_workspace, device, imsize=32, min_scale=7, max_scale=14, n_bins=4)
