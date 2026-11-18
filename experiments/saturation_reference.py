"""Témoin de mesure pour la saturation : S et L des images d'entrée et du VAE seul.

Avant d'attribuer au Global Workspace une saturation qui varie avec la forme, on vérifie ce que
donne la même extraction (masque d'Otsu, barycentre, voir SSD_utils.extract_shape_color) :
  - sur les images d'entrée du jeu de test, sans aucun modèle ;
  - sur leur reconstruction par le VAE seul (image -> VAE -> image, sans GW) ;
avec, pour comparaison, la couleur moyenne de tout le masque au lieu du barycentre.

    python experiments/saturation_reference.py                 # 6 000 images tirées au hasard
    python experiments/saturation_reference.py --n-images 600 --dataset-csv data/evaluation_set_quick/attributes.csv

Sortie : results/saturation/temoin/temoin_S_L.md et temoin_S_L.csv.
"""

import argparse
import ast
import colorsys
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
import paths  # noqa: E402
import device_compat  # noqa: E402,F401
from SSD_utils import extract_shape_color, generate_fixed_colors, segment_shape  # noqa: E402

SHAPES = ["diamond", "egg", "triangle"]


def load_vae(device):
    from shimmer_ssd.config import DomainModuleVariant, LoadedDomainConfig
    from shimmer_ssd.modules.domains import load_pretrained_domains
    domain_modules, _, _ = load_pretrained_domains(
        [LoadedDomainConfig(domain_type=DomainModuleVariant.v_latents, checkpoint_path=paths.DOMAIN_V_CKPT)],
        12, 256, 2, 256, 2)
    return domain_modules["v_latents"].to(device).eval()


def hls(rgb255):
    h, l, s = colorsys.rgb_to_hls(*(np.asarray(rgb255, dtype=float) / 255))
    return h * 360, l, s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-csv", default=str(paths.EVALSET_DIR / "attributes.csv"))
    parser.add_argument("--n-images", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / "saturation" / "temoin")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset_csv)
    if isinstance(df["location"].iloc[0], str):
        df["location"] = df["location"].apply(ast.literal_eval)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(df), size=min(args.n_images, len(df)), replace=False))
    rgb_palette, _ = generate_fixed_colors(int(df["color_index"].max()) + 1)
    im_dir = Path(args.dataset_csv).parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device)
    to_tensor = ToTensor()
    rows = []
    for start in range(0, len(idx), 512):
        batch = idx[start:start + 512]
        images = [Image.open(im_dir / f"image_{i:05d}.png").convert("RGB") for i in batch]
        x = torch.stack([to_tensor(im)[:3] for im in images]).to(device)
        with torch.no_grad():
            recon = vae.decode_images(vae.visual_module.encode(x))
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()
        r_np = recon.permute(0, 2, 3, 1).cpu().numpy()
        for k, i in enumerate(batch):
            row = df.iloc[i]
            out = {"image": int(i), "forme": SHAPES[int(row["class"])], "taille": int(row["size"])}
            _, out["L_palette"], out["S_palette"] = hls(rgb_palette[int(row["color_index"])])
            for name, img in (("entree", x_np[k]), ("vae", r_np[k])):
                mask = segment_shape(img)
                for method, bary in (("barycentre", True), ("masque", False)):
                    _, l, s = hls(extract_shape_color(img, mask, bary) * 255)
                    out[f"S_{name}_{method}"], out[f"L_{name}_{method}"] = s, l
            rows.append(out)
        print(f"{start + len(batch)}/{len(idx)} images")

    res = pd.DataFrame(rows)
    res.to_csv(args.out / "temoin_S_L.csv", index=False)
    cols = ["S_palette", "S_entree_barycentre", "S_entree_masque", "S_vae_barycentre", "S_vae_masque",
            "L_palette", "L_entree_barycentre", "L_vae_barycentre"]
    lines = ["# Témoin de mesure : S et L sans Global Workspace\n",
             f"{len(res)} images du jeu de test (`{args.dataset_csv}`), tirées au hasard (graine {args.seed}).",
             "Médianes (S et L sur 0-1). « barycentre » : couleur moyenne de 3×3 pixels au barycentre du masque "
             "d'Otsu (méthode de l'évaluation) ; « masque » : couleur moyenne de tout le masque.\n"]
    for group in ("forme", "taille"):
        table = res.groupby(group)[cols].median()
        lines += [f"\n## Par {group}\n", "| " + group + " | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        for key, r in table.iterrows():
            lines.append(f"| {key} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    total = res[cols].median()
    lines += ["\n## Ensemble\n", "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|",
              "| " + " | ".join(f"{total[c]:.3f}" for c in cols) + " |"]
    (args.out / "temoin_S_L.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
