"""Retrouve sur le disque ce qui a déjà été téléchargé ou généré, et le range dans le dépôt.

    python scripts/adopt_existing_files.py              # cherche dans ~ et affiche le plan
    python scripts/adopt_existing_files.py ~ /media     # plusieurs dossiers de recherche
    python scripts/adopt_existing_files.py --yes        # sans demander de confirmation

Éléments cherchés, et où ils sont rangés :
  - VAE pré-entraîné (domain_v.ckpt)                    -> checkpoints/pretrained/
  - les 20 modèles du mémoire, reconnus par leur contenu -> checkpoints/runs/<nom d'origine>/
  - jeu de test complet (153 600 images)                 -> data/evaluation_set/
  - mini-jeu de test (192 images)                        -> data/evaluation_set_quick/
  - Simple Shapes Dataset (entraînement)                 -> data/simple_shapes_dataset/

Les fichiers .ckpt sont COPIÉS (petits, et cela laisse intactes les archives d'origine) ;
les dossiers de données sont DÉPLACÉS (volumineux). Rien n'est fait sans confirmation.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SSD_eval"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import paths  # noqa: E402
from identify_checkpoints import fingerprint_matches, manifest_runs, read_ckpt_metadata  # noqa: E402

# Dossiers sans intérêt, ignorés pendant la recherche (gain de temps)
SKIP_DIRS = {".git", ".cache", ".conda", ".venv", "venv", "node_modules", "site-packages",
             "miniconda3", "anaconda3", "miniforge3", ".npm", ".mozilla", ".config", "snap",
             "__pycache__", "wandb"}
MAX_CKPT_MB = 300  # les .ckpt plus gros ne peuvent pas être ceux du projet

FULL_EVALSET_ROWS = 153_600
QUICK_EVALSET_ROWS = 192


def count_csv_rows(csv_path: Path) -> int:
    with open(csv_path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def in_trash(path: Path) -> bool:
    return "/.local/share/Trash/" in str(path)


def scan(roots):
    """Parcourt les dossiers et renvoie les candidats trouvés."""
    found = {"ckpt": [], "evalset": [], "quick": [], "dataset": []}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS and not x.startswith(".")
                           or x == ".local"]
            names = set(filenames)
            if "attributes.csv" in names and any(n.endswith(".png") for n in filenames[:50] + filenames[-50:]):
                rows = count_csv_rows(d / "attributes.csv")
                if rows == FULL_EVALSET_ROWS:
                    found["evalset"].append(d)
                elif rows == QUICK_EVALSET_ROWS:
                    found["quick"].append(d)
                dirnames[:] = []  # ne pas descendre dans les 153 600 images
                continue
            if {"train", "val"} <= set(dirnames) and any(n.endswith("labels.npy") for n in names):
                found["dataset"].append(d)
                dirnames[:] = []
                continue
            for name in filenames:
                p = d / name
                if name.endswith(".ckpt") and not p.is_symlink():
                    try:
                        if p.stat().st_size <= MAX_CKPT_MB * 1e6:
                            found["ckpt"].append(p)
                    except OSError:
                        pass
    return found


def pick(candidates):
    """Préfère un candidat hors de la corbeille, puis le plus récent."""
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (in_trash(p), -p.stat().st_mtime))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path.home()])
    parser.add_argument("--yes", action="store_true", help="Appliquer sans demander confirmation")
    args = parser.parse_args()

    print(f"Recherche dans : {', '.join(map(str, args.roots))} (quelques minutes possibles)…")
    found = scan(args.roots)
    plan = []          # (action, source, destination, description)
    missing = []

    def plan_dir(kind, dest, label):
        if dest.exists() and any(dest.iterdir()):
            print(f"  [déjà en place] {label} : {dest}")
            return
        cands = [c for c in found[kind] if not is_inside(c, dest)]
        src = pick(cands)
        if src is None:
            missing.append(label)
        else:
            plan.append(("déplacer", src, dest, label))

    # 1. Données
    plan_dir("evalset", paths.EVALSET_DIR, "jeu de test complet")
    plan_dir("quick", paths.DATA_DIR / "evaluation_set_quick", "mini-jeu de test")
    plan_dir("dataset", paths.DATASET_DIR, "Simple Shapes Dataset (entraînement, facultatif pour l'évaluation)")

    # 2. VAE pré-entraîné
    for dest in (paths.DOMAIN_V_CKPT,):  # domain_attr.ckpt n'est pas chargé par les variantes « legacy »
        if dest.exists() and zipfile.is_zipfile(dest):
            print(f"  [déjà en place] {dest.name} : {dest}")
            continue
        cands = [c for c in found["ckpt"] if c.name == dest.name and zipfile.is_zipfile(c)
                 and not is_inside(c, dest.parent)]
        src = pick(cands)
        if src is None:
            missing.append(dest.name)
        else:
            plan.append(("copier", src, dest, dest.name))

    # 3. Les 20 modèles du mémoire, reconnus par leur empreinte
    manifest = yaml.safe_load(open(paths.THESIS_RUNS_FILE))
    runs = list(manifest_runs(manifest))
    gw_cands = [c for c in found["ckpt"] if c.name not in ("domain_v.ckpt", "domain_attr.ckpt")]
    infos = []
    for c in gw_cands:
        try:
            infos.append(read_ckpt_metadata(c))
        except OSError:
            pass
    n_models_missing = 0
    for exp_name, _exp, run in runs:
        label = f"{exp_name}/{run['run_id']}"
        dest = paths.gw_checkpoint(run["original_run_name"], run.get("original_ckpt", run["ckpt"]))
        if dest.exists():
            info = read_ckpt_metadata(dest)
            if info["valid_zip"] and fingerprint_matches(info, run["expect"]):
                continue
        matches = [i for i in infos if i["valid_zip"] and fingerprint_matches(i, run["expect"])
                   and not is_inside(Path(i["path"]), dest.parent)]
        named = [i for i in matches if i.get("original_dir") == run["original_run_name"]] or matches
        if len({i["sha256"] for i in named}) == 1:
            plan.append(("copier", Path(named[0]["path"]), dest, f"modèle {label}"))
        else:
            n_models_missing += 1
            missing.append(f"modèle {label}" + (" (plusieurs candidats différents)" if named else ""))
    n_models_ok = len(runs) - n_models_missing

    # Affichage du plan
    print("\nPlan :")
    if not plan:
        print("  rien à déplacer.")
    for action, src, dest, label in plan:
        print(f"  {action:9s} {label}\n            {src}\n         -> {dest}")
    print(f"\nModèles du mémoire disponibles après opération : {n_models_ok}/{len(runs)}")
    if missing:
        print("Introuvables :\n  - " + "\n  - ".join(missing))

    if plan and (args.yes or input("\nAppliquer ce plan ? [o/N] ").strip().lower() in ("o", "oui", "y", "yes")):
        for action, src, dest, label in plan:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if action == "copier":
                shutil.copy2(src, dest)
            else:
                if dest.exists():
                    dest.rmdir()  # dossier vide uniquement (vérifié plus haut)
                shutil.move(str(src), str(dest))
            print(f"  ok : {label}")
    elif plan:
        print("Rien n'a été modifié.")

    # Ce qu'il reste à faire
    print("\nPour la suite :")
    if any(m.startswith(("jeu de test complet", "domain_v")) for m in missing):
        print("  bash scripts/download_data.sh --eval-only        # télécharge / génère ce qui manque")
    if n_models_missing:
        print("  bash scripts/download_checkpoints.sh             # télécharge les modèles depuis Zenodo")
    print("  bash scripts/evaluate_all.sh --use-original-names")


if __name__ == "__main__":
    main()
