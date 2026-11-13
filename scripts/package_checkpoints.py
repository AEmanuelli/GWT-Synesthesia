"""Empaquette les checkpoints utilisés dans le mémoire pour publication (Zenodo / GitHub Release).

À lancer sur la machine qui contient les checkpoints d'origine, de préférence après
scripts/identify_checkpoints.py (qui retrouve les fichiers par leur contenu) :

    python scripts/package_checkpoints.py --located located_checkpoints.yaml

ou, si les dossiers ont gardé leurs noms d'origine :

    python scripts/package_checkpoints.py --src ".../training_logs/Removing_colors"

Lit experiments/thesis_runs.yaml, copie pour chaque run (3 graines + contrôle,
5 configurations) le checkpoint évalué dans le mémoire, en suivant les liens
symboliques (last.ckpt est un lien créé par Lightning), vérifie que chaque fichier
est une archive torch complète, et écrit SHA256SUMS dans l'archive.
"""

import argparse
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_runs(manifest):
    for exp_name, exp in manifest["experiments"].items():
        for run in exp["runs"] + ([exp["control"]] if exp.get("control") else []):
            yield f"{exp_name}/{run['run_id']}", run["original_run_name"], run.get("original_ckpt", run["ckpt"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--located", type=Path, help="Fichier produit par identify_checkpoints.py")
    src.add_argument("--src", type=Path, help="Dossier contenant les dossiers de runs (noms d'origine)")
    parser.add_argument("--out", default="gwt_synesthesia_checkpoints.tar.gz", type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(open(REPO_ROOT / "experiments" / "thesis_runs.yaml"))
    located = yaml.safe_load(open(args.located)) if args.located else {}
    files, missing, broken = [], [], []
    for label, run_name, ckpt in iter_runs(manifest):
        if args.located:
            if label not in located:
                missing.append(f"{label} (non identifié)")
                continue
            path = Path(located[label]).resolve()
        else:
            path = (args.src / run_name / "checkpoints" / ckpt).resolve()
        if not path.is_file():
            missing.append(str(path))
        elif not zipfile.is_zipfile(path):
            broken.append(str(path))
        else:
            files.append((f"runs/{run_name}/checkpoints/{ckpt}", path))

    for label, items in (("INTROUVABLES", missing), ("CORROMPUS (zip invalide)", broken)):
        if items:
            print(f"{label} :\n  " + "\n  ".join(items))
    if missing or broken:
        raise SystemExit("Corrigez les chemins ci-dessus (ou --src) puis relancez.")

    sums = []
    with tarfile.open(args.out, "w:gz") as tar:
        for arcname, path in files:
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {arcname}")
            tar.add(path, arcname=arcname)  # path déjà résolu : copie le vrai fichier
        data = ("\n".join(sums) + "\n").encode()
        info = tarfile.TarInfo("SHA256SUMS")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    size_mb = args.out.stat().st_size / 1e6
    print(f"OK : {len(files)} checkpoints -> {args.out} ({size_mb:.1f} Mo)")
    print(f"sha256 de l'archive : {hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
