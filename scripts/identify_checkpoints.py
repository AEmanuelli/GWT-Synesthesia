"""Retrouve les checkpoints du mémoire sur un disque, quels que soient leurs noms/emplacements.

Chaque checkpoint Lightning contient une "empreinte" lisible sans PyTorch :
  - le dossier où il a été écrit à l'origine (callback ModelCheckpoint.dirpath),
  - epoch et global_step,
  - les coefficients de loss (hyper_parameters.loss_coefs),
  - la présence de la couleur dans les attributs (dimension d'entrée de l'encodeur attr : 8 ou 11),
  - le planning de LR (total_steps), et la meilleure val/loss (best_model_score).
Ces empreintes sont comparées à celles attendues dans experiments/thesis_runs.yaml
(champ `expect`, relevé sur les checkpoints d'origine).

    # 1. Scanner un ou plusieurs dossiers (récursif)
    python scripts/identify_checkpoints.py ~/Desktop /mnt/HD2 /media/disque_externe

    # 2. (optionnel) Croiser avec W&B : graine, alpha, couleur, meilleure val/loss
    pip install wandb && wandb login
    python scripts/identify_checkpoints.py ~/Desktop --wandb lexman-psl/Removing_colors

Sorties : un rapport à l'écran, `checkpoints_scan.csv` (tous les .ckpt trouvés) et
`located_checkpoints.yaml` (run du mémoire -> chemin), utilisable par
`python scripts/package_checkpoints.py --located located_checkpoints.yaml`.
"""

import argparse
import collections
import csv
import hashlib
import io
import os
import pickle
import struct
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# Lecture des métadonnées d'un .ckpt sans PyTorch
# ----------------------------------------------------------------------------
class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _CkptUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.startswith("torch._utils") and name.startswith("_rebuild"):
            return lambda *a, **k: {"tensor": True, "storage": a[0] if a else None,
                                    "size": tuple(a[2]) if len(a) > 2 else None}
        return type(name, (_Stub,), {"_qualname": f"{module}.{name}"})

    def persistent_load(self, pid):
        # pid = ('storage', StorageType, key, location, numel)
        return {"dtype": getattr(pid[1], "_qualname", str(pid[1])), "key": pid[2]}


def read_ckpt_metadata(path: Path) -> dict:
    info = {"path": str(path), "size_mb": round(path.stat().st_size / 1e6, 2)}
    raw = path.read_bytes()
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    info["valid_zip"] = zipfile.is_zipfile(io.BytesIO(raw))
    zf = None
    if info["valid_zip"]:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        pkl = zf.read(pkl_name)
    else:  # fichier tronqué : data.pkl est en tête d'archive, souvent lisible quand même
        start = raw.find(b"\x80\x02")
        pkl = raw[start: raw.find(b"PK\x07\x08", start)]
    try:
        obj = _CkptUnpickler(io.BytesIO(pkl)).load()
    except Exception as e:  # pas un checkpoint Lightning, ou trop abîmé
        info["error"] = f"{type(e).__name__}: {e}"
        return info

    info["epoch"] = obj.get("epoch")
    info["global_step"] = obj.get("global_step")
    hp = obj.get("hyper_parameters") or {}
    info["loss_coefs"] = {k: float(v) for k, v in (hp.get("loss_coefs") or {}).items()}
    w = (obj.get("state_dict") or {}).get("gw_mod.gw_encoders.attr.0.weight")
    if isinstance(w, dict) and w.get("size"):
        info["attr_input_dim"] = w["size"][1]
        info["colors_in_attr"] = w["size"][1] != 8
    sched = (obj.get("lr_schedulers") or [{}])[0]
    info["total_steps"] = sched.get("total_steps")
    info["base_lr"] = (sched.get("base_lrs") or [None])[0]
    for key, cb in (obj.get("callbacks") or {}).items():
        if key.startswith("ModelCheckpoint"):
            info["original_dir"] = Path(cb.get("dirpath") or "").parent.name
            info["original_best_path"] = cb.get("best_model_path")
            score = cb.get("best_model_score")
            if zf is not None and isinstance(score, dict) and score.get("storage"):
                st = score["storage"]
                fmt = "<d" if "Double" in st["dtype"] else "<f"
                try:
                    data = zf.read(f"{prefix}data/{st['key']}")
                    info["best_val_loss"] = struct.unpack(fmt, data[: struct.calcsize(fmt)])[0]
                except (KeyError, struct.error):
                    pass
    return info


# ----------------------------------------------------------------------------
# Comparaison aux empreintes attendues
# ----------------------------------------------------------------------------
def fingerprint_matches(info: dict, expect: dict) -> bool:
    if info.get("epoch") != expect["epoch"] or info.get("global_step") != expect["global_step"]:
        return False
    if info.get("colors_in_attr") != expect["colors_in_attr"]:
        return False
    coefs = info.get("loss_coefs") or {}
    return all(abs(coefs.get(k, -1) - float(v)) < 1e-9 for k, v in expect["loss_coefs"].items())


def manifest_runs(manifest):
    for exp_name, exp in manifest["experiments"].items():
        for run in exp["runs"] + ([exp["control"]] if exp.get("control") else []):
            yield exp_name, exp, run


def wandb_check(api, wandb_path, run, exp, info):
    """Compare le run W&B portant le nom d'origine avec l'attendu et le checkpoint trouvé."""
    runs = list(api.runs(wandb_path, filters={"display_name": run["original_run_name"]}))
    if not runs:
        return "W&B: aucun run de ce nom"
    msgs = []
    for r in runs:
        cfg = r.config
        exp_colors = run["expect"]["colors_in_attr"]
        problems = []
        if "seed" in cfg and int(cfg["seed"]) != int(run.get("original_seed", run["seed"])):
            problems.append(f"seed={cfg['seed']}")
        if "alpha" in cfg and float(cfg["alpha"]) != float(exp.get("attr_alpha", 1)):
            problems.append(f"alpha={cfg['alpha']}")
        if "exclude_colors" in cfg and bool(cfg["exclude_colors"]) == exp_colors:
            problems.append(f"exclude_colors={cfg['exclude_colors']}")
        if info and info.get("best_val_loss") is not None:
            losses = [row["val/loss"] for row in r.scan_history(keys=["val/loss"]) if row.get("val/loss") is not None]
            if losses and abs(min(losses) - info["best_val_loss"]) > 1e-4:
                problems.append(f"min val/loss W&B={min(losses):.5f} != ckpt {info['best_val_loss']:.5f}")
        msgs.append(f"W&B {r.id} ({r.state}, seed={cfg.get('seed')}, alpha={cfg.get('alpha')}): "
                    + ("OK" if not problems else "ÉCART " + ", ".join(problems)))
    return " | ".join(msgs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="Dossiers à scanner (récursivement)")
    parser.add_argument("--wandb", metavar="ENTITY/PROJECT", help="Croiser avec les runs W&B de ce projet")
    parser.add_argument("--csv", default="checkpoints_scan.csv", type=Path)
    parser.add_argument("--located", default="located_checkpoints.yaml", type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(open(REPO_ROOT / "experiments" / "thesis_runs.yaml"))

    scanned = []
    for root in args.roots:
        for dirpath, _, filenames in os.walk(root, followlinks=False):
            for fn in filenames:
                p = Path(dirpath) / fn
                if fn.endswith(".ckpt") and not p.is_symlink():
                    try:
                        scanned.append(read_ckpt_metadata(p))
                    except OSError as e:
                        print(f"Illisible : {p} ({e})")
    print(f"{len(scanned)} fichiers .ckpt scannés.\n")

    cols = ["path", "valid_zip", "original_dir", "epoch", "global_step", "loss_coefs", "colors_in_attr",
            "total_steps", "best_val_loss", "size_mb", "sha256", "error"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scanned)

    api = None
    if args.wandb:
        import wandb
        api = wandb.Api()

    located, n_ok = {}, 0
    n_total = sum(1 for _ in manifest_runs(manifest))
    for exp_name, exp, run in manifest_runs(manifest):
        label = f"{exp_name}/{run['run_id']}"
        expect = run["expect"]
        matches = [i for i in scanned if fingerprint_matches(i, expect)]
        good = [i for i in matches if i["valid_zip"]]
        named = [i for i in good if i.get("original_dir") == run["original_run_name"]]
        chosen = named or good
        distinct = {i["sha256"] for i in chosen}
        if chosen and len(distinct) == 1:
            status = "OK" if named else "OK (empreinte seule, dossier d'origine différent)"
            located[label] = chosen[0]["path"]
            n_ok += 1
        elif len(distinct) > 1:
            status = f"AMBIGU ({len(distinct)} fichiers différents)"
        elif matches:
            status = "TROUVÉ MAIS CORROMPU (zip incomplet)"
        else:
            status = "ABSENT"
        print(f"[{status}] {label}  <- \"{run['original_run_name']}\" epoch {expect['epoch']}")
        for i in chosen or matches:
            print(f"      {i['path']}  (copies identiques : "
                  f"{sum(1 for j in scanned if j['sha256'] == i['sha256'])})")
        if api is not None:
            print("      " + wandb_check(api, args.wandb, run, exp, chosen[0] if chosen else None))

    with open(args.located, "w") as f:
        yaml.safe_dump(located, f, allow_unicode=True, sort_keys=True)
    print(f"\n{n_ok}/{n_total} runs identifiés sans ambiguïté -> {args.located}")
    print(f"Inventaire complet -> {args.csv}")


if __name__ == "__main__":
    main()
