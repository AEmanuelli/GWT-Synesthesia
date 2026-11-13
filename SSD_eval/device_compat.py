"""Permet de charger sur CPU des checkpoints sauvegardés sur GPU.

Les checkpoints du stage ont été écrits sur `cuda:0`. Sur une machine sans GPU,
torch.load échoue si aucun `map_location` n'est donné, ce qui arrive dans certaines
bibliothèques (migration de checkpoints de shimmer). Importer ce module fait que,
lorsque CUDA est indisponible, tout appel à torch.load sans `map_location` charge sur CPU.
"""

import torch

if not torch.cuda.is_available() and not getattr(torch.load, "_gwt_cpu_patch", False):
    _original_torch_load = torch.load

    def _torch_load_on_cpu(*args, **kwargs):
        if kwargs.get("map_location") is None:
            kwargs["map_location"] = "cpu"
        return _original_torch_load(*args, **kwargs)

    _torch_load_on_cpu._gwt_cpu_patch = True
    torch.load = _torch_load_on_cpu
