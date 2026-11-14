"""Catégories de forme en probabilités à la sortie du décodeur d'attributs du GW.

Les modèles du mémoire ont été entraînés avec une version locale de `shimmer`
(runs « corrected v5 (Logsoftmax --> exp --> log) ») dans laquelle le décodeur
GW -> attributs renvoie, pour les 3 catégories de forme, des probabilités
(log-softmax puis exponentielle) et non des logits bruts ; la loss de classification
reprend ensuite le logarithme de ces probabilités. Cette modification n'a jamais été
poussée sur le fork public de `shimmer` : avec le fork seul, tout passage par les
attributs décodés (cycle complet image -> GW -> attributs -> GW -> image) réinjecte
des logits dans un encodeur qui n'a vu que des probabilités, et la reconstruction
est inutilisable (erreur 100 à 1000 fois plus grande, voir README §5).

Ce module rétablit ce comportement sans toucher aux poids ni aux noms des paramètres :
les checkpoints publiés se chargent tels quels.
"""

import torch
import torch.nn.functional as F
from shimmer import LossOutput
from shimmer.modules.gw_module import GWDecoder_legacy
from shimmer_ssd.modules.domains.attribute import AttributeLegacyDomainModule

N_SHAPES = 3


class GWDecoderLegacyProbs(GWDecoder_legacy):
    """`GWDecoder_legacy` dont les 3 premières sorties sont des probabilités de forme."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        probs = F.log_softmax(out[:, :N_SHAPES], dim=1).exp()
        return torch.cat([probs, out[:, N_SHAPES:]], dim=1)


class AttributeLegacyProbsDomainModule(AttributeLegacyDomainModule):
    """Module d'attributs dont la loss reçoit des probabilités de forme (et non des logits)."""

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, raw_target) -> LossOutput:
        pred_cat, pred_attr = self.decode(pred)
        target_cat, target_attr = self.decode(target)
        loss_attr = F.mse_loss(pred_attr, target_attr, reduction="mean")
        loss_cat = F.cross_entropy(
            torch.log(pred_cat) / self.temperature, torch.argmax(target_cat, 1)
        )
        loss = loss_cat + self.alpha * loss_attr
        return LossOutput(loss, metrics={"loss_attr": loss_attr, "loss_cat": loss_cat})


def use_category_probabilities(domain_modules, gw_decoders) -> None:
    """Bascule le décodeur et le module d'attributs vers leur version « probabilités ».

    À appeler juste après `load_pretrained_domains`, avant de construire ou de charger
    le Global Workspace. Les objets sont modifiés en place (mêmes paramètres, mêmes noms).
    """
    decoder = gw_decoders["attr"]
    if not isinstance(decoder, GWDecoder_legacy):
        raise TypeError(f"Décodeur d'attributs inattendu : {type(decoder).__name__}")
    decoder.__class__ = GWDecoderLegacyProbs

    module = domain_modules["attr"]
    if not isinstance(module, AttributeLegacyDomainModule):
        raise TypeError(f"Module d'attributs inattendu : {type(module).__name__}")
    module.__class__ = AttributeLegacyProbsDomainModule
