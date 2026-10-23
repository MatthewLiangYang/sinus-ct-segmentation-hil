"""Combined Dice + cross-entropy loss, handling DynUNet's deep-supervision outputs."""
from __future__ import annotations
import torch, torch.nn as nn
from monai.losses import DiceCELoss
from . import config as C


class DeepSupLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.loss = DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True,
            squared_pred=True, batch=True, smooth_nr=0.0, smooth_dr=1e-5,
            weight=class_weights,
        )

    def forward(self, pred, y):
        # DynUNet deep-supervision output has shape (B, n_out, C, H, W, D);
        # weight the heads 1, 1/2, 1/4, ... and renormalise.
        if pred.dim() == 6:
            n = pred.shape[1]
            w = torch.tensor([1.0 / (2 ** i) for i in range(n)], device=pred.device)
            w = w / w.sum()
            return sum(w[i] * self.loss(pred[:, i], y.unsqueeze(1)) for i in range(n))
        return self.loss(pred, y.unsqueeze(1))


@torch.no_grad()
def dice_per_class(pred_lbl: torch.Tensor, y: torch.Tensor) -> list[float]:
    """Per-foreground-class Dice between two integer label maps.

    Returns nan for a class that is absent from both, so it can be skipped with
    nanmean rather than dragging the average to zero.
    """
    out = []
    for L in C.FG:
        p, g = (pred_lbl == L), (y == L)
        ps, gs = p.sum().item(), g.sum().item()
        if gs == 0 and ps == 0:
            out.append(float("nan"))
        else:
            out.append(2.0 * (p & g).sum().item() / max(ps + gs, 1))
    return out
