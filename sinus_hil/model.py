"""3D segmentation network with MC-dropout support.

Uses MONAI's DynUNet (the nnU-Net topology): deep supervision plus kernel/stride
schedules derived from the patch size and spacing. On a dataset this small it is
more reliable than a hand-rolled U-Net. The dropout layers can be switched back
on at inference time on their own, which is what gives the MC-dropout estimate.
"""
from __future__ import annotations
import torch, torch.nn as nn
from monai.networks.nets import DynUNet
from . import config as C


def _kernels_strides(patch, spacing):
    sizes, spac = list(patch), list(spacing)
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spac) for sp in spac]
        stride = [2 if r <= 2 and s >= 8 else 1 for r, s in zip(spacing_ratio, sizes)]
        kernel = [3 if r <= 2 else 1 for r in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spac = [i * j for i, j in zip(spac, stride)]
        kernels.append(kernel); strides.append(stride)
        if len(strides) >= 5:
            break
    strides.insert(0, len(patch) * [1])
    kernels.append(len(patch) * [3])
    return kernels, strides


def build_model(cfg: C.TrainCfg, spacing=None) -> nn.Module:
    spacing = tuple(spacing or C.PreprocCfg().spacing)
    kernels, strides = _kernels_strides(cfg.patch, spacing)
    net = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=C.NUM_CLASSES,
        kernel_size=kernels, strides=strides, upsample_kernel_size=strides[1:],
        filters=list(cfg.filters)[: len(strides)],
        norm_name=("INSTANCE", {"affine": True}),
        dropout=cfg.dropout,
        deep_supervision=cfg.deep_supervision,
        deep_supr_num=min(2, len(strides) - 2),
        res_block=True,
    )
    return net


def enable_mc_dropout(net: nn.Module):
    """Put only the dropout layers back in train mode.

    Normalisation layers stay in eval mode, which is what distinguishes proper
    MC-dropout from simply calling net.train().
    """
    net.eval()
    n = 0
    for m in net.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout3d, nn.AlphaDropout)):
            m.train(); n += 1
    return n


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
