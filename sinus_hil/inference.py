"""Sliding-window inference with optional TTA and MC-dropout.

Returns probability maps, label maps and voxel-wise uncertainty maps.
"""
from __future__ import annotations
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from . import config as C
from .model import build_model, enable_mc_dropout
from .dataset import load_case


def load_checkpoint(ckpt: str, dev=None):
    dev = dev or C.device()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = C.TrainCfg(**sd["cfg"]) if isinstance(sd.get("cfg"), dict) else C.TrainCfg()
    net = build_model(cfg)
    net.load_state_dict(sd["model"])
    net.to(dev).eval()
    return net, cfg, sd


def _norm():
    return json.load(open(os.path.join(C.CACHE, "norm.json")))


def _prep(img: np.ndarray) -> torch.Tensor:
    st = _norm()
    x = np.clip(np.asarray(img, dtype=np.float32), st["clip"][0], st["clip"][1])
    x = (x - st["mean"]) / (st["std"] + 1e-8)
    return torch.from_numpy(x)[None, None]


@torch.no_grad()
def predict_probs(net, img: np.ndarray, cfg: C.TrainCfg, dev=None,
                  tta: bool = False, sw_batch: int = 2) -> np.ndarray:
    """Return a (C, H, W, D) float32 probability map.

    With tta=True, averages predictions over flips of each axis (and one
    two-axis flip), which both sharpens boundaries and gives a disagreement
    signal used by the active-learning score.
    """
    dev = dev or C.device()
    st = _norm()
    air = (st["clip"][0] - st["mean"]) / (st["std"] + 1e-8)   # normalised value of out-of-window air
    x = _prep(img).to(dev)

    def fwd(t):
        o = net(t)
        if o.dim() == 6:
            o = o[:, 0]
        return o

    def run(t):
        return sliding_window_inference(
            t, roi_size=cfg.patch, sw_batch_size=sw_batch, predictor=fwd,
            overlap=cfg.sw_overlap, mode="gaussian", padding_mode="constant",
            cval=float(air), progress=False, device=torch.device("cpu"), sw_device=dev,
        )

    logits = run(x)
    p = torch.softmax(logits.float(), 1)
    if tta:
        acc, n = p, 1
        for axes in [(2,), (3,), (4,), (2, 3)]:
            o = run(torch.flip(x, axes))
            acc = acc + torch.softmax(torch.flip(o, axes).float(), 1)
            n += 1
        p = acc / n
    return p[0].numpy().astype(np.float32)


@torch.no_grad()
def predict_mc(net, img: np.ndarray, cfg: C.TrainCfg, n_samples: int = 8, dev=None):
    """MC-dropout sampling.

    Returns (mean probability, summed per-class predictive variance, fraction of
    voxels whose arg-max label flips between samples).
    """
    dev = dev or C.device()
    enable_mc_dropout(net)
    acc = None
    acc2 = None
    labels = []
    for _ in range(n_samples):
        p = predict_probs(net, img, cfg, dev=dev, tta=False)
        acc = p if acc is None else acc + p
        acc2 = p ** 2 if acc2 is None else acc2 + p ** 2
        labels.append(p.argmax(0).astype(np.uint8))
    net.eval()
    mean = acc / n_samples
    var = np.clip(acc2 / n_samples - mean ** 2, 0, None).sum(0)      # summed across classes
    L = np.stack(labels)
    disagree = (L != L[0]).any(0).astype(np.float32)                  # voxels whose label flipped
    return mean, var, disagree


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-7, 1.0)
    return -(p * np.log(p)).sum(0) / np.log(prob.shape[0])            # normalised to [0, 1]


def to_original(lbl: np.ndarray, meta: dict) -> np.ndarray:
    """Resample a label map from cache resolution back to the original CT shape."""
    from scipy import ndimage
    tgt = tuple(meta["orig_shape"])
    if lbl.shape == tgt:
        return lbl
    zoom = [t / s for t, s in zip(tgt, lbl.shape)]
    out = ndimage.zoom(lbl, zoom, order=0, mode="nearest")
    # zoom rounding can be off by one voxel; crop or pad to the exact target
    sl = tuple(slice(0, min(out.shape[i], tgt[i])) for i in range(3))
    fixed = np.zeros(tgt, dtype=lbl.dtype); fixed[sl] = out[sl]
    return fixed


def save_nifti(lbl: np.ndarray, meta: dict, path: str):
    """Write a NIfTI label map.

    meta['orig_*'] records the geometry *after* RAS canonicalisation, which is the
    frame the cached arrays live in, so the affine can be used directly.
    """
    import nibabel as nib
    aff = np.array(meta["orig_affine"], dtype=np.float64)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(lbl.astype(np.uint8)), aff), path)
