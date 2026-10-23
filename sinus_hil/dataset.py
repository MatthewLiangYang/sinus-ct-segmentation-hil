"""Patch-level dataset: foreground oversampling plus light 3D augmentation.

With only 25 training volumes the sampling strategy matters more than the network
architecture. Following nnU-Net, ~2/3 of patch centres are forced onto a randomly
chosen foreground class; otherwise small structures such as the frontal and
sphenoid sinuses -- about 0.01% of all voxels -- are almost never seen.
"""
from __future__ import annotations
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from . import config as C


def _norm_stats():
    return json.load(open(os.path.join(C.CACHE, "norm.json")))


def load_case(cid: str, mmap: bool = True):
    mm = "r" if mmap else None
    img = np.load(os.path.join(C.CACHE, f"{cid}_img.npy"), mmap_mode=mm)
    seg = np.load(os.path.join(C.CACHE, f"{cid}_seg.npy"), mmap_mode=mm)
    meta = json.load(open(os.path.join(C.CACHE, f"{cid}_meta.json")))
    return img, seg, meta


def fg_index(cid: str) -> dict[int, np.ndarray]:
    """Cached, subsampled index of foreground voxel coordinates.

    Lets the sampler draw a foreground patch centre in O(1) instead of scanning
    the volume on every call.
    """
    f = os.path.join(C.CACHE, f"{cid}_fgidx.npz")
    if os.path.exists(f):
        z = np.load(f)
        return {int(k.split("_")[1]): z[k] for k in z.files}
    _, seg, _ = load_case(cid)
    seg = np.asarray(seg)
    out = {}
    rng = np.random.default_rng(0)
    for L in C.FG:
        idx = np.argwhere(seg == L)
        if len(idx) == 0:
            continue
        if len(idx) > 20000:                       # subsample to keep memory small
            idx = idx[rng.choice(len(idx), 20000, replace=False)]
        out[L] = idx.astype(np.int16)
    np.savez_compressed(f, **{f"L_{k}": v for k, v in out.items()})
    return out


class SinusPatches(Dataset):
    def __init__(self, case_ids: list[str], cfg: C.TrainCfg, train: bool = True,
                 length: int | None = None):
        self.ids = case_ids
        self.cfg = cfg
        self.train = train
        self.len = length if length is not None else cfg.iters_per_epoch * cfg.batch_size
        st = _norm_stats()
        self.mean, self.std, self.clip = st["mean"], st["std"], st["clip"]
        self.cache = {cid: load_case(cid) for cid in case_ids}
        self.fg = {cid: fg_index(cid) for cid in case_ids}
        self.rng = np.random.default_rng(cfg.seed + (0 if train else 1))

    def __len__(self):
        return self.len

    # ---------- sampling ----------
    def _sample_center(self, cid, shape):
        p = self.cfg.patch
        if self.train and self.rng.random() < self.cfg.fg_prob and self.fg[cid]:
            Ls = list(self.fg[cid].keys())
            L = Ls[self.rng.integers(len(Ls))]
            idx = self.fg[cid]
            c = idx[L][self.rng.integers(len(idx[L]))].astype(np.int64)
        else:
            c = np.array([self.rng.integers(s) for s in shape])
        lo = [int(np.clip(c[i] - p[i] // 2, 0, max(0, shape[i] - p[i]))) for i in range(3)]
        return lo

    def _crop(self, arr, lo, pad_val):
        p = self.cfg.patch
        sl = tuple(slice(lo[i], min(lo[i] + p[i], arr.shape[i])) for i in range(3))
        out = np.asarray(arr[sl])
        if out.shape != tuple(p):                  # pad when the volume is smaller than a patch
            pad = [(0, p[i] - out.shape[i]) for i in range(3)]
            out = np.pad(out, pad, mode="constant", constant_values=pad_val)
        return out

    # ---------- augmentation ----------
    def _augment(self, x, y):
        r = self.rng
        # Flips are safe here: labels are not side-specific (both maxillary
        # sinuses share label 3), so mirroring does not corrupt the target.
        for ax in range(3):
            if r.random() < 0.5:
                x, y = np.flip(x, ax), np.flip(y, ax)
        if r.random() < 0.2:                       # 90-degree in-plane rotation
            k = int(r.integers(1, 4))
            x, y = np.rot90(x, k, (0, 1)), np.rot90(y, k, (0, 1))
        x = np.ascontiguousarray(x).astype(np.float32)
        y = np.ascontiguousarray(y)
        if r.random() < 0.3:                       # Gaussian noise
            x = x + r.normal(0, r.uniform(0.01, 0.08), x.shape).astype(np.float32)
        if r.random() < 0.3:                       # brightness / contrast (varying scan protocols)
            x = x * r.uniform(0.85, 1.15) + r.uniform(-0.1, 0.1)
        if r.random() < 0.2:                       # gamma
            mn, mx = x.min(), x.max()
            if mx > mn:
                x = ((x - mn) / (mx - mn)) ** r.uniform(0.7, 1.4) * (mx - mn) + mn
        return x, y

    def __getitem__(self, i):
        cid = self.ids[int(self.rng.integers(len(self.ids)))] if self.train \
            else self.ids[i % len(self.ids)]
        img, seg, meta = self.cache[cid]
        lo = self._sample_center(cid, img.shape)
        x = self._crop(img, lo, self.clip[0]).astype(np.float32)
        y = self._crop(seg, lo, 0).astype(np.int64)
        x = (np.clip(x, *self.clip) - self.mean) / (self.std + 1e-8)
        if self.train:
            x, y = self._augment(x, y)
        return (torch.from_numpy(np.ascontiguousarray(x))[None].float(),
                torch.from_numpy(np.ascontiguousarray(y)).long())


def splits(case_ids: list[str], pids: list[str], n_folds: int = 5, seed: int = 42):
    """Patient-grouped split.

    Grouping by patient is required, not cosmetic: one patient in this cohort has
    two separate scans, and letting them straddle train/val would leak.
    """
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    idx = np.arange(len(case_ids))
    out = []
    for tr, va in gkf.split(idx, groups=pids):
        out.append(dict(train=[case_ids[i] for i in tr], val=[case_ids[i] for i in va]))
    return out
