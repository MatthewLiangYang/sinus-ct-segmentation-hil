"""Preprocessing: canonical orientation -> isotropic resampling -> intensity
clipping -> .npy cache.

Each cached case also stores the geometry needed to invert the transform, so a
prediction can be written back in the original CT's shape and affine.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from . import config as C
from .manifest import load as load_manifest

PC = C.PreprocCfg()


def _reorient(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient to RAS so scanner-dependent axis conventions drop out.

    The voxel axes in this dataset are LPS; NIfTI affines always map to RAS+,
    which is an easy sign error to make when reading coordinates (it is how the
    sphenoid/frontal labels were initially mixed up -- see README).
    """
    return nib.as_closest_canonical(img)


def _resample(arr: np.ndarray, src_sp, dst_sp, order: int) -> np.ndarray:
    zoom = [s / d for s, d in zip(src_sp, dst_sp)]
    if np.allclose(zoom, 1.0, atol=1e-3):
        return arr
    return ndimage.zoom(arr, zoom, order=order, mode="nearest", prefilter=order > 1)


def process_one(case: dict, force: bool = False) -> dict:
    cid = case["case_id"]
    f_img = os.path.join(C.CACHE, f"{cid}_img.npy")
    f_seg = os.path.join(C.CACHE, f"{cid}_seg.npy")
    f_meta = os.path.join(C.CACHE, f"{cid}_meta.json")
    if not force and all(os.path.exists(p) for p in (f_img, f_seg, f_meta)):
        return json.load(open(f_meta))

    im = _reorient(nib.load(case["img"]))
    sg = _reorient(nib.load(case["seg"]))
    src_sp = tuple(float(z) for z in im.header.get_zooms()[:3])

    a = np.asanyarray(im.dataobj).astype(np.float32)
    s = np.asanyarray(sg.dataobj).astype(np.uint8)
    assert a.shape == s.shape, f"{cid}: image/mask shape mismatch {a.shape} vs {s.shape}"

    a = np.clip(a, *PC.clip)
    a = _resample(a, src_sp, PC.spacing, order=1)
    s = _resample(s, src_sp, PC.spacing, order=0)
    s[s >= C.NUM_CLASSES] = 0

    np.save(f_img, a.astype(np.int16))
    np.save(f_seg, s.astype(np.uint8))

    fg = a[s > 0]
    meta = dict(
        case_id=cid, pid=case["pid"], seg_round=case["seg_round"],
        seg_folder=case["seg_folder"], img=case["img"], seg=case["seg"],
        orig_shape=list(im.shape), orig_zoom=list(src_sp),
        orig_affine=np.asarray(im.affine).tolist(),
        shape=list(a.shape), spacing=list(PC.spacing),
        fg_mean=float(fg.mean()) if fg.size else 0.0,
        fg_std=float(fg.std()) if fg.size else 1.0,
        fg_voxels={int(k): int(v) for k, v in zip(*np.unique(s[s > 0], return_counts=True))},
    )
    json.dump(meta, open(f_meta, "w"))
    return meta


def run(force: bool = False, jobs: int = 4) -> dict:
    mf = load_manifest()
    cases = mf["pairs"]
    metas = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for i, m in enumerate(ex.map(process_one, cases, [force] * len(cases))):
                metas.append(m); print(f"[{i+1}/{len(cases)}] {m['case_id']} -> {m['shape']}", flush=True)
    else:
        for i, c in enumerate(cases):
            m = process_one(c, force); metas.append(m)
            print(f"[{i+1}/{len(cases)}] {m['case_id']} -> {m['shape']}", flush=True)

    # Dataset-wide foreground intensity statistics -> normalisation constants
    # (nnU-Net style global z-score for CT).
    w = np.array([sum(m["fg_voxels"].values()) for m in metas], dtype=np.float64)
    mu = float(np.average([m["fg_mean"] for m in metas], weights=w))
    var = float(np.average([m["fg_std"] ** 2 for m in metas], weights=w))
    stats = dict(mean=mu, std=float(np.sqrt(var)), clip=list(PC.clip),
                 spacing=list(PC.spacing), n_cases=len(metas))
    json.dump(stats, open(os.path.join(C.CACHE, "norm.json"), "w"), indent=1)
    print(f"\n归一化: mean={mu:.1f} std={np.sqrt(var):.1f} (前景 HU 加权)")
    return stats


def normalize(a: np.ndarray) -> np.ndarray:
    st = json.load(open(os.path.join(C.CACHE, "norm.json")))
    x = np.clip(a.astype(np.float32), st["clip"][0], st["clip"][1])
    return (x - st["mean"]) / (st["std"] + 1e-8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    run(**vars(ap.parse_args()))
