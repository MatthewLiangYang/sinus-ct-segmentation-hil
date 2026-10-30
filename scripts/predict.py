#!/usr/bin/env python
"""End-to-end inference on new CT volumes.

Produces, per case: a segmentation mask as .nii.gz (open it directly in 3D
Slicer or ITK-SNAP), quantitative sinus metrics, an anatomical plausibility
check, and a PNG sheet of the slices most worth human review.

Usage:
  python scripts/predict.py --ckpt work/runs/d49_fold0/best.pt --input /path/to/ct.nii.gz
  python scripts/predict.py --ckpt ... --input /path/to/dir --outdir work/pred
"""
from __future__ import annotations
import os, sys, glob, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sinus_hil import config as C
from sinus_hil.inference import load_checkpoint, predict_probs, save_nifti, entropy
from sinus_hil.active_learning import _cache_unlabeled
from sinus_hil.uncertainty import anatomy_report, top_slices
from sinus_hil.disease import sinus_metrics, finalize
from sinus_hil.report import overlay_sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="单个 .nii.gz 或目录")
    ap.add_argument("--outdir", default=C.PRED)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()

    files = [a.input] if a.input.endswith(".nii.gz") else \
        sorted(glob.glob(os.path.join(a.input, "**", "*.nii.gz"), recursive=True))
    net, cfg, sd = load_checkpoint(a.ckpt)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"checkpoint: {a.ckpt}  (fold={sd.get('fold')}, best valDice={sd.get('best')})")

    rows = []
    for i, f in enumerate(files):
        img, meta = _cache_unlabeled(f)
        prob = predict_probs(net, img, cfg, tta=not a.no_tta)
        lbl = prob.argmax(0).astype(np.uint8)
        ent = entropy(prob)
        cid = meta["case_id"]
        save_nifti(lbl, meta, os.path.join(a.outdir, f"{cid}_seg.nii.gz"))
        m = finalize(sinus_metrics(img, lbl))
        m["case_id"] = cid
        m["anatomy_check"] = anatomy_report(lbl)
        m["mean_entropy"] = float(ent[lbl > 0].mean()) if (lbl > 0).any() else None
        rows.append(m)
        if not a.no_png:
            overlay_sheet(img, lbl, os.path.join(a.outdir, f"{cid}.png"), unc=ent,
                          slices=top_slices(ent * (lbl > 0), k=8),
                          title=f"{cid}  受累总分={m['involvement']['total']}/16")
        print(f"[{i+1}/{len(files)}] {cid}  受累总分={m['involvement']['total']}/16  "
              f"解剖异常={m['anatomy_check']['total_penalty']:.3f}")
        for fl in m["flags"]:
            print(f"      ! {fl}")
    json.dump(rows, open(os.path.join(a.outdir, "predictions.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n输出目录: {a.outdir}")


if __name__ == "__main__":
    main()
