#!/usr/bin/env python
"""List masks in SEG/ that have no matching CT in DATA/.

There are 25 of them. They are free training data: recover the original CT from
PACS, drop it into the matching DATA/ round folder, rerun manifest and
preprocess, and the training set grows from 25 to ~50 cases. This is the single
highest-leverage action available on this project and needs no code changes.

Writes work/missing_images.csv with patient ID, expected shape and spacing, so
the right series can be identified in PACS.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sinus_hil import config as C
from sinus_hil.manifest import build

mf = build(verbose=False)
rows = sorted(mf["orphan_masks"], key=lambda r: (r["rnd"], r["pid"]))
out = os.path.join(C.WORK, "missing_images.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patient_id", "round", "mask_path", "expected_shape", "expected_spacing_mm",
                "期望放置目录"])
    for r in rows:
        folder = os.path.basename(os.path.dirname(r["seg"]))
        w.writerow([r["pid"], r["rnd"], r["seg"],
                    "x".join(map(str, r["shape"])),
                    ",".join(f"{z:.3f}" for z in r["zoom"]),
                    os.path.join("DATA", folder)])
print(f"{len(rows)} 个掩膜缺少配对 CT，清单写入 {out}\n")
print(f"{'病人ID':<16}{'轮次':<6}{'期望 shape':<20}{'期望层内/层厚 mm'}")
for r in rows:
    print(f"{r['pid']:<16}{r['rnd']:<6}{'x'.join(map(str,r['shape'])):<20}"
          f"{r['zoom'][0]:.3f} / {r['zoom'][2]:.3f}")
print("\n找回方式：按 病人ID + 层数 + 层厚 去 PACS 调原始序列，")
print("导出为 .nii.gz 放进上表的目录，文件名与掩膜同名（或加 _0000 后缀）即可。")
