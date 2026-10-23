"""Pair DATA/ with SEG/ into (image, mask) cases, resolving repeat annotations.

The folders named 0/1/3/4/5 are in fact human-in-the-loop annotation rounds:
  0(9)_all manual       -> round 0, fully hand-drawn
  N(k)_manual corrected -> round N, model prediction then corrected by a human
The same patient can appear in several rounds. If it is the *same scan*
(identical shape + spacing) we keep only the latest round's annotation; if it is
a different scan it counts as a separate training case.
"""
from __future__ import annotations
import os, re, json, glob, collections
import nibabel as nib
from . import config as C

ROUND_RE = re.compile(r"^(\d+)\(")


def patient_id(path: str) -> str:
    b = os.path.basename(path).replace(".nii.gz", "")
    b = re.sub(r"_0000$", "", b)        # nnU-Net style channel suffix
    b = re.sub(r"\(\d+\)$", "", b)      # trailing sequence number in the filename
    return b


def round_of(path: str) -> int:
    folder = os.path.basename(os.path.dirname(path))
    m = ROUND_RE.match(folder)
    return int(m.group(1)) if m else -1


def _scan(root: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.nii.gz"), recursive=True)):
        h = nib.load(p)
        out.append(dict(
            path=p, pid=patient_id(p), rnd=round_of(p),
            folder=os.path.basename(os.path.dirname(p)),
            shape=tuple(int(x) for x in h.shape),
            zoom=tuple(round(float(z), 4) for z in h.header.get_zooms()[:3]),
        ))
    return out


def build(verbose: bool = True) -> dict:
    imgs, segs = _scan(C.DATA_DIR), _scan(C.SEG_DIR)

    # ---- 1. Find an image for each mask. Require identical shape + spacing;
    #         the image may live in a different round folder. ----
    cand = []
    for s in segs:
        hits = [i for i in imgs if i["pid"] == s["pid"]
                and i["shape"] == s["shape"] and i["zoom"] == s["zoom"]]
        if hits:
            # prefer an image from the same round
            hits.sort(key=lambda i: (i["rnd"] != s["rnd"], i["rnd"]))
            cand.append(dict(pid=s["pid"], seg=s["path"], img=hits[0]["path"],
                             seg_round=s["rnd"], img_round=hits[0]["rnd"],
                             shape=list(s["shape"]), zoom=list(s["zoom"]),
                             seg_folder=s["folder"]))

    # ---- 2. Same scan (pid+shape+zoom) annotated in several rounds ->
    #         keep the latest round only, record the rest as superseded. ----
    by_scan: dict[tuple, dict] = {}
    superseded = []
    for c in cand:
        key = (c["pid"], tuple(c["shape"]), tuple(c["zoom"]))
        prev = by_scan.get(key)
        if prev is None or c["seg_round"] > prev["seg_round"]:
            if prev is not None:
                superseded.append(prev)
            by_scan[key] = c
        else:
            superseded.append(c)

    pairs = sorted(by_scan.values(), key=lambda c: (c["pid"], c["shape"][2]))
    # give each scan of a repeat patient its own case_id
    seen = collections.Counter()
    for c in pairs:
        seen[c["pid"]] += 1
        c["case_id"] = c["pid"] if seen[c["pid"]] == 1 else f"{c['pid']}-s{seen[c['pid']]}"

    # ---- 3. Masks with no matching image: recoverable data, see scripts/missing_images.py ----
    paired_segs = {c["seg"] for c in cand}
    orphan = [dict(seg=s["path"], pid=s["pid"], rnd=s["rnd"],
                   shape=list(s["shape"]), zoom=list(s["zoom"]))
              for s in segs if s["path"] not in paired_segs]

    mf = dict(pairs=pairs, superseded=superseded, orphan_masks=orphan,
              n_images=len(imgs), n_masks=len(segs), labels=C.LABELS)
    out = os.path.join(C.WORK, "manifest.json")
    with open(out, "w") as f:
        json.dump(mf, f, indent=1, ensure_ascii=False)

    if verbose:
        print(f"原始: {len(imgs)} 图像 / {len(segs)} 掩膜")
        print(f"可用训练样本: {len(pairs)} 个独立扫描 (来自 {len({c['pid'] for c in pairs})} 位病人)")
        print(f"被更新轮次取代的旧标注: {len(superseded)}")
        print(f"缺少配对图像的掩膜: {len(orphan)}  <-- 找回这些图像可使数据量翻倍")
        print(f"写入 {out}")
    return mf


def load() -> dict:
    with open(os.path.join(C.WORK, "manifest.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    build()
