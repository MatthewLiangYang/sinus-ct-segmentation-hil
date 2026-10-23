"""Annotation review: find the already-labelled cases and slices most likely to
be wrong.

This is the other half of human-in-the-loop. As well as "which case should be
labelled next", there is "which of the cases we already labelled should be
relabelled".

The criterion is deliberately not "low Dice". It is *high model confidence
combined with disagreement against the human annotation*:
  - where the model is vague, the model is usually the one at fault;
  - where the model is confident and still disagrees, that is where annotation
    noise concentrates.

A known example in this dataset: the sphenoid sinus of one case was annotated at
2.30 cm3 in round 0 and 0.46 cm3 in round 4 -- an 80% change on the same scan,
inter-round Dice 0.321. round_drift() below surfaces exactly this class of
problem without needing a model at all.
"""
from __future__ import annotations
import os, json, glob
import numpy as np
from scipy import ndimage
from . import config as C

VOX_CM3 = float(np.prod(C.PreprocCfg().spacing)) / 1000.0   # volume of one voxel, cm^3
from .dataset import load_case
from .inference import load_checkpoint, predict_probs, entropy
from .uncertainty import anatomy_report, top_slices
from .report import overlay_sheet, worklist_html


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def review_case(net, cfg, cid: str, conf_thr: float = 0.90) -> dict:
    img, seg, meta = load_case(cid)
    img, seg = np.asarray(img), np.asarray(seg)
    prob = predict_probs(net, img, cfg, tta=True)
    pred = prob.argmax(0).astype(np.uint8)
    conf = prob.max(0)
    ent = entropy(prob)

    dsc = {C.LABELS[L]: dice(pred == L, seg == L) for L in C.FG}
    mism = pred != seg
    # confident AND disagreeing = suspected annotation error
    suspect = mism & (conf > conf_thr)
    susp_frac = float(suspect.sum() / max(mism.sum(), 1))

    per_class = {}
    for L in C.FG:
        m_fn = (seg == L) & (pred != L)          # human labelled it, model did not
        m_fp = (pred == L) & (seg != L)          # model claims it, human did not
        per_class[C.LABELS[L]] = dict(
            dice=None if np.isnan(dsc[C.LABELS[L]]) else round(dsc[C.LABELS[L]], 4),
            gt_cm3=round(float((seg == L).sum())  * VOX_CM3, 2),
            pred_cm3=round(float((pred == L).sum())  * VOX_CM3, 2),
            confident_miss_cm3=round(float((m_fn & (conf > conf_thr)).sum())  * VOX_CM3, 3),
            confident_extra_cm3=round(float((m_fp & (conf > conf_thr)).sum())  * VOX_CM3, 3),
        )

    md = float(np.nanmean(list(dsc.values())))
    # Priority rises with lower Dice, larger suspect fraction, and worse
    # anatomical plausibility of the *human* annotation.
    anat_gt = anatomy_report(seg)
    priority = (1 - md) * 0.5 + susp_frac * 0.3 + min(anat_gt["total_penalty"], 1.0) * 0.2

    heat = suspect.astype(np.float32) * 0.6 + mism.astype(np.float32) * 0.2 + ent * 0.2
    return dict(case_id=cid, seg_folder=meta.get("seg_folder"), mean_dice=round(md, 4),
                per_class=per_class, suspect_frac=round(susp_frac, 4),
                suspect_cm3=round(float(suspect.sum())  * VOX_CM3, 2),
                anatomy_gt=anat_gt, priority=round(float(priority), 4)), \
           dict(img=img, seg=seg, pred=pred, heat=heat)


def round_drift() -> list[dict]:
    """Measure annotation drift between rounds on identical scans.

    Compares each superseded annotation (recorded by manifest.build) against the
    one that replaced it. Because both are human-delivered annotations of the
    *same* voxel grid, the resulting Dice is a direct estimate of inter-round
    annotator agreement -- i.e. the ceiling any model can be measured against.
    No model is involved.
    """
    import nibabel as nib
    mf = json.load(open(os.path.join(C.WORK, "manifest.json")))
    old = {(c["pid"], tuple(c["shape"])): c for c in mf["superseded"]}
    out = []
    for c in mf["pairs"]:
        k = (c["pid"], tuple(c["shape"]))
        if k not in old:
            continue
        o = old[k]
        a = np.asanyarray(nib.load(o["seg"]).dataobj)
        b = np.asanyarray(nib.load(c["seg"]).dataobj)
        vz = float(np.prod(nib.load(c["seg"]).header.get_zooms()[:3])) / 1000
        row = dict(case_id=c["case_id"], old_round=o["seg_round"], new_round=c["seg_round"])
        worst = 0.0
        for L in C.FG:
            va, vb = float((a == L).sum()) * vz, float((b == L).sum()) * vz
            d = dice(a == L, b == L)
            rel = abs(vb - va) / max(va, 1e-6)
            worst = max(worst, rel)
            row[C.LABELS[L]] = dict(old_cm3=round(va, 2), new_cm3=round(vb, 2),
                                    rel_change=round(rel, 3),
                                    dice=None if np.isnan(d) else round(d, 4))
        row["max_rel_change"] = round(worst, 3)
        out.append(row)
    return sorted(out, key=lambda r: -r["max_rel_change"])


def run(ckpt: str, case_ids: list[str] | None = None, make_png: bool = True,
        topk_png: int = 10) -> list[dict]:
    net, cfg, sd = load_checkpoint(ckpt)
    if case_ids is None:
        case_ids = sorted(json.load(open(f))["case_id"]
                          for f in glob.glob(os.path.join(C.CACHE, "*_meta.json")))
    rows, cache = [], {}
    for i, cid in enumerate(case_ids):
        r, arrs = review_case(net, cfg, cid)
        rows.append(r); cache[cid] = arrs
        print(f"[{i+1}/{len(case_ids)}] {cid} meanDice={r['mean_dice']:.4f} "
              f"可疑={r['suspect_cm3']:.2f}cm3 优先级={r['priority']:.3f}", flush=True)
    rows.sort(key=lambda r: -r["priority"])

    if make_png:
        for r in rows[:topk_png]:
            a = cache[r["case_id"]]
            r["png"] = overlay_sheet(
                a["img"], a["pred"], os.path.join(C.REVIEW, f"{r['case_id']}.png"),
                gt=a["seg"], unc=a["heat"], slices=top_slices(a["heat"], k=8),
                title=f"{r['case_id']}  Dice={r['mean_dice']:.3f}  "
                      f"可疑区={r['suspect_cm3']:.2f}cm3  (第二行=人工标注)")
    worklist_html([dict(case_id=r["case_id"], priority=r["priority"], png=r.get("png"),
                        reason=f"meanDice={r['mean_dice']:.3f} 可疑区={r['suspect_cm3']:.2f}cm3 "
                               f"解剖异常={r['anatomy_gt']['total_penalty']:.2f}")
                   for r in rows],
                  os.path.join(C.REVIEW, "worklist.html"), title="已标注数据 — 重标优先级")
    json.dump(rows, open(os.path.join(C.REVIEW, "review.json"), "w"), indent=1, ensure_ascii=False)
    drift = round_drift()
    json.dump(drift, open(os.path.join(C.REVIEW, "round_drift.json"), "w"), indent=1, ensure_ascii=False)
    if drift:
        print("\n跨轮次标注漂移 (同一次扫描被重标过):")
        for d in drift:
            print(f"  {d['case_id']}: 轮{d['old_round']}->轮{d['new_round']} "
                  f"最大相对变化={d['max_rel_change']:.1%}")
    return rows
