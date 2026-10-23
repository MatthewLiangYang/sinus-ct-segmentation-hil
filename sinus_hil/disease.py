"""Derive disease-related quantitative metrics from a segmentation.

CRITICAL PREMISE, verified against the data: the annotations delineate the
**aerated cavity**, not the bony sinus cavity. Mean intensity inside every
labelled structure is -600 to -940 HU, with soft-tissue fraction only 2-23%.

The consequence drives this whole module: disease (mucosal thickening, fluid,
polyps) shows up as the *air cavity shrinking*, not as the cavity brightening,
because the annotator simply excluded the diseased tissue from the mask.

A Lund-Mackay style opacification score therefore needs the bony envelope as a
denominator, which has to be inferred. See _focal_envelope for two approaches
that were tried and failed, and why the final answer is two-track:

  - focal disease     -> bounded morphological closing of the air cavity
  - global disease    -> comparison against cohort normative volumes (z-score)

Outputs:
  A. per-sinus, per-side metrics (volume, HU distribution, opacification,
     left-right asymmetry, normative z-score)
  B. an involvement score plus a classifier scaffold ready for disease labels

NOTE: this dataset carries no disease diagnosis labels at all, so the thresholds
in B are uncalibrated. Call fit_classifier() once labels exist to replace the
rule-based score with a supervised model.
"""
from __future__ import annotations
import os, json, glob
import numpy as np
from scipy import ndimage
from . import config as C

BONE_HU = 300.0        # bone wall threshold (HU)
AIR_HU = -500.0        # aerated threshold (HU)
SOFT_LO, SOFT_HI = -300.0, 200.0   # HU window counted as soft tissue inside a cavity
SOFT_FLOOR = -400.0    # lowest HU envelope growth may cross; below this is air
ENVELOPE_R = 6         # closing radius in voxels; at 0.8 mm spacing = 4.8 mm,
                       # so only concavities narrower than ~9.6 mm get filled
# Sinuses with a bony envelope, for which opacification is meaningful. The nasal
# cavity (1) and ethmoid (2) are open / honeycombed and are handled separately.
ENVELOPE_LABELS = (3, 4, 5)
GRADE_T1, GRADE_T2 = 0.60, 1.20   # involvement score -> grade 0/1/2 cut-points
# Asymmetry alert thresholds. Frontal and sphenoid sinuses are naturally highly
# asymmetric, so their thresholds must be loose or half the cohort is flagged.
ASYM_ALERT = {"maxillary": 0.35, "ethmoid": 0.40, "frontal": 0.65, "sphenoid": 0.65, "nasal_cavity": 0.25}


def _bbox(m: np.ndarray, pad: int):
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, m.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _focal_envelope(air: np.ndarray, ct: np.ndarray, r: int = ENVELOPE_R) -> np.ndarray:
    """Estimate the bony sinus envelope, bounded version.

    Two approaches were tried first and both failed. Recording them so nobody
    repeats the work:

      1) Geodesic dilation outward from the air cavity, constrained by bone.
         Leaks straight through the ostium -- which is itself an air channel --
         into the nasal cavity. Result: every maxillary sinus in the cohort
         scored ~65% opacified, with the "filled" region averaging -76 HU
         (i.e. still air).
      2) Growth restricted to soft-tissue HU. No longer leaks through the
         ostium, but escapes through vascular foramina and thin spots in the
         bony wall into the cheek. The filled region averaged 17-45 HU, which is
         muscle and fat, so the estimate was still inflated.

    Conclusion: **the bony cavity volume cannot be recovered from the air cavity
    alone.** This function therefore uses morphological closing, which is a
    bounded operator (dilate then erode) and by construction only fills
    concavities and gaps narrower than 2r. It cannot run away.

    That makes it reliable for *focal* disease -- air-fluid levels, polyps, local
    mucosal thickening. It is blind to *global* disease, where the whole cavity
    shrinks uniformly: that is not a concavity. Global disease is covered by the
    cohort normative layer at the bottom of this module.

    Validation: focal opacification lands at 0.005-0.10 on this cohort, which is
    the expected range for largely normal sinuses.
    """
    out = np.zeros_like(air)
    if not air.any():
        return out
    sl = _bbox(air, r + 3)
    a, c = air[sl], ct[sl]
    closed = ndimage.binary_closing(a, iterations=r, border_value=0)
    env = ndimage.binary_fill_holes(closed & (c < BONE_HU)) | a
    out[sl] = env
    return out


def _side_split(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a mask into left and right halves.

    x is the left-right axis in RAS. The midline comes from the foreground
    centroid rather than the geometric centre, which is more robust to an
    off-centre field of view.
    """
    if not mask.any():
        mid = mask.shape[0] // 2
    else:
        mid = int(round(np.argwhere(mask)[:, 0].mean()))
    left = np.zeros_like(mask); right = np.zeros_like(mask)
    left[mid:] = mask[mid:]            # RAS: +x is the patient's left
    right[:mid] = mask[:mid]
    return left, right


def sinus_metrics(ct: np.ndarray, seg: np.ndarray, spacing=None) -> dict:
    spacing = tuple(spacing or C.PreprocCfg().spacing)
    vz = float(np.prod(spacing)) / 1000.0
    ct = ct.astype(np.float32)
    # Midline from the nasal cavity centroid: it is the only single-component
    # midline structure, so it is the most stable reference available.
    nasal = seg == 1
    mid = int(round(np.argwhere(nasal)[:, 0].mean())) if nasal.any() else seg.shape[0] // 2

    out: dict = dict(midline_x=mid, spacing=list(spacing))
    for L in C.FG:
        air = seg == L
        name = C.LABELS[L]
        rec: dict = {}
        for side, m in (("L", _mask_side(air, mid, "L")), ("R", _mask_side(air, mid, "R")),
                        ("both", air)):
            if not m.any():
                rec[side] = dict(air_cm3=0.0, env_cm3=0.0, opacification=None,
                                 focal_opacification=None, mean_HU=None, soft_frac=None)
                continue
            air_v = float(m.sum()) * vz
            hu = ct[m]
            d = dict(air_cm3=round(air_v, 2), mean_HU=round(float(hu.mean()), 1),
                     p90_HU=round(float(np.percentile(hu, 90)), 1),
                     soft_frac=round(float(((hu > SOFT_LO) & (hu < SOFT_HI)).mean()), 4))
            # soft_frac = fraction of voxels inside the annotated air cavity that are
            # already soft-tissue density; catches fluid levels and polypoid filling
            if L in ENVELOPE_LABELS:
                env = _focal_envelope(m, ct)
                env_v = float(env.sum()) * vz
                opac = max(0.0, 1.0 - air_v / max(env_v, 1e-6))
                filled = env & ~m
                d.update(env_cm3=round(env_v, 2), focal_opacification=round(opac, 4),
                         opacification=round(opac, 4),
                         filled_mean_HU=round(float(ct[filled].mean()), 1) if filled.any() else None)
            else:
                d.update(env_cm3=None, opacification=None, focal_opacification=None)
            rec[side] = d
        # Left-right asymmetry. Sinus disease is often unilateral, so asymmetry is
        # a strong signal in its own right.
        al, ar = rec["L"]["air_cm3"], rec["R"]["air_cm3"]
        rec["asym"] = round(abs(al - ar) / max(al + ar, 1e-6), 4)
        out[name] = rec
    hw = head_size_mm(ct, spacing)
    out["head_width_mm"] = round(hw, 1)
    out["head_scale"] = round(hw / 160.0, 4)      # relative to a 160 mm reference head width
    out["flags"] = []
    return out


def finalize(row: dict, norm: dict | None = None) -> dict:
    """Attach normative z-scores, then grade, then raise flags.

    The order is load-bearing: the involvement score reads norm_z, and the flags
    read the grades.
    """
    row = apply_normative(row, norm)
    row["involvement"] = _grade_all(row)
    row["flags"] = row.get("flags", []) + _flags(row)
    return row


def _mask_side(mask: np.ndarray, mid: int, side: str) -> np.ndarray:
    o = np.zeros_like(mask)
    if side == "L":
        o[mid:] = mask[mid:]
    else:
        o[:mid] = mask[:mid]
    return o


def involvement_score(d: dict, asym: float = 0.0) -> float:
    """Composite involvement score for one sinus on one side.

    0 = normal, >= 1.2 = strongly suspicious.

    Why not just use opacification: measured across all 50 maxillary sinuses in
    this cohort, focal opacification has median 0.013 and maximum 0.029. The
    Lund-Mackay cut-points of 0.15 / 0.85 would never fire once. The reason is
    the air-cavity-only annotation -- uniform mucosal thickening shrinks the
    whole cavity proportionally, which is not a concavity, so bounded closing
    cannot see it by definition.

    This score therefore takes the maximum of four signals, so any one of them
    being abnormal lifts the result:

      focal opacification  fluid levels, polyps, local thickening
      in-cavity soft_frac  soft-tissue density already inside the annotated air
      normative z-score    uniform cavity shrinkage (closing's blind spot)
      left-right asymmetry unilateral disease
    """
    o = d.get("focal_opacification") or 0.0
    sf = d.get("soft_frac") or 0.0
    z = d.get("norm_z")
    parts = [o / 0.25, sf / 0.30]
    if z is not None:
        parts.append(max(0.0, -float(z)) / 3.0)
    parts.append(max(0.0, asym - 0.30) / 0.40)
    return float(max(parts))


def grade_from_score(v: float) -> int:
    """Involvement score -> grade 0/1/2. Thresholds are UNCALIBRATED; see README."""
    if v < GRADE_T1:
        return 0
    return 1 if v < GRADE_T2 else 2


def _grade_all(m: dict) -> dict:
    """Grade 4 sinuses x 2 sides on 0/1/2 and sum to a 0-16 involvement total.

    Deliberately NOT called Lund-Mackay. This annotation scheme has no
    anterior/posterior ethmoid split and no ostiomeatal complex score, and the
    thresholds are uncalibrated. Using the name would imply the number is
    comparable with published LM scores, which it is not.
    """
    per, score, tot = {}, {}, 0
    for name in ("maxillary", "ethmoid", "frontal", "sphenoid"):
        asym = m[name]["asym"]
        for side in ("L", "R"):
            v = involvement_score(m[name][side], asym)
            g = grade_from_score(v)
            per[f"{name}_{side}"] = g
            score[f"{name}_{side}"] = round(v, 3)
            m[name][side]["involvement"] = round(v, 3)
            m[name][side]["grade"] = g
            tot += g
    return dict(per_sinus=per, per_sinus_score=score, total=tot, max_possible=16,
                note="自定义受累评分，非标准 Lund-Mackay；阈值未经诊断标签标定")


def _zh(name: str) -> str:
    for k, v in C.LABELS.items():
        if v == name:
            return C.LABELS_ZH.get(k, name)
    return name


def _flags(m: dict) -> list[str]:
    f = []
    for name in ("maxillary", "ethmoid", "frontal", "sphenoid"):
        for side in ("L", "R"):
            d = m[name][side]
            g = d.get("grade", 0)
            if g >= 1:
                f.append(f"{_zh(name)}{'左' if side == 'L' else '右'}侧受累 "
                         f"(评分 {d.get('involvement')}, 分级 {g})")
        if m[name]["asym"] > ASYM_ALERT.get(name, 0.5):
            f.append(f"{_zh(name)} 左右不对称 {m[name]['asym']:.0%} "
                     f"(超过 {ASYM_ALERT[name]:.0%}，额窦/蝶窦本就常不对称，需人工确认)")
    if m["frontal"]["both"]["air_cm3"] < 0.5:
        f.append("额窦几乎无气化（多为先天缺如，不一定是病变）")
    return f


FEATURE_ORDER: list[str] = []


def feature_vector(m: dict) -> tuple[np.ndarray, list[str]]:
    """Flatten metrics into a fixed-length feature vector for a downstream classifier."""
    vals, names = [], []
    for name in ("nasal_cavity", "ethmoid", "maxillary", "frontal", "sphenoid"):
        for side in ("L", "R"):
            d = m[name][side]
            for k in ("air_cm3", "env_cm3", "focal_opacification", "mean_HU", "p90_HU",
                      "soft_frac", "involvement"):
                v = d.get(k)
                vals.append(np.nan if v is None else float(v))
                names.append(f"{name}_{side}_{k}")
            vals.append(d.get("norm_z", np.nan) if d.get("norm_z") is not None else np.nan)
            names.append(f"{name}_{side}_norm_z")
        vals.append(m[name]["asym"]); names.append(f"{name}_asym")
    vals.append(m.get("involvement", {}).get("total", np.nan)); names.append("involvement_total")
    vals.append(m.get("head_scale", np.nan)); names.append("head_scale")
    return np.array(vals, dtype=np.float32), names


def run_from_cache(case_ids: list[str] | None = None, use_pred_ckpt: str | None = None) -> list[dict]:
    """Compute metrics for cached cases.

    Uses the manual annotations by default. Pass a checkpoint to use model
    predictions instead, which is what deployment would actually look like.
    """
    from .dataset import load_case
    if case_ids is None:
        case_ids = sorted(json.load(open(f))["case_id"]
                          for f in glob.glob(os.path.join(C.CACHE, "*_meta.json")))
    net = cfg = None
    if use_pred_ckpt:
        from .inference import load_checkpoint
        net, cfg, _ = load_checkpoint(use_pred_ckpt)
    rows = []
    for i, cid in enumerate(case_ids):
        img, seg, meta = load_case(cid)
        img = np.asarray(img)
        if net is not None:
            from .inference import predict_probs
            seg = predict_probs(net, img, cfg, tta=True).argmax(0).astype(np.uint8)
        else:
            seg = np.asarray(seg)
        m = sinus_metrics(img, seg)
        m["case_id"] = cid
        m["source"] = "model" if net is not None else "manual"
        rows.append(m)
        print(f"[{i+1}/{len(case_ids)}] {cid}", flush=True)
    norm = build_normative(rows)
    rows = [finalize(r, norm) for r in rows]
    out = os.path.join(C.WORK, f"disease_metrics_{'model' if net else 'manual'}.json")
    json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)

    for i, r in enumerate(rows):
        print(f"  {r['case_id']:<18} 受累总分={r['involvement']['total']:>2}/16"
              + ("  " + "; ".join(r["flags"][:2]) if r["flags"] else ""))
    X = np.stack([feature_vector(r)[0] for r in rows])
    names = feature_vector(rows[0])[1]
    np.savez(os.path.join(C.WORK, "disease_features.npz"), X=X,
             names=np.array(names), case_ids=np.array([r["case_id"] for r in rows]))
    print(f"\n写入 {out}  特征矩阵 {X.shape} -> work/disease_features.npz")
    return rows


def fit_classifier(labels_csv: str, out_model: str | None = None):
    """Train the disease classifier once labels are available.

    labels_csv needs two columns: case_id,label -- label being 0/1, or a score
    that will be binarised at > 0.
    """
    import csv
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import joblib

    z = np.load(os.path.join(C.WORK, "disease_features.npz"), allow_pickle=True)
    X, cids = z["X"], list(z["case_ids"])
    lab = {r["case_id"]: float(r["label"]) for r in csv.DictReader(open(labels_csv))}
    keep = [i for i, c in enumerate(cids) if c in lab]
    if not keep:
        raise SystemExit("labels_csv 里的 case_id 与特征矩阵无交集")
    Xs = X[keep]
    ys = np.array([lab[cids[i]] for i in keep])
    y = (ys > 0).astype(int) if set(np.unique(ys)) - {0, 1} else ys.astype(int)
    pipe = make_pipeline(SimpleImputer(strategy="median"),
                         GradientBoostingClassifier(random_state=0))
    n = min(5, int(np.bincount(y).min()))
    if n >= 2:
        cv = StratifiedKFold(n, shuffle=True, random_state=0)
        s = cross_val_score(pipe, Xs, y, cv=cv, scoring="roc_auc")
        print(f"{n} 折交叉验证 AUC = {s.mean():.3f} ± {s.std():.3f}  (n={len(y)})")
    pipe.fit(Xs, y)
    out_model = out_model or os.path.join(C.WORK, "disease_clf.joblib")
    joblib.dump(dict(model=pipe, names=list(z["names"])), out_model)
    print(f"模型写入 {out_model}")
    return pipe


# ====================================================================
# Cohort normative layer.
#
# The closing-based envelope only catches focal disease. Uniform shrinkage or
# complete opacification of a whole cavity has to be caught by comparison against
# the cohort: normalise volumes by head size, then compute a robust z-score
# (median / MAD) per sinus per side.
# ====================================================================
NORM_FILE = os.path.join(C.WORK, "normative.json")


def head_size_mm(ct: np.ndarray, spacing=None) -> float:
    """Head width in mm, used to cancel out body-size differences.

    Takes the x extent of tissue (HU > -500) at the 90th percentile across
    sampled slices.
    """
    spacing = tuple(spacing or C.PreprocCfg().spacing)
    head = ct > -500
    # sample slices and take a high percentile, avoiding FOV edge artefacts
    widths = []
    for z in range(0, head.shape[2], max(1, head.shape[2] // 24)):
        xs = np.argwhere(head[:, :, z])
        if len(xs) > 500:
            widths.append((xs[:, 0].max() - xs[:, 0].min() + 1) * spacing[0])
    return float(np.percentile(widths, 90)) if widths else 150.0


def _norm_key(name: str, side: str) -> str:
    return f"{name}_{side}"


def build_normative(rows: list[dict], out: str = NORM_FILE) -> dict:
    """Build the normative table from a set of cases, ideally radiologically normal.

    CAVEAT: this cohort is specialist sinus CT and is almost certainly enriched
    for disease rather than representative of normals. What this computes is
    closer to "the median of our cohort" than "normal". Rebuild it once a proper
    normal control group is available.
    """
    stats = {}
    for name in ("nasal_cavity", "ethmoid", "maxillary", "frontal", "sphenoid"):
        for side in ("L", "R"):
            v = np.array([r[name][side]["air_cm3"] / (r["head_scale"] ** 3)
                          for r in rows if r.get("head_scale")], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size < 3:
                continue
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-6
            stats[_norm_key(name, side)] = dict(median=round(med, 4), mad=round(mad, 4),
                                                n=int(v.size),
                                                p10=round(float(np.percentile(v, 10)), 4))
    d = dict(stats=stats, note="体积已按 (头宽/160mm)^3 标准化；单位 cm^3", ref_head_mm=160.0)
    json.dump(d, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"正常值表写入 {out} ({len(stats)} 项)")
    return d


def apply_normative(row: dict, norm: dict | None = None) -> dict:
    """Attach z-scores and a global hypoaeration flag to one case."""
    if norm is None:
        if not os.path.exists(NORM_FILE):
            return row
        norm = json.load(open(NORM_FILE))
    st = norm["stats"]
    z = {}
    for name in ("nasal_cavity", "ethmoid", "maxillary", "frontal", "sphenoid"):
        for side in ("L", "R"):
            k = _norm_key(name, side)
            if k not in st or not row.get("head_scale"):
                continue
            v = row[name][side]["air_cm3"] / (row["head_scale"] ** 3)
            zz = (v - st[k]["median"]) / st[k]["mad"]
            z[k] = round(float(zz), 2)
            row[name][side]["norm_z"] = z[k]
            # z < -2, excluding frontal aplasia, suggests global hypoaeration
            if zz < -2.0 and not (name == "frontal" and v < 0.3):
                row.setdefault("flags", []).append(
                    f"{C.LABELS_ZH.get([kk for kk,vv in C.LABELS.items() if vv==name][0], name)}"
                    f"{'左' if side=='L' else '右'}侧含气量显著低于队列 (z={zz:.1f})")
    row["norm_z"] = z
    row["global_hypoaeration"] = sorted([k for k, v in z.items() if v < -2.0])
    return row


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="给了就用模型预测代替人工标注")
    run_from_cache(use_pred_ckpt=ap.parse_args().ckpt)
