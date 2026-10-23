"""Uncertainty and anatomical-plausibility scoring: the ranking signal for
active learning.

For an unlabelled volume we compute four complementary signals:
  1. entropy      predictive entropy -- where the model knows it is unsure
  2. mc_dropout   variance / label-flip rate across MC-dropout samples
                  (model-parameter uncertainty)
  3. tta_disagree disagreement between flip-augmented predictions -- where the
                  model is not robust to spatial transforms
  4. anatomy      violation of anatomical priors (component count, volume,
                  left-right symmetry)

Signal 4 is the one that matters most in practice: it catches the cases where the
model is *confidently wrong*, which entropy-based measures by construction cannot
see. Measured baseline: human annotations score 0.016 on this penalty, model
output 0.036.

The weighted composite decides which cases go to the annotator next, and which
slices within a case they should look at first.
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy import ndimage
from . import config as C
from .inference import predict_probs, predict_mc, entropy

WEIGHTS = dict(entropy=0.30, mc=0.25, tta=0.20, anatomy=0.25)


# ------------------------------------------------- anatomical plausibility
def anatomy_report(lbl: np.ndarray, spacing=None) -> dict:
    """Check each structure's component count, volume and symmetry against priors.

    Two details that are easy to get wrong, both found by checking the score on
    known-good human annotations:

      - The midline must come from the *nasal cavity centroid*, not the image
        centre. Fields of view are often off-centre, and using the image centre
        reports a spurious ~0.28 asymmetry on normal cases.
      - "Fragmentation" must be the volume fraction held by small fragments, not
        1 - largest/total. The maxillary sinus is supposed to have two
        equally-sized components, so the latter penalises a perfect annotation
        by 0.25.
    """
    spacing = tuple(spacing or C.PreprocCfg().spacing)
    vox = float(np.prod(spacing)) / 1000.0            # cm^3 / voxel
    nasal = lbl == 1
    mid = int(round(np.argwhere(nasal)[:, 0].mean())) if nasal.any() else lbl.shape[0] // 2
    out, pen = {}, 0.0
    for L in C.FG:
        m = lbl == L
        n = int(m.sum())
        vol = n * vox
        cc_rng, vol_rng = C.ANAT_PRIOR[L]
        if n == 0:
            cc, sym, frag = 0, np.nan, 0.0
        else:
            cl, _ = ndimage.label(m)
            sz = np.bincount(cl.ravel())[1:]
            cc = int((sz * vox > 0.05).sum())          # ignore fragments below 0.05 cm^3
            l_, r_ = m[mid:].sum(), m[:mid].sum()      # RAS: +x is the patient's left
            sym = float(abs(int(l_) - int(r_)) / max(int(l_) + int(r_), 1))
            # fragmentation = volume fraction in components under 10% of the largest
            frag = float(sz[sz < 0.1 * sz.max()].sum() / sz.sum())
        p_cc = (max(0.0, cc_rng[0] - cc) + max(0.0, cc - cc_rng[1])) / max(cc_rng[1], 1)
        p_vol = (max(0.0, vol_rng[0] - vol) + max(0.0, vol - vol_rng[1])) / max(vol_rng[1], 1e-6)
        # Strong asymmetry in a paired structure is suspicious, but it can also be
        # genuine unilateral disease, hence the reduced weight.
        p_sym = 0.0 if np.isnan(sym) else max(0.0, sym - 0.45) / 0.55
        score = p_cc + p_vol + 0.5 * p_sym + 0.5 * frag
        out[C.LABELS[L]] = dict(vol_cm3=round(vol, 2), cc=cc,
                                asym=None if np.isnan(sym) else round(sym, 3),
                                frag=round(frag, 3), penalty=round(float(score), 3))
        pen += score
    out["total_penalty"] = round(float(pen / len(C.FG)), 4)
    return out


# ------------------------------------------------------ volume-level scoring
def score_volume(net, cfg, img: np.ndarray, mc_samples: int = 6,
                 roi: np.ndarray | None = None) -> dict:
    """Score one volume.

    Returns (summary dict, voxel-wise maps). The maps feed top_slices() to pick
    which slices a human should review.
    """
    prob = predict_probs(net, img, cfg, tta=False)
    lbl = prob.argmax(0).astype(np.uint8)
    ent = entropy(prob)

    # Restrict statistics to predicted foreground plus a margin. Averaging over the
    # whole volume would dilute every score with large confident air regions.
    if roi is None:
        fg = lbl > 0
        roi = ndimage.binary_dilation(fg, iterations=6)
    roi = roi & (prob[0] < 0.999)
    if roi.sum() < 1000:
        roi = np.ones_like(lbl, dtype=bool)

    prob_t = predict_probs(net, img, cfg, tta=True)
    tta_dis = np.abs(prob_t - prob).sum(0) / 2.0

    mean_mc, var_mc, flip = predict_mc(net, img, cfg, n_samples=mc_samples)

    anat = anatomy_report(lbl, cfg_spacing(cfg))

    def agg(v):
        x = v[roi]
        return dict(mean=float(x.mean()), p95=float(np.percentile(x, 95)),
                    frac_hi=float((x > 0.3).mean()))

    s = dict(
        entropy=agg(ent), mc_var=agg(var_mc), mc_flip=float(flip[roi].mean()),
        tta=agg(tta_dis), anatomy=anat,
        pred_voxels={C.LABELS[L]: int((lbl == L).sum()) for L in C.FG},
    )
    s["composite"] = float(
        WEIGHTS["entropy"] * s["entropy"]["mean"] * 4
        + WEIGHTS["mc"] * (s["mc_var"]["mean"] * 8 + s["mc_flip"] * 2) / 2
        + WEIGHTS["tta"] * s["tta"]["mean"] * 4
        + WEIGHTS["anatomy"] * min(anat["total_penalty"], 1.5) / 1.5
    )
    maps = dict(label=lbl, entropy=ent, mc_var=var_mc, mc_flip=flip, tta=tta_dis,
                uncertainty=np.clip(0.5 * ent + 0.3 * tta_dis + 0.2 * flip, 0, 1))
    return s, maps


def cfg_spacing(cfg) -> tuple:
    return tuple(C.PreprocCfg().spacing)


def top_slices(unc: np.ndarray, k: int = 8, axis: int = 2, min_gap: int = 6) -> list[int]:
    """Pick the k highest-uncertainty slices, spaced at least min_gap apart.

    The spacing constraint matters: without it the top slices are all neighbours
    of a single problem region and the reviewer sees the same error eight times.
    """
    prof = unc.sum(axis=tuple(a for a in range(3) if a != axis))
    order = np.argsort(prof)[::-1]
    picked: list[int] = []
    for i in order:
        if all(abs(int(i) - p) >= min_gap for p in picked):
            picked.append(int(i))
        if len(picked) >= k:
            break
    return sorted(picked)
