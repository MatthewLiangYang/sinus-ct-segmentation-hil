"""Active-learning round management.

One full HIL round:
  1. run the current model over the unlabelled pool -> uncertainty +
     anatomical-plausibility scores
  2. select k cases by uncertainty *combined with diversity*, so the batch is not
     five near-identical scans that waste the annotator's time
  3. export pre-annotations, uncertainty maps and priority-slice sheets for the
     annotator to correct rather than draw from scratch
  4. corrected cases join the training set, round += 1, retrain

state.json records which cases each round selected and the validation Dice at
that point, so the HIL learning curve can be plotted directly from it.
"""
from __future__ import annotations
import os, json, glob, shutil
import numpy as np
import torch
from . import config as C
from .inference import load_checkpoint, predict_probs, save_nifti
from .uncertainty import score_volume, top_slices
from .report import overlay_sheet, worklist_html

STATE = os.path.join(C.AL_DIR, "state.json")


# ---------------------------------------------------------------- state
def init_state(labeled: list[str]) -> dict:
    st = dict(round=0, labeled=sorted(labeled), history=[])
    save_state(st)
    return st


def load_state() -> dict:
    if not os.path.exists(STATE):
        from .train import all_cases
        return init_state(all_cases()[0])
    return json.load(open(STATE))


def save_state(st: dict):
    json.dump(st, open(STATE, "w"), indent=1, ensure_ascii=False)


# ------------------------------------------------------------ diversity
@torch.no_grad()
def embed(net, img: np.ndarray, cfg, n_patch: int = 8) -> np.ndarray:
    """Case fingerprint: globally pooled encoder bottleneck features.

    Used only for the diversity term in selection, never for prediction.
    """
    from .inference import _prep
    dev = C.device()
    x = _prep(img).to(dev)
    p = cfg.patch
    feats = []
    hook_out = {}

    def hook(_m, _i, o):
        hook_out["f"] = o

    # deepest bottleneck block of the DynUNet
    deepest = net.bottleneck if hasattr(net, "bottleneck") else list(net.children())[-1]
    h = deepest.register_forward_hook(hook)
    rng = np.random.default_rng(0)
    sh = x.shape[2:]
    for _ in range(n_patch):
        lo = [int(rng.integers(0, max(1, sh[i] - p[i] + 1))) for i in range(3)]
        sl = (slice(None), slice(None)) + tuple(slice(lo[i], lo[i] + p[i]) for i in range(3))
        try:
            net(x[sl])
        except Exception:
            continue
        f = hook_out.get("f")
        if f is not None:
            feats.append(f.float().mean(dim=(2, 3, 4))[0].cpu().numpy())
    h.remove()
    if not feats:
        return np.zeros(8, dtype=np.float32)
    v = np.mean(feats, 0)
    return v / (np.linalg.norm(v) + 1e-8)


def diverse_topk(scores: np.ndarray, feats: np.ndarray, k: int, lam: float = 0.5) -> list[int]:
    """Greedy selection maximising score - lam * max similarity to already-picked.

    Higher score means more worth annotating.
    """
    k = min(k, len(scores))
    picked: list[int] = []
    s = (scores - scores.min()) / (scores.ptp() + 1e-9)
    sim = feats @ feats.T if feats.size else np.zeros((len(scores), len(scores)))
    for _ in range(k):
        best, bv = -1, -1e18
        for i in range(len(scores)):
            if i in picked:
                continue
            pen = max((sim[i, j] for j in picked), default=0.0)
            v = s[i] - lam * pen
            if v > bv:
                best, bv = i, v
        picked.append(best)
    return picked


# -------------------------------------------------- scoring the unlabelled pool
def _cache_unlabeled(nii_path: str) -> tuple[np.ndarray, dict]:
    """Run an unlabelled CT through the identical preprocessing path.

    Must stay in sync with preprocess.process_one, otherwise pool scores are not
    comparable with validation numbers.
    """
    import nibabel as nib
    from scipy import ndimage
    PC = C.PreprocCfg()
    im = nib.as_closest_canonical(nib.load(nii_path))
    sp = tuple(float(z) for z in im.header.get_zooms()[:3])
    a = np.clip(np.asanyarray(im.dataobj).astype(np.float32), *PC.clip)
    zoom = [s / d for s, d in zip(sp, PC.spacing)]
    if not np.allclose(zoom, 1.0, atol=1e-3):
        a = ndimage.zoom(a, zoom, order=1, mode="nearest")
    meta = dict(case_id=os.path.basename(nii_path).replace(".nii.gz", ""),
                img=nii_path, orig_shape=list(im.shape), orig_zoom=list(sp),
                orig_affine=np.asarray(im.affine).tolist(),
                shape=list(a.shape), spacing=list(PC.spacing))
    return a.astype(np.int16), meta


def score_pool(ckpt: str, pool_dir: str, mc_samples: int = 6,
               out_dir: str | None = None) -> list[dict]:
    """Score every *.nii.gz under pool_dir. These are new, unlabelled cases."""
    net, cfg, _ = load_checkpoint(ckpt)
    out_dir = out_dir or os.path.join(C.AL_DIR, f"round{load_state()['round']}")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(pool_dir, "**", "*.nii.gz"), recursive=True))
    rows = []
    for i, f in enumerate(files):
        img, meta = _cache_unlabeled(f)
        s, maps = score_volume(net, cfg, img, mc_samples=mc_samples)
        e = embed(net, img, cfg)
        cid = meta["case_id"]
        np.save(os.path.join(out_dir, f"{cid}_unc.npy"), maps["uncertainty"].astype(np.float16))
        save_nifti(maps["label"], meta, os.path.join(out_dir, f"{cid}_pred.nii.gz"))
        png = overlay_sheet(img, maps["label"], os.path.join(out_dir, f"{cid}.png"),
                            unc=maps["uncertainty"],
                            slices=top_slices(maps["uncertainty"], k=8),
                            title=f"{cid}  composite={s['composite']:.3f}")
        rows.append(dict(case_id=cid, path=f, scores=s, embed=e.tolist(), png=png,
                         meta_shape=meta["shape"]))
        print(f"[{i+1}/{len(files)}] {cid} composite={s['composite']:.4f} "
              f"anat_pen={s['anatomy']['total_penalty']:.3f}", flush=True)
    json.dump(rows, open(os.path.join(out_dir, "pool_scores.json"), "w"), indent=1)
    return rows


def select_batch(rows: list[dict], k: int = 5, lam: float = 0.5) -> list[dict]:
    sc = np.array([r["scores"]["composite"] for r in rows], dtype=np.float64)
    fe = np.array([r["embed"] for r in rows], dtype=np.float32)
    idx = diverse_topk(sc, fe, k, lam=lam)
    out = []
    for rank, i in enumerate(idx, 1):
        r = rows[i]
        a = r["scores"]["anatomy"]
        bad = [n for n, v in a.items() if isinstance(v, dict) and v["penalty"] > 0.25]
        out.append(dict(case_id=r["case_id"], rank=rank,
                        priority=float(r["scores"]["composite"]), png=r.get("png"),
                        reason=f"熵={r['scores']['entropy']['mean']:.3f} "
                               f"MC翻转={r['scores']['mc_flip']:.3f} "
                               f"TTA={r['scores']['tta']['mean']:.3f} "
                               f"解剖异常={a['total_penalty']:.2f}"
                               + (f" [{','.join(C.LABELS_ZH[k2] for k2, n2 in C.LABELS.items() if n2 in bad)}]" if bad else "")))
    return out


def run_round(ckpt: str, pool_dir: str, k: int = 5, mc_samples: int = 6) -> dict:
    """Run one full AL round: score the pool, select a batch, emit the worklist."""
    st = load_state()
    rd = st["round"]
    out_dir = os.path.join(C.AL_DIR, f"round{rd}")
    rows = score_pool(ckpt, pool_dir, mc_samples=mc_samples, out_dir=out_dir)
    sel = select_batch(rows, k=k)
    worklist_html(sel, os.path.join(out_dir, "worklist.html"),
                  title=f"AL 第 {rd} 轮 — 建议优先标注的 {len(sel)} 例")
    json.dump(sel, open(os.path.join(out_dir, "selected.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n第 {rd} 轮建议标注：")
    for s in sel:
        print(f"  {s['rank']}. {s['case_id']}  优先级={s['priority']:.4f}  {s['reason']}")
    print(f"\n工单: {os.path.join(out_dir,'worklist.html')}")
    print(f"预标注 mask 与不确定性图在: {out_dir}")
    return dict(round=rd, selected=sel, out_dir=out_dir)


def commit_round(new_labeled: list[str], val_mean_dice: float | None = None):
    """Call once the annotator has finished: record the round and advance the counter."""
    st = load_state()
    st["history"].append(dict(round=st["round"], added=sorted(new_labeled),
                              n_labeled_after=len(set(st["labeled"]) | set(new_labeled)),
                              val_mean_dice=val_mean_dice))
    st["labeled"] = sorted(set(st["labeled"]) | set(new_labeled))
    st["round"] += 1
    save_state(st)
    print(f"-> 进入第 {st['round']} 轮，labeled={len(st['labeled'])}")
    return st
