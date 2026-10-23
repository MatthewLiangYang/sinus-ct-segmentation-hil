"""Training entry point.

Patient-grouped K-fold cross-validation, patch-based training, sliding-window
validation on whole volumes. Validation cases are never seen during training.
"""
from __future__ import annotations
import os, json, time, glob, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import config as C
from .dataset import SinusPatches, splits, load_case
from .model import build_model, count_params
from .losses import DeepSupLoss, dice_per_class
from .inference import predict_probs


def all_cases() -> tuple[list[str], list[str]]:
    metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(C.CACHE, "*_meta.json")))]
    metas.sort(key=lambda m: m["case_id"])
    return [m["case_id"] for m in metas], [m["pid"] for m in metas]


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def class_weights(case_ids: list[str]) -> torch.Tensor:
    """Inverse-square-root frequency weights for the cross-entropy term.

    Lifts the small structures (ethmoid, sphenoid, frontal) that would otherwise
    be swamped by the maxillary sinus and nasal cavity.
    """
    cnt = np.zeros(C.NUM_CLASSES)
    for cid in case_ids:
        m = json.load(open(os.path.join(C.CACHE, f"{cid}_meta.json")))
        for k, v in m["fg_voxels"].items():
            cnt[int(k)] += v
    cnt[0] = cnt[1:].sum()                       # damp background using total foreground mass
    w = 1.0 / np.maximum(cnt, 1) ** 0.5
    w = w / w.sum() * C.NUM_CLASSES
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def validate(net, cfg, case_ids, dev, tta=False) -> dict:
    net.eval()
    rows, per_class = [], []
    for cid in case_ids:
        img, seg, meta = load_case(cid)
        prob = predict_probs(net, np.asarray(img), cfg, dev=dev, tta=tta)
        pred = torch.from_numpy(prob.argmax(0).astype(np.int16))
        d = dice_per_class(pred, torch.from_numpy(np.asarray(seg).astype(np.int16)))
        per_class.append(d)
        rows.append(dict(case_id=cid, dice={C.LABELS[L]: (None if np.isnan(x) else round(x, 4))
                                            for L, x in zip(C.FG, d)},
                         mean_dice=float(np.nanmean(d))))
    arr = np.array(per_class, dtype=np.float64)
    return dict(cases=rows,
                dice_per_class={C.LABELS[L]: float(np.nanmean(arr[:, i])) for i, L in enumerate(C.FG)},
                mean_dice=float(np.nanmean(arr)))


def train_fold(fold: int, cfg: C.TrainCfg, n_folds: int = 5, tag: str = "base",
               extra_train: list[str] | None = None, resume: str | None = None):
    seed_all(cfg.seed + fold)
    dev = C.device()
    ids, pids = all_cases()
    sp = splits(ids, pids, n_folds=n_folds, seed=cfg.seed)[fold]
    tr, va = sp["train"], sp["val"]
    if extra_train:
        tr = sorted(set(tr) | set(extra_train))
    out = os.path.join(C.RUNS, f"{tag}_fold{fold}")
    os.makedirs(out, exist_ok=True)

    print(f"== fold {fold} | train {len(tr)} / val {len(va)} | device {dev} ==", flush=True)
    print(f"   val: {va}", flush=True)
    ds = SinusPatches(tr, cfg, train=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                    shuffle=False, drop_last=True, pin_memory=False)

    net = build_model(cfg).to(dev)
    print(f"   params {count_params(net)/1e6:.2f}M", flush=True)
    loss_fn = DeepSupLoss(class_weights=class_weights(tr).to(dev)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * cfg.iters_per_epoch,
        pct_start=0.1, div_factor=10, final_div_factor=100)

    start_ep, best = 0, -1.0
    if resume and os.path.exists(resume):
        sd = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"]); start_ep = sd["epoch"] + 1; best = sd.get("best", -1.0)
        print(f"   resumed from {resume} at epoch={start_ep}")

    hist = []
    log_f = os.path.join(out, "log.jsonl")
    for ep in range(start_ep, cfg.epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            l = loss_fn(net(x), y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
            opt.step(); sched.step()
            tot += float(l.detach())
            if i + 1 >= cfg.iters_per_epoch:
                break
        tr_loss = tot / max(i + 1, 1)
        rec = dict(epoch=ep, loss=round(tr_loss, 4), lr=round(sched.get_last_lr()[0], 6),
                   sec=round(time.time() - t0, 1))

        last = (ep == cfg.epochs - 1)
        if (ep + 1) % cfg.val_every == 0 or last:
            v = validate(net, cfg, va, dev)
            rec["val"] = {k: round(x, 4) for k, x in v["dice_per_class"].items()}
            rec["val_mean_dice"] = round(v["mean_dice"], 4)
            if v["mean_dice"] > best:
                best = v["mean_dice"]
                torch.save(dict(model=net.state_dict(), cfg=C.dump(cfg), epoch=ep,
                                best=best, fold=fold, train=tr, val=va,
                                val_detail=v), os.path.join(out, "best.pt"))
            json.dump(v, open(os.path.join(out, f"val_ep{ep}.json"), "w"), indent=1)
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        cfg=C.dump(cfg), epoch=ep, best=best, fold=fold, train=tr, val=va),
                   os.path.join(out, "last.pt"))
        hist.append(rec)
        with open(log_f, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"ep{ep:03d} loss={rec['loss']:.4f} lr={rec['lr']:.2e} {rec['sec']:.0f}s"
              + (f" | valDice={rec.get('val_mean_dice')} {rec.get('val')}" if "val" in rec else ""),
              flush=True)
    print(f"== fold {fold} done, best mean Dice = {best:.4f} ==")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    cfg = C.TrainCfg()
    if a.epochs: cfg.epochs = a.epochs
    if a.patch:  cfg.patch = (a.patch,) * 3
    if a.iters:  cfg.iters_per_epoch = a.iters
    train_fold(a.fold, cfg, n_folds=a.folds, tag=a.tag, resume=a.resume)
