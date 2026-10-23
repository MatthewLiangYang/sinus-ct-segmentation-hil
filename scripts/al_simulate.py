#!/usr/bin/env python
"""Controlled back-test: does active learning actually beat random selection?

Method: pretend only a few cases are labelled and treat the rest as an
unlabelled pool.
  - AL strategy:     each round, score the pool with the current model and
                     "reveal" the labels of the top k by uncertainty x diversity
  - Random strategy: each round, reveal k at random
Both curves are evaluated on the same held-out test set, which neither strategy
can draw from. If the AL curve is higher at equal annotation budget, it is worth
spending annotator time on the cases AL picks.

This is the experiment that justifies (or refutes) the whole active-learning
component on this specific data distribution. It has not been run yet.

Usage:
  python scripts/al_simulate.py --quick          # ~2 h on an M2 Max
  python scripts/al_simulate.py --epochs 30      # slower, more trustworthy
"""
from __future__ import annotations
import os, sys, json, argparse, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sinus_hil import config as C
from sinus_hil.train import all_cases, train_fold, validate, class_weights, seed_all
from sinus_hil.dataset import SinusPatches, load_case
from sinus_hil.model import build_model
from sinus_hil.losses import DeepSupLoss
from sinus_hil.uncertainty import score_volume
from sinus_hil.active_learning import diverse_topk, embed
from torch.utils.data import DataLoader


def quick_train(train_ids, cfg, dev):
    seed_all(cfg.seed)
    ds = SinusPatches(train_ids, cfg, train=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
    net = build_model(cfg).to(dev)
    lf = DeepSupLoss(class_weights=class_weights(train_ids).to(dev)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                              total_steps=cfg.epochs * cfg.iters_per_epoch,
                                              pct_start=0.15)
    for ep in range(cfg.epochs):
        net.train()
        for i, (x, y) in enumerate(dl):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            lf(net(x), y).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 12.0)
            opt.step(); sch.step()
            if i + 1 >= cfg.iters_per_epoch:
                break
    return net


def pick_al(net, cfg, pool, k, mc=4):
    sc, fe = [], []
    for cid in pool:
        img, _, _ = load_case(cid)
        img = np.asarray(img)
        s, _ = score_volume(net, cfg, img, mc_samples=mc)
        sc.append(s["composite"]); fe.append(embed(net, img, cfg))
    idx = diverse_topk(np.array(sc), np.array(fe, dtype=np.float32), k)
    return [pool[i] for i in idx], dict(zip(pool, map(float, sc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-n", type=int, default=6)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--test-n", type=int, default=6)
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.iters, a.rounds = 14, 30, 3

    dev = C.device()
    ids, pids = all_cases()
    rng = random.Random(0)
    # hold out a fixed test set, grouped by patient
    upid = sorted(set(pids)); rng.shuffle(upid)
    test_pid = set(upid[: a.test_n])
    test = [c for c, p in zip(ids, pids) if p in test_pid]
    rest = [c for c, p in zip(ids, pids) if p not in test_pid]
    rng.shuffle(rest)
    seed = rest[: a.seed_n]
    pool0 = rest[a.seed_n:]
    print(f"test={len(test)} seed={len(seed)} pool={len(pool0)}  epochs={a.epochs} iters={a.iters}")
    print(f"test cases: {test}")

    cfg = C.TrainCfg(epochs=a.epochs, iters_per_epoch=a.iters)
    results = {}
    for strat in ("al", "random"):
        labeled, pool = list(seed), list(pool0)
        curve = []
        for rd in range(a.rounds + 1):
            net = quick_train(labeled, cfg, dev)
            v = validate(net, cfg, test, dev)
            curve.append(dict(round=rd, n_labeled=len(labeled), labeled=sorted(labeled),
                              mean_dice=round(v["mean_dice"], 4),
                              per_class={k: round(x, 4) for k, x in v["dice_per_class"].items()}))
            print(f"[{strat}] round {rd}: n={len(labeled)} meanDice={v['mean_dice']:.4f} "
                  f"{curve[-1]['per_class']}", flush=True)
            if rd == a.rounds or not pool:
                break
            if strat == "al":
                add, sc = pick_al(net, cfg, pool, a.step)
                print(f"       AL 选中: {add}")
            else:
                add = rng.sample(pool, min(a.step, len(pool)))
                print(f"       随机选中: {add}")
            labeled += add
            pool = [c for c in pool if c not in add]
            del net
            if dev.type == "mps":
                torch.mps.empty_cache()
        results[strat] = curve
    out = os.path.join(C.AL_DIR, "simulation.json")
    json.dump(dict(test=test, seed=seed, config=vars(a), curves=results),
              open(out, "w"), indent=1)
    print(f"\n写入 {out}")
    print("\n标注量    AL Dice   Random Dice   差值")
    for i in range(len(results["al"])):
        a_, r_ = results["al"][i], results["random"][i]
        print(f"  {a_['n_labeled']:>3}      {a_['mean_dice']:.4f}    {r_['mean_dice']:.4f}"
              f"      {a_['mean_dice']-r_['mean_dice']:+.4f}")


if __name__ == "__main__":
    main()
