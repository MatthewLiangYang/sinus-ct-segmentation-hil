"""Global configuration: paths, label semantics, preprocessing and training hyperparameters."""
from __future__ import annotations
import os, torch
from dataclasses import dataclass, field, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "DATA")      # raw CT volumes
SEG_DIR  = os.path.join(ROOT, "SEG")       # raw manual annotations
WORK     = os.path.join(ROOT, "work")      # all derived artifacts
CACHE    = os.path.join(WORK, "cache")     # preprocessed .npy volumes
RUNS     = os.path.join(WORK, "runs")      # training checkpoints / logs
AL_DIR   = os.path.join(WORK, "al")        # active-learning state and scores
REVIEW   = os.path.join(WORK, "review")    # human-facing review sheets
PRED     = os.path.join(WORK, "pred")      # inference outputs

for _d in (WORK, CACHE, RUNS, AL_DIR, REVIEW, PRED):
    os.makedirs(_d, exist_ok=True)

# ---- Label semantics (no label dictionary shipped with the data; recovered
#      from RAS centroid position + connected-component count + intensity) ----
LABELS = {
    0: "background",
    1: "nasal_cavity",   # single connected component in 29/29 masks; higher HU (turbinates)
    2: "ethmoid",        # honeycomb air cells -> highest component count (2-11)
    3: "maxillary",      # largest volume, always exactly 2 components (bilateral pair)
    4: "sphenoid",       # smallest RAS y = most posterior; sits directly above nasopharynx
    5: "frontal",        # largest RAS y and z = most anterior and superior
}
LABELS_ZH = {
    1: "鼻腔", 2: "筛窦", 3: "上颌窦", 4: "蝶窦", 5: "额窦",
}
NUM_CLASSES = len(LABELS)
FG = list(range(1, NUM_CLASSES))

# Per-structure anatomical priors, used for the plausibility / anomaly score.
# Format: (expected connected-component range, expected volume range in cm^3).
# Ranges come from the 29-mask audit; see README "Anatomical plausibility check".
ANAT_PRIOR = {
    1: ((1, 4),   (10.0, 80.0)),
    2: ((4, 40),  (2.0,  30.0)),
    3: ((1, 4),   (10.0, 110.0)),
    4: ((1, 4),   (1.0,  35.0)),   # sphenoid
    5: ((0, 4),   (0.0,  40.0)),   # frontal may be aplastic/hypoplastic -> lower bound 0
}


@dataclass
class PreprocCfg:
    # 0.8 mm rather than a finer 0.6 mm: a 96^3 patch at 0.6 mm spans only 58 mm,
    # while the whole paranasal sinus complex spans roughly 70-80 mm. A model that
    # cannot see the surrounding landmarks cannot tell ethmoid from sphenoid --
    # this is a *semantic* failure, not a resolution failure. At 0.8 mm the same
    # patch spans 77 mm: identical compute per step, double the context.
    # Sinus cavities are high-contrast air-vs-bone structures, so 0.8 mm is ample
    # for volumetry. Measured effect: validation Dice at epoch 9 went 0.255 -> 0.397.
    spacing: tuple = (0.8, 0.8, 0.8)   # target isotropic resolution (mm)
    clip: tuple = (-1000.0, 1500.0)    # CT intensity window (HU)
    orientation: str = "RAS"


@dataclass
class TrainCfg:
    patch: tuple = (96, 96, 96)
    batch_size: int = 2
    iters_per_epoch: int = 50
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    fg_prob: float = 0.70             # probability a patch centre lands on foreground
    dropout: float = 0.10             # doubles as the MC-dropout uncertainty source
    deep_supervision: bool = True
    amp: bool = False                 # autocast gains little on MPS and is less stable
    num_workers: int = 0              # MPS + fork is flaky; load in the main process
    seed: int = 42
    val_every: int = 10
    sw_overlap: float = 0.5
    # 5.7M-parameter variant. Benchmarked on an M2 Max against the nnU-Net default
    # (32,64,128,256,320) = 16.7M params: 1.88 s/iter vs 2.68 s/iter, and with only
    # 25 training volumes the larger model brings no benefit.
    filters: tuple = (16, 32, 64, 128, 256)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dump(cfg) -> dict:
    return asdict(cfg)
