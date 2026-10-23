"""Human-facing output: overlay PNG sheets and HTML worklists.

This is the interface layer of the HIL loop -- what an annotator actually opens.
Runtime strings stay in Chinese because that is the annotators' working language.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# Figure titles are in Chinese; without an explicit CJK font matplotlib renders
# every glyph as a tofu box.
_CJK = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",
        "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"]
_have = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [f for f in _CJK if f in _have] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.colors import ListedColormap
from . import config as C

# index 0 is transparent background; 1..5 follow config.LABELS
_PALETTE = np.array([
    [0, 0, 0, 0.0],
    [0.20, 0.60, 1.00, 0.45],   # 1 nasal cavity  blue
    [1.00, 0.75, 0.10, 0.50],   # 2 ethmoid       yellow
    [0.20, 0.85, 0.35, 0.45],   # 3 maxillary     green
    [0.75, 0.45, 1.00, 0.50],   # 4 sphenoid      purple
    [1.00, 0.35, 0.35, 0.50],   # 5 frontal       red
])
CMAP = ListedColormap(_PALETTE)


def _win(ct: np.ndarray, lo=-1000.0, hi=800.0) -> np.ndarray:
    return np.clip((ct.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _slice(a: np.ndarray, i: int, axis: int) -> np.ndarray:
    s = [slice(None)] * 3
    s[axis] = i
    return np.rot90(a[tuple(s)])


def overlay_sheet(ct, pred, out_png, gt=None, unc=None, slices=None,
                  axis=2, title="", ncol=4):
    """Lay several slices out in one PNG.

    Each slice gets one row per panel: CT + prediction, optionally CT + manual
    annotation, optionally an uncertainty heat map. Stacking prediction directly
    above the human annotation is what makes a disagreement obvious at a glance.
    """
    if slices is None:
        from .uncertainty import top_slices
        base = unc if unc is not None else (pred > 0).astype(np.float32)
        slices = top_slices(base, k=ncol * 2, axis=axis)
    rows = 1 + (gt is not None) + (unc is not None)
    nrow = rows * int(np.ceil(len(slices) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow),
                             squeeze=False, facecolor="black")
    for ax in axes.ravel():
        ax.axis("off")
    for j, z in enumerate(slices):
        blk, col = divmod(j, ncol)
        bg = _win(_slice(ct, z, axis))
        panels = [("模型预测", _slice(pred, z, axis), "seg")]
        if gt is not None:
            panels.append(("人工标注", _slice(gt, z, axis), "seg"))
        if unc is not None:
            panels.append(("不确定性", _slice(unc, z, axis), "heat"))
        for r, (name, ov, kind) in enumerate(panels):
            ax = axes[blk * rows + r][col]
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
            if kind == "seg":
                ax.imshow(np.ma.masked_where(ov == 0, ov), cmap=CMAP, vmin=0, vmax=5,
                          interpolation="nearest")
            else:
                ax.imshow(np.ma.masked_where(ov < 0.05, ov), cmap="inferno",
                          vmin=0, vmax=max(0.3, float(ov.max())), alpha=0.6)
            ax.set_title(f"{name} z={z}", color="w", fontsize=8)
            ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                          color=_PALETTE[L][:3], label=f"{L} {C.LABELS_ZH[L]}") for L in C.FG]
    fig.legend(handles=handles, loc="lower center", ncol=5, facecolor="black",
               labelcolor="w", fontsize=9, frameon=False)
    fig.suptitle(title, color="w", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, facecolor="black")
    plt.close(fig)
    return out_png


def worklist_html(rows: list[dict], out_html: str, title="待复核清单"):
    """Render a one-page HTML worklist, sorted by priority, linking to slice sheets."""
    head = """<meta charset="utf-8"><style>
body{font:14px/1.5 -apple-system,PingFang SC,sans-serif;background:#111;color:#eee;margin:24px}
table{border-collapse:collapse;width:100%}th,td{padding:7px 10px;border-bottom:1px solid #333;text-align:left}
th{background:#1c1c1c;position:sticky;top:0}tr:hover{background:#1a1a1a}
.b{display:inline-block;height:8px;background:#4a9eff;border-radius:4px;vertical-align:middle}
a{color:#6cf}.hi{color:#ff6b6b;font-weight:600}.ok{color:#51cf66}
</style>"""
    mx = max([r.get("priority", 0) for r in rows] + [1e-6])
    tr = []
    for i, r in enumerate(rows, 1):
        p = r.get("priority", 0)
        bar = f'<span class="b" style="width:{int(120*p/mx)}px"></span> {p:.3f}'
        link = f'<a href="{os.path.relpath(r["png"], os.path.dirname(out_html))}">查看层面</a>' \
            if r.get("png") else ""
        cls = "hi" if i <= max(1, len(rows) // 4) else ""
        tr.append(f"<tr><td>{i}</td><td class='{cls}'>{r['case_id']}</td><td>{bar}</td>"
                  f"<td>{r.get('reason','')}</td><td>{link}</td></tr>")
    html = f"""{head}<h2>{title}</h2>
<p>共 {len(rows)} 例，按"标注收益/可疑程度"降序。红色为建议优先处理的前 25%。</p>
<table><tr><th>#</th><th>病例</th><th>优先级</th><th>依据</th><th></th></tr>
{''.join(tr)}</table>"""
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w") as f:
        f.write(html)
    return out_html
