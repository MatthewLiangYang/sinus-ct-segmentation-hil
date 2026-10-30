#!/usr/bin/env bash
# End-to-end pipeline, from raw data to a usable model.
# Every step is idempotent and can be rerun on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== 1/5 配对 DATA 与 SEG =="
$PY -m sinus_hil.manifest

echo "== 2/5 清点缺失图像（找回即可让数据翻倍）=="
$PY scripts/missing_images.py | head -5

echo "== 3/5 预处理到 0.8mm 各向同性 =="
$PY -m sinus_hil.preprocess --jobs 5

echo "== 4/5 训练 fold 0 (M2 Max 约 3.2 小时) =="
$PY -m sinus_hil.train --fold 0 --tag d49

CKPT=work/runs/d49_fold0/best.pt
echo "== 5/5 标注复核 + 疾病量化 =="
$PY -c "from sinus_hil.review import run; run('$CKPT')"
$PY -m sinus_hil.disease --ckpt "$CKPT"

echo
echo "产物:"
echo "  模型       $CKPT"
echo "  复核工单   work/review/worklist.html"
echo "  疾病指标   work/disease_metrics_model.json"
echo
echo "下一轮主动学习（需要一批未标注 CT）:"
echo "  $PY -c \"from sinus_hil.active_learning import run_round; run_round('$CKPT', '/path/to/未标注CT目录', k=5)\""
