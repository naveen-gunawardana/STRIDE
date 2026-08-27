#!/usr/bin/env bash
set -e
PY=./.venv/Scripts/python.exe
echo "=== STAGE 1 (silver v3b, broadened rules) ==="
$PY code/train_layer2.py --epochs 2 --gold data/layer2/gold_train.csv \
    --gold-weight 20 --out models/layer2_tags_v3b_s1 > run_logs/train_l2v3b_s1.log 2>&1
echo "=== STAGE 2 (gold fine-tune) ==="
$PY code/train_layer2.py --init models/layer2_tags_v3b_s1 --no-silver \
    --gold data/layer2/gold_train.csv --epochs 18 --lr 1e-5 \
    --out models/layer2_tags_v3b > run_logs/train_l2v3b_s2.log 2>&1
echo "=== EVAL ==="
$PY code/eval_layer2.py --model models/layer2_tags_v3b \
    --gold data/layer2/gold_test.csv --dev data/layer2/gold_dev.csv \
    --save-thr models/layer2_tags_v3b/thresholds.json > run_logs/eval_l2v3b.log 2>&1
tail -25 run_logs/eval_l2v3b.log
echo "=== PIPELINE DONE ==="
