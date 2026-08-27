#!/usr/bin/env bash
set -e
PY=./.venv/Scripts/python.exe
echo "=== SILVER (excludes the expanded gold) ==="
$PY code/layer2_silver.py > run_logs/layer2_silver_v6.log 2>&1
tail -20 run_logs/layer2_silver_v6.log
echo "=== STAGE 1 ==="
$PY code/train_layer2.py --epochs 2 --gold data/layer2/gold_train.csv --gold-weight 20 \
    --out models/layer2_tags_v6_s1 > run_logs/train_l2v6_s1.log 2>&1
echo "=== STAGE 2 ==="
$PY code/train_layer2.py --init models/layer2_tags_v6_s1 --no-silver \
    --gold data/layer2/gold_train.csv --epochs 18 --lr 1e-5 \
    --out models/layer2_tags_v6 > run_logs/train_l2v6_s2.log 2>&1
echo "=== EVAL on the expanded 696-row test set ==="
$PY code/eval_layer2.py --model models/layer2_tags_v6 --gold data/layer2/gold_test.csv \
    --dev data/layer2/gold_dev.csv --save-thr models/layer2_tags_v6/thresholds.json \
    > run_logs/eval_l2v6.log 2>&1
grep -vE "Loading weights|it/s\]$" run_logs/eval_l2v6.log | tail -26
echo "=== DONE ==="
