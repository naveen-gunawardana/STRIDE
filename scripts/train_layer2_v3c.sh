#!/usr/bin/env bash
# v3c: same broadened rules/silver as v3b, but batch 16 (v3b used 32 and regressed across tags).
# Isolates the batch-size variable: stage-1 (silver+gold) -> stage-2 (gold-only) -> eval.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

echo "[pipe] STAGE 1: silver+gold from DAPT, batch 16 ..."
$PY code/train_layer2.py --silver data/layer2/silver.csv --gold data/layer2/gold_train.csv \
  --batch 16 --epochs 2 --out models/layer2_tags_v3c_s1 > run_logs/train_l2v3c_s1.log 2>&1
grep -aqiE "DONE -> models/layer2_tags_v3c_s1" run_logs/train_l2v3c_s1.log \
  || { echo "[pipe] STAGE 1 FAILED"; tail -5 run_logs/train_l2v3c_s1.log; exit 1; }
echo "[pipe] stage-1 done."

echo "[pipe] STAGE 2: gold-only fine-tune from stage-1, batch 16, 3 epochs ..."
$PY code/train_layer2.py --no-silver --gold data/layer2/gold_train.csv \
  --init models/layer2_tags_v3c_s1 --batch 16 --epochs 3 \
  --out models/layer2_tags_v3c > run_logs/train_l2v3c_s2.log 2>&1
grep -aqiE "DONE -> models/layer2_tags_v3c" run_logs/train_l2v3c_s2.log \
  || { echo "[pipe] STAGE 2 FAILED"; tail -5 run_logs/train_l2v3c_s2.log; exit 1; }
echo "[pipe] stage-2 done."

echo "[pipe] EVAL on gold-test ..."
$PY code/eval_layer2.py --gold data/layer2/gold_test.csv --dev data/layer2/gold_dev.csv \
  --model models/layer2_tags_v3c --save-thr models/layer2_tags_v3c/thresholds.json \
  > run_logs/eval_l2v3c.log 2>&1
echo "[pipe] eval exit $?. PIPELINE COMPLETE."
