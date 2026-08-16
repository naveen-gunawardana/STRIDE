#!/usr/bin/env bash
# v3d: selective broadening (exercise_dependence only). Wait for silver regen, then retrain + eval.
# Batch 16 throughout, so v3d vs v3c isolates the substance/trauma broadening removal.
set -u
cd "C:/Users/navee_xqu8e3o/OneDrive/Documents/programming/AM"
PY=.venv/Scripts/python.exe

echo "[pipe] waiting for selective silver regen ..."
for i in $(seq 1 120); do
  grep -aqiE "\] wrote" run_logs/layer2_silver_v3e.log && break
  sleep 12
done
grep -aqiE "\] wrote" run_logs/layer2_silver_v3e.log || { echo "[pipe] SILVER FAILED"; exit 1; }
echo "[pipe] silver done: $(grep -aoE 'wrote [0-9,]+ rows' run_logs/layer2_silver_v3e.log | tail -1)"

# leak guard
$PY - <<'PY'
import csv; csv.field_size_limit(2**31-1)
sil=set((r.get("text") or "").strip() for r in csv.DictReader(open("data/layer2/silver.csv",encoding="utf-8")))
bad=0
for sp in ("train","dev","test"):
    g=set((r.get("text") or "").strip() for r in csv.DictReader(open(f"data/layer2/gold_{sp}.csv",encoding="utf-8-sig")))
    n=len(g&sil); bad+=n; print(f"  gold_{sp} leaked: {n}")
import sys; sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && { echo "[pipe] LEAK DETECTED - abort"; exit 1; }

echo "[pipe] STAGE 1: silver+gold from DAPT, batch 16 ..."
$PY code/train_layer2.py --silver data/layer2/silver.csv --gold data/layer2/gold_train.csv \
  --batch 16 --epochs 2 --out models/layer2_tags_v3d_s1 > run_logs/train_l2v3d_s1.log 2>&1
grep -aqiE "DONE -> models/layer2_tags_v3d_s1" run_logs/train_l2v3d_s1.log \
  || { echo "[pipe] STAGE 1 FAILED"; tail -5 run_logs/train_l2v3d_s1.log; exit 1; }

echo "[pipe] STAGE 2: gold-only fine-tune, batch 16, 3 epochs ..."
$PY code/train_layer2.py --no-silver --gold data/layer2/gold_train.csv \
  --init models/layer2_tags_v3d_s1 --batch 16 --epochs 3 \
  --out models/layer2_tags_v3d > run_logs/train_l2v3d_s2.log 2>&1
grep -aqiE "DONE -> models/layer2_tags_v3d" run_logs/train_l2v3d_s2.log \
  || { echo "[pipe] STAGE 2 FAILED"; tail -5 run_logs/train_l2v3d_s2.log; exit 1; }

echo "[pipe] EVAL on gold-test ..."
$PY code/eval_layer2.py --gold data/layer2/gold_test.csv --dev data/layer2/gold_dev.csv \
  --model models/layer2_tags_v3d --save-thr models/layer2_tags_v3d/thresholds.json \
  > run_logs/eval_l2v3d.log 2>&1
echo "[pipe] eval exit $?. PIPELINE COMPLETE."
