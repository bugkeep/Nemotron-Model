#!/bin/bash
#SBATCH --job-name=nemotron-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO_DIR="${REPO_DIR:?set REPO_DIR}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?set BASE_MODEL_PATH}"
TRAIN_JSONL="${TRAIN_JSONL:?set TRAIN_JSONL}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"

VAL_JSONL="${VAL_JSONL:-}"
WARM_START_ADAPTER="${WARM_START_ADAPTER:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"

echo "Running on $(hostname)"
echo "Repo dir     : ${REPO_DIR}"
echo "Model path   : ${BASE_MODEL_PATH}"
echo "Train JSONL  : ${TRAIN_JSONL}"
echo "Output dir   : ${OUTPUT_DIR}"

cd "${REPO_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" scripts/train_trl_kaggle_sim.py
  --model-path "${BASE_MODEL_PATH}"
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-length "${MAX_LENGTH}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --num-train-epochs "${NUM_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --logging-steps "${LOGGING_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --local-files-only
)

if [[ -n "${VAL_JSONL}" ]]; then
  CMD+=(--val-jsonl "${VAL_JSONL}")
fi

if [[ -n "${WARM_START_ADAPTER}" ]]; then
  CMD+=(--warm-start-adapter "${WARM_START_ADAPTER}")
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"
