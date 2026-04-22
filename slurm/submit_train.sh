#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?set REPO_DIR}"
PARTITION="${PARTITION:?set PARTITION}"
GPU_COUNT="${GPU_COUNT:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-120G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

ARGS=(
  --partition "${PARTITION}"
  --gres "gpu:${GPU_COUNT}"
  --cpus-per-task "${CPUS_PER_TASK}"
  --mem "${MEMORY}"
  --time "${TIME_LIMIT}"
)

if [[ -n "${ACCOUNT:-}" ]]; then
  ARGS+=(--account "${ACCOUNT}")
fi

if [[ -n "${QOS:-}" ]]; then
  ARGS+=(--qos "${QOS}")
fi

sbatch "${ARGS[@]}" "${REPO_DIR}/slurm/train_nemotron_sft.sbatch"
