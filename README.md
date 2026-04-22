# Nemotron-Model

Nemotron SSH training and Kaggle-compatible submission workflow for the NVIDIA Nemotron reasoning challenge.

## What is in this repo

- `src/nemotron_model/data_bridge.py`
  - prepares balanced SFT JSONL files from Kaggle `train.csv`
  - keeps the safer final-answer formatting we already validated locally
  - still exposes the earlier Tinker subcommands if we want that path again later
- `scripts/train_trl_kaggle_sim.py`
  - SSH/HPC-side LoRA or warm-start adapter training with `transformers + peft + trl`
  - designed for a Kaggle-like offline setup with local model paths
- `scripts/build_improved_notebook.py`
  - converts the original Kaggle notebook into the improved notebook version
- `scripts/sync_repo_to_hpc.py`
  - uploads this repo to the HFUT SSH machine over SFTP
- `slurm/train_nemotron_sft.sbatch`
  - Slurm training body
- `slurm/submit_train.sh`
  - helper wrapper so partition/account/qos can be passed at submit time

## Current remote status

The HFUT SSH endpoint is reachable and login works:

- host: `210.45.253.131`
- port: `20003`
- user format: `u + student id`

Important findings from the live probe:

- the SSH endpoint is a Slurm entry machine, not a ready-to-train GPU shell
- `srun --partition=8-card --gres=gpu:1 ...` currently fails for this account with:
  - `invalid account or account/partition combination specified`
- login node Python is only `3.6.8`
- outbound container pulls from Docker Hub timed out

So this repo is ready for the SSH route, but the actual full Nemotron training still needs:

1. GPU partition access for the account
2. either a newer Python environment on the cluster, or a working container/image path

## Recommended workflow

### 1. Prepare Kaggle data locally or on the cluster

Do not commit Kaggle competition data into Git.

```bash
python -m nemotron_model.data_bridge prepare \
  --train-csv /path/to/train.csv \
  --output-dir /path/to/prepared \
  --target-samples-per-task 1200 \
  --val-fraction 0.05 \
  --val-min-size-per-task 2
```

This writes:

- `train_sft.jsonl`
- `val_sft.jsonl`
- `dataset_summary.json`

### 2. Run Kaggle-style LoRA training

```bash
python scripts/train_trl_kaggle_sim.py \
  --model-path /path/to/base-model \
  --train-jsonl /path/to/prepared/train_sft.jsonl \
  --val-jsonl /path/to/prepared/val_sft.jsonl \
  --output-dir /path/to/outputs/nemotron-sft \
  --warm-start-adapter /path/to/tong-adapter \
  --max-length 4096 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 1 \
  --learning-rate 5e-5
```

### 3. Submit through Slurm

Edit nothing inside the sbatch body unless you have to. Pass cluster-specific values at submit time:

```bash
REPO_DIR=$HOME/Nemotron-Model \
PARTITION=8-card \
QOS=duzhan \
GPU_COUNT=1 \
CPUS_PER_TASK=16 \
MEMORY=120G \
TIME_LIMIT=24:00:00 \
BASE_MODEL_PATH=/path/to/base-model \
TRAIN_JSONL=/path/to/prepared/train_sft.jsonl \
VAL_JSONL=/path/to/prepared/val_sft.jsonl \
OUTPUT_DIR=$HOME/nemotron-runs/run1 \
WARM_START_ADAPTER=/path/to/tong-adapter \
bash slurm/submit_train.sh
```

If your cluster admin later gives you an account name, also pass:

```bash
ACCOUNT=<your_gpu_account>
```

### 4. Sync this repo to the cluster

```bash
python scripts/sync_repo_to_hpc.py \
  --host 210.45.253.131 \
  --port 20003 \
  --user u2025171971 \
  --remote-dir /home/u2025171971/Nemotron-Model
```

Password can be passed with `--password` or `HPC_PASSWORD`.

## Notes

- This repo intentionally excludes Kaggle raw data, adapters, and checkpoints.
- The current SSH route is prepared and tested up to remote login, file sync, and cluster probing.
- The blocking issue is cluster authorization for the GPU partition, not the training code layout.
