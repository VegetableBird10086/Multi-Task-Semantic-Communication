#!/usr/bin/env bash

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  distributed_main.py \
  --datasets VQAv2 \
  --bs 32 \
  --epochs 30 \
  --save_path checkpoints \
  "$@"
