#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <checkpoint_path>" >&2
  exit 2
fi

checkpoint_path="$1"
if [[ "$checkpoint_path" != /* ]]; then
  checkpoint_path="$PWD/$checkpoint_path"
fi
if [[ ! -f "$checkpoint_path" ]]; then
  echo "Checkpoint not found: $checkpoint_path" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python eval.py \
  --load_from "$checkpoint_path" \
  --bs 1 \
  --device_id 0
