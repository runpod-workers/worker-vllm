# /usr/local/bin/start.sh
#!/usr/bin/env bash
set -euo pipefail

# If user didn’t set it explicitly, infer from visible GPUs.
if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    # Fallback to PyTorch if available
    COUNT="$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 0)
except Exception:
    print(0)
PY
)"
  fi

  # Respect CUDA_VISIBLE_DEVICES (both methods above do, since they see only visible GPUs).
  if [[ "${COUNT}" -lt 1 ]]; then
    COUNT=1
  fi
  export TENSOR_PARALLEL_SIZE="${COUNT}"
fi

echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
