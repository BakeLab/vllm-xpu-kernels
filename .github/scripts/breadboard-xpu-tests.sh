#!/usr/bin/env bash
set -eo pipefail
source /opt/intel/oneapi/setvars.sh
set -u

# Dockerfile.xpu supplies /opt/venv, the compiler and GPU runtime. Installing
# requirements here keeps the build and test interpreter identical.
uv pip install -r requirements.txt
/opt/venv/bin/python - <<'PY'
import torch

assert torch.xpu.is_available(), "The allocated XPU is unavailable in the container"
assert torch.xpu.device_count() == 1, "This lane requires one isolated GPU"
name = torch.xpu.get_device_name(0)
assert any(model in name for model in ("B70", "B60", "B580", "B570")), (
    f"This lane runs the Battlemage test suite, got {name}"
)
print(f"Testing {name}", flush=True)
PY

# Native compiles use about 8 GiB per process. Respect the sandbox allocation
# instead of the persistent PVC runner's hard-coded 128 parallel builds.
memory_jobs=$(awk '/MemTotal:/ {print int($2 / 8388608)}' /proc/meminfo)
export MAX_JOBS=$(nproc)
if (( memory_jobs < MAX_JOBS )); then MAX_JOBS=$memory_jobs; fi
if (( MAX_JOBS < 1 )); then MAX_JOBS=1; fi

VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf \
  uv build --no-build-isolation --wheel --out-dir dist-default
# Set both explicitly: the previous build's CMake cache retains its presets.
VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_full.conf \
VLLM_PAGED_DECODE_CONFIG=paged_decode_full.conf \
  uv build --no-build-isolation --wheel --out-dir dist
wheels=(dist/*.whl)
test "${#wheels[@]}" -eq 1
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="${wheels[0]}" \
  uv pip install --no-build-isolation -e .

# Preserve the existing BMG suite and its documented exclusions. The VM has
# one GPU, so do not copy the persistent runner's ZE_AFFINITY_MASK=0,1.
export ZE_AFFINITY_MASK=0
/opt/venv/bin/python -m pytest -v -s tests/ \
  --ignore=tests/test_lora_ops.py --ignore=tests/test_fp8_quant.py \
  --ignore=tests/test_moe_align_block_size.py --ignore=tests/test_moe_lora_align_sum.py \
  --ignore=tests/test_topk_per_row.py --ignore=tests/test_fp8_gemm_onednn.py \
  --deselect=tests/test_cache.py::test_swap_blocks
/opt/venv/bin/python -m pytest -v -s tests/test_fp8_gemm_onednn.py
/opt/venv/bin/python -m pytest -v -s tests/test_lora_ops.py
VLLM_XPU_FORCE_XE_DEFAULT_KERNEL=1 /opt/venv/bin/python -m pytest -v -s \
  tests/fused_moe/test_grouped_gemm.py::test_grouped_gemm
