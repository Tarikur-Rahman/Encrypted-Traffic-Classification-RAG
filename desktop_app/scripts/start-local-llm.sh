#!/usr/bin/env bash
# User-supplied validated launch settings; all persistent output is app-local.
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
project_dir="$(cd -- "$app_dir/.." && pwd)"
conda_root="${PG_CONDA_ROOT:-$HOME/miniforge3}"
cache="$app_dir/runtime/local-llm"
mkdir -p "$cache" "$cache/tmp" "$cache/home" "$cache/config" "$cache/torch" "$cache/hf" "$cache/triton" "$cache/cuda" "$cache/inductor"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$cache/tmp" TMP="$cache/tmp" TEMP="$cache/tmp"
export XDG_CACHE_HOME="$cache" XDG_CONFIG_HOME="$cache/config" HF_HOME="$cache/hf" TORCH_HOME="$cache/torch"
export VLLM_CACHE_ROOT="$cache/vllm" VLLM_CONFIG_ROOT="$cache/config/vllm"
# Linux abstract IPC sockets are kernel-only: no unsupported socket files on
# /mnt/c and no temporary files outside desktop_app. vLLM appends unique UUIDs.
export VLLM_RPC_BASE_PATH="@provenanceguard-$UID"
export TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda" TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
export NUMBA_CACHE_DIR="$cache/numba" FLASHINFER_WORKSPACE_BASE="$cache/flashinfer" OUTLINES_CACHE_DIR="$cache/outlines"
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0
source "$conda_root/etc/profile.d/conda.sh"
conda activate rag-llm
cd -- "$app_dir"
# The V5 contract uses a 5120-token context envelope. No model or decoding
# policy is changed by this desktop launcher.
exec python -B "$app_dir/scripts/start_local_llm.py" "$project_dir" >> "$cache/server.log" 2>&1
