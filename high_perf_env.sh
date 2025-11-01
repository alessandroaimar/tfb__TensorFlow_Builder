#!/usr/bin/env bash
#
# Source this helper to opt into aggressive runtime tuning for TensorFlow and
# Numba. It only exports performance-centric environment variables; no runtime
# housekeeping or filesystem edits are performed.
#
#   $ source high_perf_env.sh
#
# Existing manual exports always win so the script can be layered on demand.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
fi

# --------------------------------------------------------------------------- #
# Helper utility                                                              #
# --------------------------------------------------------------------------- #

set_env_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name+x}" || "${!name}" == "" ]]; then
    printf -v "${name}" '%s' "${value}"
    export "${name}"
  fi
}

# --------------------------------------------------------------------------- #
# Threading configuration                                                     #
# --------------------------------------------------------------------------- #

cpu_threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
if ! [[ "${cpu_threads}" =~ ^[0-9]+$ ]] || (( cpu_threads < 1 )); then
  cpu_threads=1
fi

interop_threads_default=1
if (( cpu_threads >= 32 )); then
  interop_threads_default=8
elif (( cpu_threads >= 16 )); then
  interop_threads_default=4
elif (( cpu_threads >= 8 )); then
  interop_threads_default=2
fi

export TF_CPP_MIN_LOG_LEVEL=2
export ABSL_MIN_LOG_LEVEL=2

set_env_default TF_NUM_INTRAOP_THREADS "${cpu_threads}"
set_env_default TF_NUM_INTEROP_THREADS "${interop_threads_default}"
set_env_default OMP_NUM_THREADS "${cpu_threads}"
set_env_default OMP_PROC_BIND "close"
set_env_default OMP_WAIT_POLICY "active"
set_env_default KMP_AFFINITY "granularity=fine,compact,1,0"
set_env_default KMP_BLOCKTIME "1"

# Keep BLAS-backed libraries single-threaded to avoid oversubscription when
# TensorFlow / Numba already saturate CPU threads.
set_env_default OPENBLAS_NUM_THREADS "1"
set_env_default MKL_NUM_THREADS "1"
set_env_default MKL_DYNAMIC "0"
set_env_default VECLIB_MAXIMUM_THREADS "1"
set_env_default BLIS_NUM_THREADS "1"
set_env_default NUMEXPR_NUM_THREADS "1"
set_env_default NUMEXPR_MAX_THREADS "1"

# --------------------------------------------------------------------------- #
# TensorFlow runtime knobs                                                    #
# --------------------------------------------------------------------------- #

set_env_default TF_ENABLE_XLA "1"
# XLA/Autotune is controlled below through XLA_FLAGS (not TF_XLA_FLAGS).
set_env_default TF_USE_CUBLASLT "1"
set_env_default TF_ENABLE_AUTO_MIXED_PRECISION "1"
set_env_default TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE "1"
set_env_default TF_GPU_THREAD_MODE "gpu_private"
set_env_default TF_GPU_ALLOCATOR "cuda_malloc_async"
set_env_default TF_CUDA_ALLOCATOR "cuda_malloc_async"

# --------------------------------------------------------------------------- #
# XLA optimizer/codegen flags (XLA_FLAGS only)                                #
# --------------------------------------------------------------------------- #
default_xla_flags=(
  "--xla_cpu_enable_fast_math"
  "--xla_cpu_fast_math_honor_infs=false"
  "--xla_cpu_fast_math_honor_nans=false"
  "--xla_cpu_fast_math_honor_division=false"
#   "--xla_gpu_enable_async_collectives=true"
#   "--xla_gpu_enable_cudnn_frontend=true"
  "--xla_gpu_enable_latency_hiding_scheduler=true"
  "--xla_gpu_enable_triton_gemm=true"
  "--xla_gpu_autotune_level=4"
#   "--xla_gpu_persistent_cache_dir=${XDG_CACHE_HOME:-$HOME/.cache}/xla"
)

if [[ -z "${XLA_FLAGS:-}" ]]; then
  export XLA_FLAGS="${default_xla_flags[*]}"
else
  for f in "${default_xla_flags[@]}"; do
    [[ "${XLA_FLAGS}" == *"${f}"* ]] || XLA_FLAGS+=" ${f}"
  done
  export XLA_FLAGS
fi

# Leave TF_XLA_FLAGS unset unless you need --tf_xla_* options.

# --------------------------------------------------------------------------- #
# CUDA / NCCL tuning                                                          #
# --------------------------------------------------------------------------- #

set_env_default CUDA_CACHE_MAXSIZE "2147483648"
set_env_default CUDA_DEVICE_MAX_CONNECTIONS "32"
set_env_default NCCL_LAUNCH_MODE "GROUP"

# --------------------------------------------------------------------------- #
# Numba accelerators                                                          #
# --------------------------------------------------------------------------- #

set_env_default NUMBA_THREADING_LAYER "tbb"
# set_env_default NUMBA_CPU_NAME "znver2"
# Replace 'native' with the actual micro-architecture LLVM expects.
export NUMBA_CPU_NAME="$(
python3 - <<'PY'
import llvmlite.binding as ll
print(ll.get_host_cpu_name())
PY
)"

# Optional: also pin exact CPU features for reproducibility.
export NUMBA_CPU_FEATURES="$(
python3 - <<'PY'
import llvmlite.binding as ll
print(ll.get_host_cpu_features().flatten())
PY
)"

set_env_default NUMBA_NUM_THREADS "${cpu_threads}"
# set_env_default NUMBA_CACHE "1"
# set_env_default NUMBA_CUDA_USE_NVIDIA_BINDING "1"

echo "High-performance TensorFlow + Numba environment activated."

return 0 2>/dev/null || exit 0
