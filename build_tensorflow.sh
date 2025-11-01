#!/usr/bin/env bash

# TensorFlow from-source build orchestrator tailored for this machine.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
SRC_DIR="${ROOT_DIR}/sources"
DIST_DIR="${ROOT_DIR}/dist"
LOG_DIR="${ROOT_DIR}/logs"
VENV_DIR="${ROOT_DIR}/.tf-build-venv"
TF_REPO="${SRC_DIR}/tensorflow"
BAZELISK_BIN="${TOOLS_DIR}/bazelisk"
BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
GITHUB_API_RELEASE_URL="https://api.github.com/repos/tensorflow/tensorflow/releases/latest"
PROFILE_DIR="${ROOT_DIR}/profiles"
CLANG_PROFILE_LIB_DIR=""
declare -ag LAST_BUILT_WHEELS=()

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*"
}

ensure_directories() {
  mkdir -p "${TOOLS_DIR}" "${SRC_DIR}" "${DIST_DIR}" "${LOG_DIR}" "${ROOT_DIR}/runtime-tests"
}

ensure_system_prereqs() {
  local required_commands=(
    bash
    git
    make
    gcc
    g++
    python3
    wget
    tar
    unzip
    zip
  )

  local missing=()
  for cmd in "${required_commands[@]}"; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      missing+=("${cmd}")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "Missing required system commands: ${missing[*]}"
    log "Install the missing tools (e.g. via your package manager) and re-run the script."
    exit 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    local apt_deps=(build-essential pkg-config zlib1g-dev libffi-dev libssl-dev)
    local apt_missing=()
    for pkg in "${apt_deps[@]}"; do
      if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
        apt_missing+=("${pkg}")
      fi
    done
    if [[ "${#apt_missing[@]}" -gt 0 ]]; then
      if sudo -n true >/dev/null 2>&1; then
        log "Installing supplemental system libraries via apt: ${apt_missing[*]}"
        sudo apt-get update
        sudo apt-get install -y "${apt_missing[@]}"
      else
        log "Supplemental libraries missing (${apt_missing[*]}). Install them manually if build fails; continuing without automatic apt install."
      fi
    fi
  fi
}

check_perf_dependencies() {
  local missing=0

  if ! command -v clang >/dev/null 2>&1 || ! command -v clang++ >/dev/null 2>&1; then
    missing=1
    log "ERROR: Clang toolchain not found. Install it (e.g. 'sudo apt install clang') so the optimized build can use Clang/LLD."
  fi

  if ! command -v ld.lld >/dev/null 2>&1; then
    missing=1
    log "ERROR: LLD linker not found. Install it (e.g. 'sudo apt install lld') to enable ThinLTO with Clang."
  fi

  if command -v clang >/dev/null 2>&1; then
    CLANG_PROFILE_LIB_DIR="$(clang -print-resource-dir 2>/dev/null)/lib/linux"
  fi
  if [[ -z "${CLANG_PROFILE_LIB_DIR}" || ! -d "${CLANG_PROFILE_LIB_DIR}" ]]; then
    missing=1
    log "ERROR: Unable to locate Clang runtime libraries (expected profile libs under 'clang -print-resource-dir'). Ensure clang and llvm-dev are installed."
  fi

  if ! command -v ld.gold >/dev/null 2>&1; then
    log "WARNING: GNU gold linker not detected. Install 'binutils-gold' if you want a fallback non-LLD ThinLTO linker."
  fi

  if ! command -v llvm-profdata >/dev/null 2>&1; then
    missing=1
    log "ERROR: llvm-profdata is missing. Install it (e.g. 'sudo apt install llvm-dev') to merge PGO profiles."
  fi

  if [[ "${missing}" -ne 0 ]]; then
    log "High-performance TensorFlow build cannot proceed until the above components are installed."
    exit 1
  fi
}

reset_cuda_environment() {
  local cuda_vars=(CUDA_HOME CUDA_TOOLKIT_PATH CUDNN_INSTALL_PATH TF_SYSTEM_NCCL TF_CUDA_PATHS)
  for var in "${cuda_vars[@]}"; do
    [[ -n "${!var-}" ]] && { log "Unsetting ${var}=${!var-}"; unset "${var}"; }
  done
  # Remove any /cuda/bin entries from PATH (prevents configure from seeing system nvcc)
  PATH="$(awk -v RS=: -v ORS=: '$0!~/\/cuda\/bin($|\/)/' <<<"${PATH}")"
  PATH="${PATH%:}"
  export PATH
}

prepare_cuda_toolchain() {
  reset_cuda_environment
  export TF_NEED_CUDA=1
  export TF_NEED_ROCM="${TF_NEED_ROCM:-0}"

  local compute_caps="${TF_CUDA_COMPUTE_CAPABILITIES}"
  export TF_CUDA_COMPUTE_CAPABILITIES="${compute_caps}"
  export HERMETIC_CUDA_COMPUTE_CAPABILITIES="${HERMETIC_CUDA_COMPUTE_CAPABILITIES:-${compute_caps}}"

  export TF_CUDA_CLANG=1

  # Keep TF’s hermetic CUDA fully self‑contained.
  export USE_CUDA_TAR_ARCHIVE_FILES="${USE_CUDA_TAR_ARCHIVE_FILES:-1}"

  # Ensure hermetic Python in external repos, independent of host venv.
  export HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}"
}


download_file() {
  local url="$1"
  local destination="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${destination}"
  else
    wget -qO "${destination}" "${url}"
  fi
}

ensure_bazelisk() {
  if [[ -x "${BAZELISK_BIN}" ]]; then
    ln -sf "${BAZELISK_BIN}" "${TOOLS_DIR}/bazel"
    return
  fi

  log "Fetching Bazelisk from ${BAZELISK_URL}"
  download_file "${BAZELISK_URL}" "${BAZELISK_BIN}"
  chmod +x "${BAZELISK_BIN}"
  ln -sf "${BAZELISK_BIN}" "${TOOLS_DIR}/bazel"
}

ensure_local_jdk() {
  if command -v javac >/dev/null 2>&1; then
    local javac_path
    javac_path="$(command -v javac)"
    local java_home_candidate="${javac_path%/bin/javac}"
    export JAVA_HOME="${JAVA_HOME:-${java_home_candidate}}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
    log "Using existing system JDK at ${JAVA_HOME}"
    return
  fi

  local existing_local
  existing_local="$(find "${TOOLS_DIR}" -maxdepth 1 -type d -name "jdk-17*" -print -quit || true)"
  if [[ -n "${existing_local}" ]]; then
    export JAVA_HOME="${existing_local}"
    export PATH="${JAVA_HOME}/bin:${PATH}"
    log "Using previously downloaded JDK at ${JAVA_HOME}"
    return
  fi

  log "Downloading local Temurin JDK 17 toolchain"
  local jdk_info jdk_url jdk_filename
  jdk_info="$(
    python3 - <<'PY'
import json
import sys
import urllib.request

API_URL = "https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&heap_size=normal&image_type=jdk&jvm_impl=hotspot&os=linux"
req = urllib.request.Request(API_URL, headers={"User-Agent": "CodexTensorFlowBuilder/1.0"})
with urllib.request.urlopen(req) as response:
    data = json.load(response)

for asset in data:
    pkg = asset.get("binary", {}).get("package", {})
    link = pkg.get("link")
    name = pkg.get("name")
    if link and name and name.endswith(".tar.gz"):
        print(link)
        print(name)
        sys.exit(0)

sys.exit(1)
PY
  )"

  jdk_url="$(printf '%s\n' "${jdk_info}" | head -n 1)"
  jdk_filename="$(printf '%s\n' "${jdk_info}" | tail -n 1)"

  if [[ -z "${jdk_url}" || -z "${jdk_filename}" ]]; then
    log "Failed to resolve JDK download location."
    exit 1
  fi

  local tarball="${TOOLS_DIR}/${jdk_filename}"
  if [[ ! -f "${tarball}" ]]; then
    log "Fetching JDK archive ${jdk_filename}"
    download_file "${jdk_url}" "${tarball}"
  fi

  shopt -s nullglob
  local candidates=("${TOOLS_DIR}"/jdk-17*)
  shopt -u nullglob

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    log "Extracting JDK to ${TOOLS_DIR}"
    tar -xf "${tarball}" -C "${TOOLS_DIR}"
    shopt -s nullglob
    candidates=("${TOOLS_DIR}"/jdk-17*)
    shopt -u nullglob
  fi

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    log "Failed to locate extracted JDK directory."
    exit 1
  fi

  local jdk_home="${candidates[0]}"
  export JAVA_HOME="${jdk_home}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
  log "Using locally provisioned JDK at ${JAVA_HOME}"
}

ensure_python_venv() {
  if ! python3 -m venv --help >/dev/null 2>&1; then
    log "Python venv module unavailable. Install python3-venv via your package manager and re-run."
    exit 1
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating Python build virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  log "Ensuring Python build dependencies are up to date"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --upgrade numpy packaging absl-py keras_preprocessing opt_einsum
}

detect_cuda_compute_capabilities() {
  local caps=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d $'\r\"' | sed '/^$/d')"
  fi
  if [[ -z "${caps}" ]]; then
    caps="$("${VENV_DIR}/bin/python" - <<'PY' || true
import tensorflow as tf
s=set()
try:
  for d in tf.config.list_physical_devices('GPU'):
    det=tf.config.experimental.get_device_details(d)
    cc=det.get('compute_capability')
    if isinstance(cc, tuple) and len(cc)==2:
      s.add(f"{cc[0]}.{cc[1]}")
except Exception:
  pass
print(",".join(sorted(s,key=lambda x:tuple(map(int,x.split("."))))) )
PY
)"
  fi
  if [[ -z "${caps}" ]]; then
    caps="8.0,8.6,8.9,9.0"
    log "WARNING: compute capability autodetect failed; using default ${caps}"
  else
    caps="$(printf '%s\n' "${caps}" | tr ',' '\n' | sed 's/^[ \t]*//;s/[ \t]*$//' | awk -F. 'NF==2{print $1 "." $2}' | sort -t. -k1,1n -k2,2n | uniq | paste -sd, -)"
    log "Detected compute capabilities: ${caps}"
  fi
  export TF_CUDA_COMPUTE_CAPABILITIES="${caps}"
  export HERMETIC_CUDA_COMPUTE_CAPABILITIES="${caps}"
}

fetch_latest_tag() {
  local tag
  tag="$(
    python3 - <<'PY'
import json
import sys
import urllib.request

req = urllib.request.Request(
    "https://api.github.com/repos/tensorflow/tensorflow/releases/latest",
    headers={"User-Agent": "CodexTensorFlowBuilder/1.0"}
)
with urllib.request.urlopen(req) as response:
    data = json.load(response)

print(data.get("tag_name", ""))
PY
  )"
  tag="$(printf '%s' "${tag}" | tr -d '\r')"
  if [[ -z "${tag}" || "${tag}" == "null" ]]; then
    log "Failed to resolve latest tag from GitHub API."
    exit 1
  fi
  echo "${tag}"
}

sync_tensorflow_source() {
  local tag="$1"

  if [[ -d "${TF_REPO}/.git" ]]; then
    log "Updating existing TensorFlow repository"
    git -C "${TF_REPO}" fetch --tags origin
  else
    log "Cloning TensorFlow repository"
    git clone --depth=1 https://github.com/tensorflow/tensorflow.git "${TF_REPO}"
  fi

  log "Checking out ${tag}"
  git -C "${TF_REPO}" fetch --tags origin "${tag}"
  git -C "${TF_REPO}" checkout -f "${tag}"
  git -C "${TF_REPO}" submodule update --init --recursive
}



bazel_full_clean() {
  log "Cleaning Bazel build outputs"
  "${BAZELISK_BIN}" shutdown >/dev/null 2>&1 || true
  "${BAZELISK_BIN}" clean --expunge >/dev/null 2>&1 || true
}

run_profile_workload() {
  local profile_dir="$1"
  local wheel_path="$2"

  log "Running profiling workload using ${wheel_path}"
  source "${VENV_DIR}/bin/activate"
  pip install --quiet --upgrade --force-reinstall "${wheel_path}"

  rm -f "${profile_dir}"/*.profraw
  LLVM_PROFILE_FILE="${profile_dir}/tensorflow-%p.profraw" \
    TF_CPP_MIN_LOG_LEVEL=1 \
    python - <<'PY'
import tensorflow as tf
import numpy as np

tf.random.set_seed(0)
np.random.seed(0)

inputs = tf.keras.Input(shape=(128,))
x = tf.keras.layers.Dense(256, activation="relu")(inputs)
x = tf.keras.layers.Dense(256, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse")

x_data = np.random.rand(4096, 128).astype(np.float32)
y_data = np.random.rand(4096, 1).astype(np.float32)

model.fit(x_data, y_data, epochs=2, batch_size=128, verbose=0)
model.predict(x_data[:256], verbose=0)
PY

  if command -v deactivate >/dev/null 2>&1; then
    deactivate
  fi
}

merge_profile_data() {
  local profile_dir="$1"
  local profdata="${profile_dir}/tensorflow.profdata"
  local profraws=("${profile_dir}/"*.profraw)

  if [[ ! -e "${profraws[0]}" ]]; then
    log "ERROR: No profile data collected in ${profile_dir}"
    exit 1
  fi

  log "Merging profile data into ${profdata}"
  rm -f "${profdata}"
  llvm-profdata merge --output="${profdata}" "${profile_dir}/"*.profraw
}

build_profiled_tensorflow() {
  rm -rf "${PROFILE_DIR}"
  mkdir -p "${PROFILE_DIR}"

  bazel_full_clean
  build_tensorflow_wheel profile_generate "${PROFILE_DIR}"
  local instrumentation_wheel=""
  if [[ "${#LAST_BUILT_WHEELS[@]}" -gt 0 ]]; then
    instrumentation_wheel="${LAST_BUILT_WHEELS[0]}"
  else
    log "ERROR: Instrumented wheel not found after profile_generate build."
    exit 1
  fi

  run_profile_workload "${PROFILE_DIR}" "${instrumentation_wheel}"
  merge_profile_data "${PROFILE_DIR}"

  bazel_full_clean
  build_tensorflow_wheel profile_use "${PROFILE_DIR}"
}

run_tensorflow_runtime_check() {
  local wheel_path="$1"
  local mode="$2"
  local use_high_perf="$3"
  local venv_dir="${ROOT_DIR}/runtime-tests/${mode}"
  local log_file="${LOG_DIR}/runtime-test-${mode}.log"

  log "Validating ${wheel_path} (${mode})"
  rm -rf "${venv_dir}"
  python3 -m venv "${venv_dir}"

  (
    set -euo pipefail
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
    python -m pip install --quiet --upgrade pip setuptools wheel numpy
    python -m pip install --quiet --upgrade "${wheel_path}"
    if [[ "${use_high_perf}" == "1" ]]; then
      # shellcheck disable=SC1090
      source "${ROOT_DIR}/activate_high_perf.sh"
    fi
    python - <<'PY'
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_logical_devices())

def run_matmul(device_name: str) -> float:
    size = 2048
    with tf.device(device_name):
        a = tf.random.uniform((size, size), dtype=tf.float32)
        b = tf.random.uniform((size, size), dtype=tf.float32)
        c = tf.matmul(a, b)
        return float(tf.reduce_sum(c).numpy())

cpu_sum = run_matmul("/CPU:0")
print(f"CPU matmul checksum: {cpu_sum:.6f}")

physical_gpus = tf.config.list_physical_devices("GPU")
if not physical_gpus:
    print("WARNING: No GPU devices detected by TensorFlow runtime; skipping GPU matmul.")
else:
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"WARNING: Unable to enable memory growth on {gpu.name}: {exc}")

    gpu_sum = run_matmul("/GPU:0")
    print(f"GPU matmul checksum: {gpu_sum:.6f}")
PY
  ) >"${log_file}" 2>&1

  if grep -Ei '(warning|error)' "${log_file}" > "${log_file}.diag"; then
    log "Diagnostics detected during ${mode} validation; inspect ${log_file}"
    sed 's/^/  /' "${log_file}.diag"
  else
    rm -f "${log_file}.diag"
    local mode_pretty="${mode^}"
    log "${mode_pretty} runtime check completed without warnings (log: ${log_file})"
  fi
}

validate_tensorflow_runtime() {
  local wheel_path="$1"
  run_tensorflow_runtime_check "${wheel_path}" "baseline" "0"
  run_tensorflow_runtime_check "${wheel_path}" "high-perf" "1"
}

configure_tensorflow() {
  source "${VENV_DIR}/bin/activate"

  log "Configuring TensorFlow build (JAVA_HOME=${JAVA_HOME:-unknown})"

  pushd "${TF_REPO}" >/dev/null
  export PYTHON_BIN_PATH="${VENV_DIR}/bin/python"
  export PYTHON_LIB_PATH="$("${PYTHON_BIN_PATH}" -c 'import site; print(site.getsitepackages()[0])')"
  export CC_OPT_FLAGS="-march=native -mtune=native -O3 -fomit-frame-pointer -fno-semantic-interposition"
  export TF_NEED_CUDA="${TF_NEED_CUDA:-1}"
  export TF_NEED_ROCM="${TF_NEED_ROCM:-0}"
  export TF_CUDA_CLANG=1
  export TF_NEED_MPI=0
  export TF_NEED_TENSORRT=0
  export TF_ENABLE_XLA=1
  export TF_SET_ANDROID_WORKSPACE=0
  export TF_CONFIGURE_IOS=0
  export TF_ENABLE_ONEDNN_OPTS=1
  export TF_IGNORE_MAX_BAZEL_VERSION=1

  if [[ "${TF_CUDA_CLANG:-0}" == "1" ]]; then
    export CLANG_CUDA_COMPILER_PATH="${CLANG_CUDA_COMPILER_PATH:-$(command -v clang)}"
    unset GCC_HOST_COMPILER_PATH
  else
    if command -v gcc >/dev/null 2>&1; then
      export GCC_HOST_COMPILER_PATH="$(command -v gcc)"
    fi
  fi

  if [[ "${TF_NEED_CUDA}" == "1" ]]; then
    reset_cuda_environment
    export TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES}"
  fi

  local configure_log="${LOG_DIR}/configure.log"
  # Prevent host libs from leaking into config detection.
  local _saved_ld="${LD_LIBRARY_PATH-}"
  local _saved_preload="${LD_PRELOAD-}"
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD

  : > "${configure_log}"
  set +e
  yes "" | "${PYTHON_BIN_PATH}" configure.py 2>&1 | tee "${configure_log}"
  local pipe_status=("${PIPESTATUS[@]}")
  set -e

  # Restore host environment.
  if [[ -n "${_saved_ld}" ]]; then export LD_LIBRARY_PATH="${_saved_ld}"; else unset LD_LIBRARY_PATH; fi
  if [[ -n "${_saved_preload}" ]]; then export LD_PRELOAD="${_saved_preload}"; else unset LD_PRELOAD; fi

  local configure_rc="${pipe_status[1]:-1}"
  if [[ "${configure_rc}" -ne 0 ]]; then
    log "TensorFlow configure script failed (exit code ${configure_rc})."
    exit "${configure_rc}"
  fi

  if [[ "${TF_NEED_CUDA}" == "1" ]]; then
    if grep -qiE "nvcc.*(not found|missing)" "${configure_log}"; then
      log "Confirmed: configure will fetch the hermetic CUDA toolchain."
    else
      log "WARNING: configure output did not indicate hermetic CUDA fetch; verify no system CUDA leaked in PATH."
    fi
  fi

  popd >/dev/null
}


build_tensorflow_wheel() {
  source "${VENV_DIR}/bin/activate"

  pushd "${TF_REPO}" >/dev/null
  reset_cuda_environment
  unset CC CXX

  # Make actions hermetic: block host library interposition that crashes nvprune.
  local _saved_ld="${LD_LIBRARY_PATH-}"
  local _saved_preload="${LD_PRELOAD-}"
  unset LD_LIBRARY_PATH
  unset LD_PRELOAD

  local mode="${1:-release}"
  local profile_dir="${2:-}"

  local bazel="${BAZELISK_BIN}"
  local jobs
  jobs="$(nproc)"
  local build_opts=(
    "--config=opt" "-c" "opt" "--config=monolithic" "--strip=always"
    "--jobs=${jobs}" "--verbose_failures"
    "--repo_env=ML_WHEEL_TYPE=release"
    "--define=build_with_bfloat16=true"

    # High‑performance host code; no -ffast-math to satisfy XLA.
    "--copt=-march=native" "--copt=-mtune=native" "--copt=-O3"
    "--copt=-fomit-frame-pointer" "--copt=-fno-semantic-interposition"
    "--copt=-fno-plt" "--copt=-ffunction-sections" "--copt=-fdata-sections"
    "--copt=-fno-math-errno" "--copt=-fno-trapping-math"
    "--copt=-ffp-contract=fast" "--copt=-fdenormal-fp-math=positive-zero"
    "--copt=-DNDEBUG" "--copt=-Wno-unused-command-line-argument"

    "--cxxopt=-march=native" "--cxxopt=-mtune=native" "--cxxopt=-O3"
    "--cxxopt=-fomit-frame-pointer" "--cxxopt=-fno-semantic-interposition"
    "--cxxopt=-fno-plt" "--cxxopt=-ffunction-sections" "--cxxopt=-fdata-sections"
    "--cxxopt=-fno-math-errno" "--cxxopt=-fno-trapping-math"
    "--cxxopt=-ffp-contract=fast" "--cxxopt=-fdenormal-fp-math=positive-zero"
    "--cxxopt=-DNDEBUG" "--cxxopt=-Wno-unused-command-line-argument"

    #Disable TensorFlow and Abseil verbose logging.
    "--copt=-DTF_MIN_LOG_LEVEL=2" "--cxxopt=-DTF_MIN_LOG_LEVEL=2" "--host_copt=-DTF_MIN_LOG_LEVEL=2" "--host_cxxopt=-DTF_MIN_LOG_LEVEL=2"
    "--copt=-DABSL_MIN_LOG_LEVEL=2" "--cxxopt=-DABSL_MIN_LOG_LEVEL=2" "--host_copt=-DABSL_MIN_LOG_LEVEL=2" "--host_cxxopt=-DABSL_MIN_LOG_LEVEL=2"



    "--host_copt=-march=native" "--host_copt=-mtune=native" "--host_copt=-O3"
    "--host_copt=-fomit-frame-pointer" "--host_copt=-fno-semantic-interposition"
    "--host_copt=-fno-plt" "--host_copt=-ffunction-sections" "--host_copt=-fdata-sections"
    "--host_copt=-fno-math-errno" "--host_copt=-fno-trapping-math"
    "--host_copt=-ffp-contract=fast" "--host_copt=-fdenormal-fp-math=positive-zero"
    "--host_copt=-DNDEBUG" "--host_copt=-Wno-unused-command-line-argument"

    "--linkopt=-Wl,-O2" "--linkopt=-Wl,--as-needed" "--linkopt=-Wl,--gc-sections"
    "--linkopt=-L/usr/lib/x86_64-linux-gnu"
    "--linkopt=-L/lib/x86_64-linux-gnu" "--linkopt=-fuse-ld=lld"

    # Hermetic action environment that prevents Conda/TensorRT from leaking in.
    "--action_env=LD_LIBRARY_PATH="
    "--action_env=LD_PRELOAD="
    "--action_env=PATH=/usr/bin:/bin"

    # Hermetic toolchain and Python.
    "--repo_env=TF_CUDA_CLANG=1"
    "--repo_env=TF_NEED_CUDA=1"
    "--repo_env=TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES}"
    "--action_env=TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES}"
    "--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=${HERMETIC_CUDA_COMPUTE_CAPABILITIES:-${TF_CUDA_COMPUTE_CAPABILITIES}}"
    "--repo_env=USE_CUDA_TAR_ARCHIVE_FILES=1"
    "--repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION:-3.12}"
  )

  if [[ "${mode}" == "profile_generate" ]]; then
    if [[ -z "${profile_dir}" ]]; then
      log "ERROR: profile_generate mode requires a profile directory."
      exit 1
    fi
    build_opts+=(
      "--copt=-fprofile-instr-generate=${profile_dir}"
      "--cxxopt=-fprofile-instr-generate=${profile_dir}"
      "--linkopt=-fprofile-instr-generate=${profile_dir}"
      "--linkopt=-L${CLANG_PROFILE_LIB_DIR}"
      "--linkopt=-lclang_rt.profile-x86_64"
    )
  elif [[ "${mode}" == "profile_use" ]]; then
    if [[ -z "${profile_dir}" ]]; then
      log "ERROR: profile_use mode requires a profile directory."
      exit 1
    fi
    local profdata="${profile_dir}/tensorflow.profdata"
    if [[ ! -f "${profdata}" ]]; then
      log "ERROR: Profile data ${profdata} not found."
      exit 1
    fi
    build_opts+=(
      "--copt=-fprofile-instr-use=${profdata}"
      "--copt=-Wno-profile-instr-out-of-date"
      "--cxxopt=-fprofile-instr-use=${profdata}"
      "--cxxopt=-Wno-profile-instr-out-of-date"
      "--linkopt=-fprofile-instr-use=${profdata}"
      "--linkopt=-L${CLANG_PROFILE_LIB_DIR}"
      "--linkopt=-lclang_rt.profile-x86_64"
    )
  fi

  if [[ "${TF_NEED_CUDA:-0}" == "1" ]]; then
    local clang_bin
    clang_bin="$(command -v clang)"
    build_opts+=(
      "--config=cuda_clang"
      "--config=cuda_wheel"
      "--repo_env=TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES}"
      "--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES}"
      "--action_env=CLANG_CUDA_COMPILER_PATH=${clang_bin}"
      "--action_env=TF_CUDA_CLANG=1"
    )
  fi

  log "Starting Bazel build (hermetic actions)"
  build_opts+=('--repo_env=WHEEL_NAME=tensorflow')
  "${bazel}" build "${build_opts[@]}" //tensorflow/tools/pip_package:wheel

  local wheel_dir="${TF_REPO}/bazel-bin/tensorflow/tools/pip_package/wheel_house"
  local out_dir="${ROOT_DIR}/dist"
  mkdir -p "${out_dir}"

  if [[ ! -d "${wheel_dir}" ]]; then
    log "Wheel directory ${wheel_dir} not found after build."
    exit 1
  fi

  local produced_wheels
  produced_wheels=($(find "${wheel_dir}" -maxdepth 1 -type f -name "*.whl"))
  if [[ "${#produced_wheels[@]}" -eq 0 ]]; then
    log "No wheel found in ${wheel_dir}."
    exit 1
  fi

  for wheel_path in "${produced_wheels[@]}"; do
    local wheel_name
    wheel_name="$(basename "${wheel_path}")"
    rm -f "${out_dir}/${wheel_name}"
    cp "${wheel_path}" "${out_dir}/${wheel_name}"
    chmod u+w "${out_dir}/${wheel_name}"
  done

  LAST_BUILT_WHEELS=($(find "${out_dir}" -maxdepth 1 -type f -name "*.whl" -print))

  # Restore host environment before leaving the workspace.
  if [[ -n "${_saved_ld}" ]]; then export LD_LIBRARY_PATH="${_saved_ld}"; else unset LD_LIBRARY_PATH; fi
  if [[ -n "${_saved_preload}" ]]; then export LD_PRELOAD="${_saved_preload}"; else unset LD_PRELOAD; fi

  popd >/dev/null

  if [[ "${mode}" == "profile_use" || "${mode}" == "release" ]]; then
    for wheel in "${LAST_BUILT_WHEELS[@]}"; do
      validate_tensorflow_runtime "${wheel}"
    done
  fi
}


main() {
  ensure_directories
  ensure_system_prereqs
  check_perf_dependencies
  ensure_local_jdk
  ensure_bazelisk
  export PATH="${TOOLS_DIR}:${PATH}"
  ensure_python_venv

  detect_cuda_compute_capabilities

  local latest_tag
  log "Querying GitHub for latest TensorFlow release tag"
  latest_tag="$(fetch_latest_tag)"
  log "Latest TensorFlow release: ${latest_tag}"

  sync_tensorflow_source "${latest_tag}"
  prepare_cuda_toolchain
  configure_tensorflow
  build_profiled_tensorflow
  "${BAZELISK_BIN}" shutdown >/dev/null 2>&1 || true

  log "SUCCESS: Build complete. Wheels are available under ${DIST_DIR}"
}


main "$@"
