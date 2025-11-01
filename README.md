# TensorFlow Builder

Highly opinionated automation for compiling a custom, GPU-enabled TensorFlow build with Clang, ThinLTO, and profile-guided optimisation (PGO). The project wraps the official TensorFlow build in a hermetic workflow that downloads the toolchain it needs, harvests runtime profiles, and validates the resulting wheels.

## Highlights
- Fully automated end-to-end build via `build_tensorflow.sh`, including dependency checks, virtualenv management, and TensorFlow source syncing.
- Hermetic CUDA toolchain defaults (Clang + LLD + `cuda_malloc_async`) tuned for NVIDIA GPUs with compute capabilities auto-detected from the host.
- Two-phase PGO pipeline: build an instrumented wheel, execute a profiling workload, merge coverage, and rebuild with `profile_use`.
- Runtime sanity checks that exercise the wheel twice—baseline environment and the high-performance configuration defined here.
- Optional environment overlay (`high_perf_env.sh`) that toggles TensorFlow/Numba tuning knobs without polluting the system profile.

## Prerequisites
The script expects a recent Linux distribution with:
- GNU build tooling (`git`, `make`, `gcc`, `g++`, `pkg-config`, `zlib1g-dev`, `libffi-dev`, `libssl-dev`)
- Python 3.10+ with `venv`
- LLVM/Clang 16+ plus `ld.lld`, `llvm-profdata`, and the Clang profile runtime
- NVIDIA CUDA-capable GPUs and drivers (for CUDA builds)

`build_tensorflow.sh` validates most of the above and explains how to fix missing components. When invoked on Ubuntu with passwordless sudo it can auto-install the supplementary apt packages.

## Quick Start
1. Ensure Clang/LLVM and the NVIDIA driver stack are installed and working (`nvidia-smi` should report your GPUs).
2. (Optional) Inspect `nvidia_diagnose_install.sh` to confirm the GPU driver provenance if the source is unclear.
3. Launch the build:
   ```bash
   ./build_tensorflow.sh
   ```
4. Wait for the script to:
   - Fetch the latest TensorFlow release tag from GitHub.
   - Provision a local Python build environment under `.tf-build-venv`.
   - Clone/update TensorFlow in `sources/tensorflow`.
   - Compile an instrumented wheel, run the profiling workload, merge profiles, and produce optimised wheels in `dist/`.
   - Execute runtime checks, first with default settings and then with the high-performance environment.
5. Install the desired wheel from `dist/` into your target environment.

## Repository Layout
- `build_tensorflow.sh` – Main orchestrator for building TensorFlow with hermetic toolchains and PGO.
- `high_perf_env.sh` – Source-only helper that exports CPU thread pinning, TensorFlow, XLA, CUDA, and Numba tuning variables. Existing exports always win so it can be layered selectively.
- `activate_high_perf.sh` – Thin wrapper used by the runtime checks; sourcing it is equivalent to sourcing `high_perf_env.sh`.
- `nvidia_diagnose_install.sh` – Utility that classifies how the NVIDIA driver stack was installed (APT, graphics-drivers PPA, `.run`, etc.).
- `tools/cuda_nvprune_stub` – Minimal Bazel workspace that provides a stub `nvprune` binary for hermetic CUDA builds.
- `sources/README.md` – Placeholder describing the expected TensorFlow checkout location; the actual source tree is ignored to keep the repository lightweight.
- `dist/`, `logs/`, `profiles/`, `runtime-tests/` – Build output directories ignored by Git. They will be recreated automatically.
- `.tf-build-venv/` – Python virtual environment that hosts build dependencies; recreated as needed.

## High-Performance Environment Overlay
Source `activate_high_perf.sh` (or directly `high_perf_env.sh`) inside a shell prior to launching TensorFlow or Numba workloads that should inherit the tuned settings:

```bash
source activate_high_perf.sh
python your_script.py
```

Notable toggles include enabling XLA, auto mixed precision, CUDA async allocator, aggressive thread tuning, and Numba CPU feature autodetection. Override any variable manually before sourcing to keep customised values.

## Cleaning Up
- Remove build artifacts: `rm -rf dist profiles runtime-tests logs`
- Reset the TensorFlow source tree: delete `sources/tensorflow` and rerun the build to re-clone.
- Remove the build virtual environment: `rm -rf .tf-build-venv`

## Troubleshooting Notes
- `tools/bazel` and `tools/bazelisk` are managed by the script; delete them to trigger a fresh download.
- If Clang profile libraries are missing (`check_perf_dependencies` aborts), install an LLVM toolchain package that ships `libclang_rt.profile-x86_64.a`.
- The profiling workload installs the freshly built wheel into an isolated venv under `runtime-tests/`—inspect `logs/runtime-test-*.log` if validation fails.
