#!/usr/bin/env python3
"""Execute the TensorFlow workload suite inside the current Python env."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import List

WORKLOADS: List[str] = [
    "sanity_runtime_check.py",
    "mlp_a.py",
    "mlp_b.py",
    "mlp_c.py",
    "mlp_d.py",
    "cnn_a.py",
    "cnn_b.py",
    "cnn_c.py",
    "cnn_d.py",
    "cnn_e.py",
    "cnn_f.py",
    "lstm_a.py",
    "lstm_b.py",
    "lstm_c.py",
    "gru_d.py",
    "rnn_e_sparse.py",
    "transformer_a.py",
    "transformer_b.py",
    "transformer_c.py",
    "transformer_d.py",
    "transformer_e.py",
    "cnn_g.py",
    "cnn_h.py",
    "mlp_e.py",
    "mlp_f.py",
    "autoencoder_a.py",
    "autoencoder_b.py",
    "transformer_f.py",
    "convnext_a.py",
    "spectrogram_a.py",
    "waveform_a.py",
    "generator_a.py",
    "generator_b.py",
    "generator_c.py",
    "generator_d.py",
    "generator_e.py",
]

ERROR_PATTERNS = [
    "traceback (most recent call last)",
    "cuda error",
    "cuda_error",
    "cublas_status",
    "cudnn_status",
    "illegal memory access",
    "segmentation fault",
    "core dumped",
]


def detect_log_issue(log_text: str) -> str | None:
    lowered = log_text.lower()
    for pattern in ERROR_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


def run(cmd: List[str], log_path: pathlib.Path) -> float:
    start = time.perf_counter()
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - start
    log_path.write_text(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} exited with {process.returncode}. See {log_path}."
        )
    issue = detect_log_issue(process.stdout)
    if issue:
        raise RuntimeError(
            f"Detected '{issue}' while running {' '.join(cmd)}. Inspect {log_path}."
        )
    return elapsed


def install_wheel(wheel: pathlib.Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        "--force-reinstall",
        str(wheel),
    ]
    subprocess.run(cmd, check=True)


def normalize_selection(raw: List[str]) -> List[str]:
    """Normalize workload names passed via CLI (strip paths)."""
    cleaned = []
    for item in raw:
        name = pathlib.Path(item).name
        cleaned.append(name)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=pathlib.Path, help="Optional wheel to install")
    parser.add_argument("--workloads-dir", required=True, type=pathlib.Path)
    parser.add_argument("--log-dir", required=True, type=pathlib.Path)
    parser.add_argument("--summary-file", type=pathlib.Path)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the specified workloads (can be given multiple times).",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip the specified workloads (can be given multiple times).",
    )
    args = parser.parse_args()

    workloads_dir = args.workloads_dir.resolve()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.wheel:
        if not args.wheel.is_file():
            raise FileNotFoundError(f"Wheel not found: {args.wheel}")
        install_wheel(args.wheel)

    if args.only:
        requested = normalize_selection(args.only)
        unknown = set(requested) - set(WORKLOADS)
        if unknown:
            raise ValueError(f"Unknown workload(s) specified via --only: {sorted(unknown)}")
        workload_list = [w for w in WORKLOADS if w in requested]
    else:
        workload_list = WORKLOADS.copy()

    if args.skip:
        skip_set = set(normalize_selection(args.skip))
        unknown = skip_set - set(WORKLOADS)
        if unknown:
            raise ValueError(f"Unknown workload(s) specified via --skip: {sorted(unknown)}")
        workload_list = [w for w in workload_list if w not in skip_set]

    if not workload_list:
        raise ValueError("No workloads selected to run.")

    durations = {}
    for workload in workload_list:
        script_path = workloads_dir / workload
        if not script_path.is_file():
            raise FileNotFoundError(f"Workload script missing: {script_path}")
        log_path = log_dir / f"{script_path.stem}.log"
        print(f"[run-workloads] Running {script_path} ...", flush=True)
        duration = run([sys.executable, str(script_path)], log_path)
        print(f"[run-workloads] Completed {script_path} in {duration:.1f}s", flush=True)
        durations[workload] = duration

    print("[run-workloads] Suite finished successfully.")
    summary_path = args.summary_file or (log_dir / "runtime-summary.json")
    summary_path.write_text(json.dumps(durations, indent=2, sort_keys=True))
    print(f"[run-workloads] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
