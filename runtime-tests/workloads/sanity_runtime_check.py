#!/usr/bin/env python3
"""Lightweight TensorFlow sanity workload used across profiling and validation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def maybe_enable_memory_growth() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(
            "WARNING: No GPU devices detected by TensorFlow runtime; skipping GPU matmul."
        )
        return False

    enabled = False
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            enabled = True
        except RuntimeError as exc:  # pylint: disable=broad-except
            print(f"WARNING: Unable to enable memory growth on {gpu.name}: {exc}")
    return enabled


def run_matmul(device_name: str) -> float:
    size = 2048
    with tf.device(device_name):
        a = tf.random.uniform((size, size), dtype=tf.float32)
        b = tf.random.uniform((size, size), dtype=tf.float32)
        c = tf.matmul(a, b)
        return float(tf.reduce_sum(c).numpy())


def main() -> None:
    tf.random.set_seed(0)
    np.random.seed(0)

    gpu_available = maybe_enable_memory_growth()

    print("TensorFlow version:", tf.__version__)
    print("Available devices:", tf.config.list_logical_devices())

    cpu_sum = run_matmul("/CPU:0")
    print(f"CPU matmul checksum: {cpu_sum:.6f}")

    if gpu_available:
        gpu_sum = run_matmul("/GPU:0")
        print(f"GPU matmul checksum: {gpu_sum:.6f}")


if __name__ == "__main__":
    main()
