#!/usr/bin/env python3
"""Shared utilities for the TensorFlow profiling workloads."""

import os
import random
import numpy as np

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # pylint: disable=wrong-import-position


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_device(device: str, program_name: str) -> None:
    """Force CPU or verify GPU availability based on the workload spec."""
    device = device.upper()
    if device == "GPU":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError(f"{program_name} requires /GPU:0 but none is visible")
        tf.config.set_visible_devices(gpus, "GPU")
    else:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # Already initialized; continue.
            pass


def configure_threads(inter_op: int, intra_op: int) -> None:
    """Set explicit thread counts for intra/inter op parallelism."""
    try:
        tf.config.threading.set_inter_op_parallelism_threads(inter_op)
        tf.config.threading.set_intra_op_parallelism_threads(intra_op)
    except RuntimeError:
        # Threads already configured; continue with existing settings.
        pass


class BatchBeginCallback(tf.keras.callbacks.Callback):
    """Lightweight callback toggled on batch begin."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.signal = tf.Variable(0.0, dtype=tf.float32)

    def on_train_batch_begin(self, batch: int, logs=None):
        del logs  # unused
        self.signal.assign_add(1.0)


class BatchEndCallback(tf.keras.callbacks.Callback):
    """Lightweight callback toggled on batch end."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.signal = tf.Variable(0.0, dtype=tf.float32)

    def on_train_batch_end(self, batch: int, logs=None):
        del logs  # unused
        self.signal.assign_add(1.0)


def dataset_options() -> tf.data.Options:
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return opts


@tf.function
def tf_identity_fn(x: tf.Tensor) -> tf.Tensor:
    """Small tf.function utility so every workload can invoke it."""
    return tf.identity(x)


def identity_layer(name: str | None = None) -> tf.keras.layers.Layer:
    """Keras-friendly layer that applies tf.identity."""
    return tf.keras.layers.Lambda(lambda t: tf.identity(t), name=name)


def apply_common_pipeline(dataset: tf.data.Dataset,
                          batch_size: int,
                          prefetch_buffer,
                          use_prefetch_to_gpu: bool = False) -> tf.data.Dataset:
    """Apply the required tf.data transforms (map already handled externally)."""
    dataset = dataset.flat_map(lambda *elems: tf.data.Dataset.from_tensors(elems))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch_buffer)
    if use_prefetch_to_gpu:
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    return dataset.with_options(dataset_options())
