#!/usr/bin/env python3
"""GRU-D: GPU micro-batch workload with tf.map_fn post-processing."""

import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
    tf_identity_fn,
)

SEQ_LEN = 50
FEATURES = 32
HIDDEN_UNITS = 128


def _py_noise(seq: tf.Tensor) -> np.ndarray:
    arr = np.array(seq, dtype=np.float32)
    noise = 0.01 * np.random.randn(*arr.shape).astype(np.float32)
    return arr + noise


@tf.function
def identity_seq(seq):
    return tf_identity_fn(seq)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 1024
    sequences = np.random.randn(num_samples, SEQ_LEN, FEATURES).astype(np.float32)
    labels = np.random.randint(0, 4, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))

    def map_fn(seq, label):
        seq = tf.py_function(_py_noise, inp=[seq], Tout=tf.float32)
        seq.set_shape((SEQ_LEN, FEATURES))
        seq = identity_seq(seq)
        return seq, tf.cast(label, tf.int32)

    dataset = dataset.map(map_fn, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=2)
    return dataset


@tf.autograph.experimental.do_not_convert
def _map_sum(step_tensor: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(step_tensor)


def map_reduce(seq_outputs: tf.Tensor) -> tf.Tensor:
    time_major = tf.transpose(seq_outputs, [1, 0, 2])
    tf.map_fn(_map_sum, time_major)
    return time_major[-1]


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(SEQ_LEN, FEATURES))
    x = tf.keras.layers.GRU(HIDDEN_UNITS, return_sequences=True)(inputs)
    x = tf.keras.layers.Lambda(map_reduce)(x)
    logits = identity_layer()(tf.keras.layers.Dense(4)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(999)
    configure_device("GPU", "GRU-D")
    configure_threads(inter_op=1, intra_op=8)

    batch_size = 1
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("gru_d_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, SEQ_LEN, FEATURES))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits:", outputs.numpy())


if __name__ == "__main__":
    main()
