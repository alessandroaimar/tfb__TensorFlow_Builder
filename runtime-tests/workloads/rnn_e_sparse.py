#!/usr/bin/env python3
"""RNN-E: CPU sparse-tensor workload with explicit sparse matmul."""

import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    configure_device,
    configure_threads,
    dataset_options,
    set_global_seed,
    tf_identity_fn,
)

VECTOR_DIM = 512


@tf.function
def sparse_prep(vec):
    return tf_identity_fn(vec)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    dense = np.random.randn(num_samples, VECTOR_DIM).astype(np.float32)
    labels = np.random.randint(0, 3, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((dense, labels))

    def map_fn(vec, label):
        vec = sparse_prep(tf.cast(vec, tf.float32))
        sparse_tensor = tf.sparse.from_dense(tf.where(tf.abs(vec) > 0.5, vec, tf.zeros_like(vec)))
        label = tf.cast(label, tf.int32)
        return sparse_tensor, label

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.flat_map(lambda feats, lbl: tf.data.Dataset.from_tensors((feats, lbl)))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(8)
    return dataset.with_options(dataset_options())


class SparseClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.kernel = self.add_weight(shape=(VECTOR_DIM, 64), initializer="glorot_uniform")
        self.bias = self.add_weight(shape=(64,), initializer="zeros")
        self.dense = tf.keras.layers.Dense(3)

    @tf.function
    def call(self, inputs, training=False):
        sparse_inputs = inputs
        hidden = tf.sparse.sparse_dense_matmul(sparse_inputs, self.kernel) + self.bias
        mask = tf.where(hidden > 0.0, 1.0, 0.5)
        hidden = tf.nn.relu(hidden * mask)
        shape_info = tf.shape(mask)
        _ = tf.identity(shape_info)
        logits = tf.identity(self.dense(hidden, training=training))
        return logits


def main():
    set_global_seed(707)
    configure_device("CPU", "RNN-E")
    configure_threads(inter_op=32, intra_op=16)

    batch_size = 64
    dataset = build_dataset(batch_size)

    model = SparseClassifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("rnn_e_batch_end")
    model.fit(dataset, epochs=70, steps_per_epoch=3000, callbacks=[callback])

    sparse_sample = tf.sparse.from_dense(tf.where(tf.random.uniform((batch_size, VECTOR_DIM)) > 0.7, 1.0, 0.0))
    for _ in range(100):
        outputs = model(sparse_sample, training=False)
    print("Inference logits:", outputs[:1])


if __name__ == "__main__":
    main()
