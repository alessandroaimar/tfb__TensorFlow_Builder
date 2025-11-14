#!/usr/bin/env python3
"""LSTM-C: CPU ragged-input workload with unique/argsort ops."""

import itertools
import numpy as np
import tensorflow as tf

from common import (
    BatchBeginCallback,
    configure_device,
    configure_threads,
    dataset_options,
    set_global_seed,
    tf_identity_fn,
)

FEATURES = 32


def ragged_generator():
    for i in itertools.count():
        length = np.random.randint(20, 70)
        seq = tf.ragged.constant(np.random.randn(length, FEATURES).astype(np.float32))
        label = np.int32(i % 7)
        yield seq, label


@tf.function
def ragged_scale(seq):
    return tf_identity_fn(seq * 1.0)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    output_signature = (
        tf.RaggedTensorSpec(shape=(None, FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(ragged_generator, output_signature=output_signature)

    def map_fn(seq, label):
        seq = ragged_scale(seq)
        label = tf.cast(label, tf.int32)
        return seq, label

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.flat_map(lambda seq, lbl: tf.data.Dataset.from_tensors((seq, lbl)))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(32)
    return dataset.with_options(dataset_options())


class RaggedLSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Masking(),
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(7),
            ]
        )

    def call(self, inputs, training=False):
        logits = self.model(inputs, training=training)
        return tf.identity(logits)

    def train_step(self, data):
        ragged_seq, labels = data
        dense_inputs = ragged_seq.to_tensor()
        logs = super().train_step((dense_inputs, labels))
        lengths = tf.reduce_sum(tf.cast(tf.reduce_any(tf.not_equal(dense_inputs, 0.0), axis=-1), tf.int32), axis=1)
        unique_lengths = tf.unique(lengths).y
        _ = tf.argsort(unique_lengths)
        return logs


def main():
    set_global_seed(404)
    configure_device("CPU", "LSTM-C")
    configure_threads(inter_op=16, intra_op=32)

    batch_size = 8
    dataset = build_dataset(batch_size)

    model = RaggedLSTM()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchBeginCallback("lstm_c_batch_begin")
    model.fit(dataset, epochs=2, steps_per_epoch=1000, callbacks=[callback])

    ragged = tf.ragged.constant([np.random.randn(30, FEATURES), np.random.randn(40, FEATURES)], dtype=tf.float32)
    for _ in range(100):
        outputs = model(ragged.to_tensor(), training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
