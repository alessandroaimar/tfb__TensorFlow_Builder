#!/usr/bin/env python3
"""LSTM-B: GPU mixed-precision CuDNN LSTM workload with XLA."""

import numpy as np
import tensorflow as tf

from common import (
    BatchBeginCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    set_global_seed,
    tf_identity_fn,
)

SEQ_LEN = 80
FEATURES = 48


@tf.function
def scale_sequences(seq):
    return tf_identity_fn(seq * 0.99)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    sequences = np.random.randn(num_samples, SEQ_LEN, FEATURES).astype(np.float32)
    labels = np.random.randint(0, 5, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))

    def map_fn(seq, label):
        seq = tf.cast(seq, tf.float32)
        seq = scale_sequences(seq)
        label = tf.cast(label, tf.int32)
        return seq, label

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=tf.data.AUTOTUNE,
        use_prefetch_to_gpu=True,
    )
    return dataset


class CuDNNLSTMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(
            512, return_sequences=True, return_state=True
        )
        self.lstm2 = tf.keras.layers.LSTM(512)
        self.proj = tf.keras.layers.Dense(5)

    def call(self, inputs, training=False):
        seq_output, state_h, state_c = self.lstm1(inputs, training=training)
        state_h = tf.identity(state_h)
        _ = tf.identity(state_c)
        x = self.lstm2(seq_output, training=training)
        logits = tf.identity(self.proj(x))
        return logits


def main():
    set_global_seed(246)
    configure_device("GPU", "LSTM-B")
    configure_threads(inter_op=4, intra_op=16)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    batch_size = 64
    dataset = build_dataset(batch_size)

    model = CuDNNLSTMModel()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-3))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchBeginCallback("lstm_b_batch_begin")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, SEQ_LEN, FEATURES), dtype=tf.float16)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
