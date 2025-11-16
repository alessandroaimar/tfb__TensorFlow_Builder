#!/usr/bin/env python3
"""AUTOENCODER-B: CPU sequence autoencoder with bidirectional LSTMs."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, set_global_seed


def build_dataset(batch_size: int) -> tf.data.Dataset:
    sequences = np.random.randn(9000, 40, 32).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.repeat().map(lambda s: (s, s), num_parallel_calls=4)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(40, 32))
    encoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(inputs)
    encoded = tf.keras.layers.Dense(64, activation="tanh")(encoded)
    repeated = tf.keras.layers.RepeatVector(40)(encoded)
    decoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(repeated)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32))(decoded)
    return tf.keras.Model(inputs, outputs)


def main():
    set_global_seed(222)
    configure_device("CPU", "AUTOENCODER-B")
    configure_threads(inter_op=2, intra_op=4)

    batch_size = 32
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="mae", jit_compile=False)

    callback = BatchEndCallback("autoencoder_b_batch_end")
    model.fit(dataset, epochs=1, steps_per_epoch=800, callbacks=[callback])

    x_infer = tf.random.normal((batch_size, 40, 32))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference reconstruction std:", tf.math.reduce_std(outputs).numpy())


if __name__ == "__main__":
    main()
