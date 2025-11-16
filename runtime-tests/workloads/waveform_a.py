#!/usr/bin/env python3
"""Waveform-A: GPU temporal convolution plus dilated residual stacks."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 7000
    wave = np.random.randn(num_samples, 1024, 2).astype(np.float32)
    labels = np.random.randint(0, 30, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((wave, labels))
    dataset = dataset.shuffle(1024).repeat()
    dataset = dataset.map(lambda x, y: (tf.complex(x, tf.zeros_like(x)), y), num_parallel_calls=8)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32, use_prefetch_to_gpu=True)


def residual_block(x, dilation):
    shortcut = x
    x = tf.keras.layers.Conv1D(128, 3, padding="causal", dilation_rate=dilation, activation="relu")(x)
    x = tf.keras.layers.Conv1D(128, 1, activation="relu")(x)
    return tf.keras.layers.Add()([shortcut, x])


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(1024, 2), dtype=tf.complex64)
    x = tf.keras.layers.Lambda(lambda z: tf.math.real(z))(inputs)
    x = tf.keras.layers.Conv1D(128, 5, padding="causal", activation="relu")(x)
    for dilation in (1, 2, 4, 8, 16):
        x = residual_block(x, dilation)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(30)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(929)
    configure_device("GPU", "Waveform-A")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 36
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("waveform_a_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=200, callbacks=[callback])

    x_infer = tf.complex(tf.random.normal((batch_size, 1024, 2)), tf.random.normal((batch_size, 1024, 2)))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits min:", tf.reduce_min(outputs).numpy())


if __name__ == "__main__":
    main()
