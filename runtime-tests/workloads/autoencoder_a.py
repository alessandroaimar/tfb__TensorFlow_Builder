#!/usr/bin/env python3
"""AUTOENCODER-A: GPU convolutional autoencoder with skip connections."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, set_global_seed


def build_dataset(batch_size: int) -> tf.data.Dataset:
    samples = np.random.rand(10000, 48, 48, 1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(samples)
    dataset = dataset.map(lambda x: tf.image.random_contrast(x, 0.8, 1.2), num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.map(lambda x: (x, x), num_parallel_calls=8)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32, use_prefetch_to_gpu=True)


def encoder_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def decoder_block(x, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(48, 48, 1))
    skips = []
    x = inputs
    for filters in (32, 64, 96):
        x = encoder_block(x, filters)
        skips.append(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    for filters, skip in zip((96, 64, 32), reversed(skips)):
        x = tf.keras.layers.Concatenate()([x, skip])
        x = decoder_block(x, filters)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    set_global_seed(404)
    configure_device("GPU", "AUTOENCODER-A")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 56
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", jit_compile=True)

    callback = BatchEndCallback("autoencoder_a_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 48, 48, 1))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference reconstruction mean:", tf.reduce_mean(outputs).numpy())


if __name__ == "__main__":
    main()
