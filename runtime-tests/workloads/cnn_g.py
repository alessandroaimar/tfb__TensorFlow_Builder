#!/usr/bin/env python3
"""CNN-G: GPU depthwise-separable CNN with squeeze-excite blocks."""

import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
)


def squeeze_excite(x, ratio=8):
    filters = x.shape[-1]
    squeezed = tf.keras.layers.GlobalAveragePooling2D()(x)
    squeezed = tf.keras.layers.Dense(filters // ratio, activation="relu")(squeezed)
    excited = tf.keras.layers.Dense(filters, activation="sigmoid")(squeezed)
    excited = tf.keras.layers.Reshape((1, 1, filters))(excited)
    return tf.keras.layers.Multiply()([x, excited])


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 8192
    images = np.random.rand(num_samples, 64, 64, 3).astype(np.float32)
    labels = np.random.randint(0, 20, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(1024).map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=8)
    dataset = dataset.repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32, use_prefetch_to_gpu=True)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    x = inputs
    for filters in (64, 96, 128):
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", activation="swish")(x)
        x = squeeze_excite(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(192, 1, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(20)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(131)
    configure_device("GPU", "CNN-G")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 46
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("cnn_g_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 64, 64, 3))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
