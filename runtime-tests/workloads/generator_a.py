#!/usr/bin/env python3
"""Generator-A: GPU stateless FFT features combining spatial and spectral cues."""

import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
)

FEATURE_SHAPE = (12, 24)
NUM_CLASSES = 20


def build_dataset(batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.range(8000)

    def to_example(index):
        seed = tf.stack([tf.cast(index, tf.int64), tf.cast(index * 17 + 3, tf.int64)])
        base = tf.random.stateless_normal(FEATURE_SHAPE, seed=seed)
        spectrum = tf.signal.fft2d(tf.cast(base, tf.complex64))
        features = tf.math.real(spectrum)
        rotated = tf.image.rot90(
            tf.expand_dims(features, axis=-1), k=tf.cast(index % 4, tf.int32)
        )
        resized = tf.image.resize(rotated, FEATURE_SHAPE)
        final = resized[:, :, 0]
        final.set_shape(FEATURE_SHAPE)
        label = tf.math.floormod(tf.cast(index, tf.int32), NUM_CLASSES)
        return final, label

    dataset = dataset.map(to_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024)

    def augment(feat, label):
        aug = tf.image.random_flip_left_right(tf.expand_dims(feat, -1))[:, :, 0]
        aug.set_shape(FEATURE_SHAPE)
        return aug, label

    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16, use_prefetch_to_gpu=True)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=FEATURE_SHAPE)
    x = tf.keras.layers.LayerNormalization()(inputs)
    x = tf.keras.layers.Reshape(FEATURE_SHAPE + (1,))(x)
    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="swish")(x)
    x = tf.keras.layers.Conv2D(64, 1, activation="swish")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    logits = identity_layer()(tf.keras.layers.Dense(NUM_CLASSES)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(135)
    configure_device("GPU", "Generator-A")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 44
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("generator_a_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1500, callbacks=[callback])

    x_infer = tf.random.normal((batch_size,) + FEATURE_SHAPE)
    for _ in range(100):
        _ = model(x_infer, training=False)
    print("Inference logits shape:", model.output_shape)


if __name__ == "__main__":
    main()
