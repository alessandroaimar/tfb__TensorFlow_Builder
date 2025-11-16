#!/usr/bin/env python3
"""Generator-E: GPU scan-based dataset with dynamic conditioning."""

import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
)

FEATURE_SHAPE = (6, 48)
NUM_CLASSES = 18


def build_dataset(batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.range(5000)

    def scan_fn(state, value):
        next_state = (state + tf.cast(value, tf.int32) + 1) % 10007
        return next_state, next_state

    dataset = dataset.scan(tf.constant(0, tf.int32), scan_fn)

    def to_example(state):
        seed = tf.stack([state, state + 17])
        features = tf.random.stateless_normal(FEATURE_SHAPE, seed=seed)
        features = tf.signal.fft(tf.cast(features, tf.complex64))
        features = tf.math.real(features)
        label = tf.math.floormod(state, NUM_CLASSES)
        return features, tf.cast(label, tf.int32)

    dataset = dataset.map(to_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16, use_prefetch_to_gpu=True)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=FEATURE_SHAPE)
    x = tf.keras.layers.Reshape((FEATURE_SHAPE[0], FEATURE_SHAPE[1], 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (1, 3), padding="same", activation="swish")(x)
    x = tf.keras.layers.Conv2D(64, (1, 3), padding="same", activation="swish")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(96, activation="gelu")(x)
    logits = identity_layer()(tf.keras.layers.Dense(NUM_CLASSES)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(579)
    configure_device("GPU", "Generator-E")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 36
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(9e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("generator_e_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1800, callbacks=[callback])

    x_infer = tf.random.normal((batch_size,) + FEATURE_SHAPE)
    for _ in range(100):
        _ = model(x_infer, training=False)
    print("Inference logits min:", tf.reduce_min(model(x_infer, training=False)))


if __name__ == "__main__":
    main()
