#!/usr/bin/env python3
"""Generator-D: CPU interleave-heavy pipeline with stateless transforms."""

import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
)

FEATURE_SHAPE = (18, 12)
NUM_CLASSES = 16


def build_dataset(batch_size: int) -> tf.data.Dataset:
    def expand(idx):
        idx = tf.cast(idx, tf.int32)
        return tf.data.Dataset.range(3).map(lambda offset: (idx, tf.cast(offset, tf.int32)))

    dataset = tf.data.Dataset.range(3000)
    dataset = dataset.interleave(expand, cycle_length=8, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)

    def to_example(idx, offset):
        seed = tf.stack([idx * 11 + offset, idx * 7 + 3])
        base = tf.random.stateless_uniform(FEATURE_SHAPE, seed=seed)
        ramp = tf.linspace(0.0, 1.0, FEATURE_SHAPE[1])
        features = base + tf.expand_dims(ramp, 0)
        label = tf.math.floormod(idx + offset, NUM_CLASSES)
        return features, tf.cast(label, tf.int32)

    dataset = dataset.map(to_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(1024)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=FEATURE_SHAPE)
    x = tf.keras.layers.LayerNormalization()(inputs)
    x = tf.keras.layers.Reshape((FEATURE_SHAPE[0], FEATURE_SHAPE[1], 1))(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    logits = identity_layer()(tf.keras.layers.Dense(NUM_CLASSES)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(468)
    configure_device("CPU", "Generator-D")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 32
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("generator_d_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1800, callbacks=[callback])

    x_infer = tf.random.normal((batch_size,) + FEATURE_SHAPE)
    for _ in range(100):
        _ = model(x_infer, training=False)
    print("Inference logits max:", tf.reduce_max(model(x_infer, training=False)))


if __name__ == "__main__":
    main()
