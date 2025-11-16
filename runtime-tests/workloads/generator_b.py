#!/usr/bin/env python3
"""Generator-B: CPU windowed tf.data pipeline with stateless noise."""

import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
)

FEATURE_SHAPE = (8, 32)
NUM_CLASSES = 14


def build_dataset(batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.range(6000)
    dataset = dataset.window(5, shift=2, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(5))

    def make_example(indices):
        indices = tf.cast(indices, tf.int32)
        base_seed = tf.stack([indices[0] * 13 + 7, indices[-1] * 5 + 3])
        noise = tf.random.stateless_normal((5, FEATURE_SHAPE[0], FEATURE_SHAPE[1]), seed=base_seed)
        weights = tf.linspace(0.2, 1.0, 5)[:, tf.newaxis, tf.newaxis]
        features = tf.reduce_sum(noise * tf.cast(weights, tf.float32), axis=0)
        label = tf.math.floormod(tf.reduce_sum(indices), NUM_CLASSES)
        return features, tf.cast(label, tf.int32)

    dataset = dataset.map(make_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(1024)
    dataset = dataset.repeat()
    opts = tf.data.Options()
    opts.threading.private_threadpool_size = 4
    dataset = dataset.with_options(opts)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=FEATURE_SHAPE)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Reshape((4, 32))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(x)
    logits = identity_layer()(tf.keras.layers.Dense(NUM_CLASSES)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(246)
    configure_device("CPU", "Generator-B")
    configure_threads(inter_op=2, intra_op=4)

    batch_size = 36
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(8e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("generator_b_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1600, callbacks=[callback])

    x_infer = tf.random.normal((batch_size,) + FEATURE_SHAPE)
    for _ in range(100):
        _ = model(x_infer, training=False)
    print("Inference logits std:", tf.math.reduce_std(model(x_infer, training=False)))


if __name__ == "__main__":
    main()
