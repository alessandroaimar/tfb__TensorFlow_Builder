#!/usr/bin/env python3
"""Generator-C: GPU ragged sequences projected through attention stack."""

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

MAX_STEPS = 20
EMBED_SIZE = 24
NUM_CLASSES = 26


def build_dataset(batch_size: int) -> tf.data.Dataset:
    def gen():
        rng = np.random.default_rng(357)
        while True:
            length = rng.integers(6, MAX_STEPS + 1)
            features = rng.standard_normal((length, EMBED_SIZE)).astype(np.float32)
            label = np.int32(rng.integers(NUM_CLASSES))
            yield features, label

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, EMBED_SIZE), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    def pad_sequences(feat, label):
        length = tf.shape(feat)[0]
        length = tf.minimum(length, MAX_STEPS)
        truncated = feat[:length]
        paddings = [[0, MAX_STEPS - length], [0, 0]]
        dense = tf.pad(truncated, paddings)
        mask = tf.concat([tf.ones((length, 1)), tf.zeros((MAX_STEPS - length, 1))], axis=0)
        return (dense, mask), label

    dataset = dataset.map(pad_sequences, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda packed, label: (tf.concat([packed[0], packed[1]], axis=-1), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16, use_prefetch_to_gpu=True)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(MAX_STEPS, EMBED_SIZE + 1))
    x = tf.keras.layers.LayerNormalization()(inputs)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.Add()([attn, x])
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48))(x)
    logits = identity_layer()(tf.keras.layers.Dense(NUM_CLASSES)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(357)
    configure_device("GPU", "Generator-C")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 40
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("generator_c_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.normal((batch_size, MAX_STEPS, EMBED_SIZE + 1))
    for _ in range(100):
        _ = model(x_infer, training=False)
    print("Inference logits mean:", tf.reduce_mean(model(x_infer, training=False)))


if __name__ == "__main__":
    main()
