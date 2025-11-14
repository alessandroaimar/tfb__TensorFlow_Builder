#!/usr/bin/env python3
"""Generator-E: GPU reshape-heavy latent projector."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed

FEATURE_SHAPE = (14, 14)
NUM_CLASSES = 28

ROWS, COLS = FEATURE_SHAPE



def build_dataset(batch_size: int) -> tf.data.Dataset:
    def gen():
        rng = np.random.default_rng(579)
        while True:
            features = rng.standard_normal(FEATURE_SHAPE).astype(np.float32)
            label = rng.integers(NUM_CLASSES, dtype=np.int32)
            yield features, label

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=FEATURE_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    dataset = dataset.repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16, use_prefetch_to_gpu=True)


ROWS, COLS = FEATURE_SHAPE


def matmul_features(t: tf.Tensor) -> tf.Tensor:
    prod = tf.linalg.matmul(t, t, transpose_b=True)
    transposed = tf.transpose(t, perm=(0, 2, 1))
    prod_flat = tf.reshape(prod, (-1, ROWS * ROWS))
    trans_flat = tf.reshape(transposed, (-1, COLS * ROWS))
    return tf.concat([prod_flat, trans_flat], axis=1)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=FEATURE_SHAPE)
    x = tf.keras.layers.Reshape(FEATURE_SHAPE)(inputs)
    x = tf.keras.layers.Lambda(matmul_features, output_shape=(ROWS * ROWS + COLS * ROWS,))(x)
    x = tf.keras.layers.Dense(256, activation="swish")(x)
    x = tf.keras.layers.Reshape((-1, 32))(x)
    x = tf.keras.layers.Lambda(lambda t: tf.transpose(t, perm=(0, 2, 1)))(x)
    x = tf.keras.layers.Flatten()(x)
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
        optimizer=tf.keras.optimizers.Adam(7e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("generator-e_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.normal((batch_size,) + FEATURE_SHAPE)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
