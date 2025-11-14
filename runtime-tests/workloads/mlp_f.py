#!/usr/bin/env python3
"""MLP-F: CPU residual MLP with stochastic depth."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed


def stochastic_dense(x, units, survival_prob=0.9):
    out = tf.keras.layers.Dense(units, activation="relu")(x)
    if tf.random.uniform(()) > survival_prob:
        return x
    return tf.keras.layers.Add()([x, out])


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 5000
    x = np.random.rand(num_samples, 512).astype(np.float32)
    y = np.random.randint(0, 15, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(lambda a, b: (tf.math.log1p(a), b), num_parallel_calls=4)
    dataset = dataset.repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(512,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    for _ in range(4):
        x = stochastic_dense(x, 256, survival_prob=0.85)
    x = tf.keras.layers.LayerNormalization()(x)
    logits = identity_layer()(tf.keras.layers.Dense(15)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(99)
    configure_device("CPU", "MLP-F")
    configure_threads(inter_op=2, intra_op=4)

    batch_size = 64
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("mlp_f_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 512))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits std:", tf.math.reduce_std(outputs).numpy())


if __name__ == "__main__":
    main()
