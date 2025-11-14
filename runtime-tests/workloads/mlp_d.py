#!/usr/bin/env python3
"""MLP-D: GPU micro-batch workload with host prefetch and py_function."""

import numpy as np
import tensorflow as tf

from common import (
    BatchBeginCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
    tf_identity_fn,
)


def _py_scale(x: tf.Tensor) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    return (arr * 1.5).astype(np.float32)


@tf.function
def project_features(x):
    return tf_identity_fn(x)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 2048
    features = np.random.rand(num_samples, 1024).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    def map_fn(x, y):
        x = tf.cast(x, tf.float32)
        x = tf.py_function(_py_scale, inp=[x], Tout=tf.float32)
        x.set_shape((1024,))
        x = project_features(x)
        y = tf.cast(y, tf.int32)
        return x, y

    dataset = dataset.map(map_fn, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=1)
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(1024,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(77)
    configure_device("GPU", "MLP-D")
    configure_threads(inter_op=1, intra_op=1)

    batch_size = 3
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchBeginCallback("mlp_d_batch_begin")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 1024))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits:", outputs.numpy())


if __name__ == "__main__":
    main()
