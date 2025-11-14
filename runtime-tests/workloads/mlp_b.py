#!/usr/bin/env python3
"""MLP-B: GPU mixed-precision workload with XLA compilation."""

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


def _py_center(x: tf.Tensor) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    return (arr - np.mean(arr)).astype(np.float32)


@tf.function
def amplify_features(x):
    return tf_identity_fn(x * 1.01)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    features = np.random.randn(num_samples, 1024).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    def map_fn(x, y):
        x = tf.cast(x, tf.float32)
        x = tf.py_function(_py_center, inp=[x], Tout=tf.float32)
        x.set_shape((1024,))
        x = amplify_features(x)
        y = tf.cast(y, tf.int32)
        return x, y

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=8,
        use_prefetch_to_gpu=True,
    )
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(1024,))
    relu = tf.keras.layers.Activation("relu")

    x = tf.keras.layers.Dense(1024)(inputs)
    x = relu(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(2048)(x)
    x = relu(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(2048)(x)
    x = relu(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = relu(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = relu(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(123)
    configure_device("GPU", "MLP-B")
    configure_threads(inter_op=4, intra_op=16)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    batch_size = 16
    dataset = build_dataset(batch_size)

    model = build_model()
    base_opt = tf.keras.optimizers.Adam(2e-3)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_opt)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchBeginCallback("mlp_b_batch_begin")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 1024), dtype=tf.float16)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
