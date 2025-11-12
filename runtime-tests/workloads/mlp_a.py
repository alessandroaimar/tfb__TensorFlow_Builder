#!/usr/bin/env python3
"""MLP-A: GPU fp32 workload with fused activation and prefetch-to-device."""

import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    identity_layer,
    set_global_seed,
    tf_identity_fn,
)


@tf.custom_gradient
def fused_activation(x):
    """Custom ReLU-style activation that keeps gradients deterministic."""

    def grad(dy):
        return dy * tf.cast(x > 0.0, x.dtype)

    return tf.identity(tf.nn.relu(x)), grad


@tf.function
def preprocess_features(x):
    return tf_identity_fn(x)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    features = np.random.randn(num_samples, 1024).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    def map_fn(x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)
        x = preprocess_features(x)
        return x, y

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=32,
        use_prefetch_to_gpu=True,
    )
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(1024,))
    activation = tf.keras.layers.Lambda(fused_activation)

    x = tf.keras.layers.Dense(1024)(inputs)
    x = activation(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(2048)(x)
    x = activation(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(2048)(x)
    x = activation(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = activation(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = activation(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(42)
    configure_device("GPU", "MLP-A")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 64
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("mlp_a_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 1024), dtype=tf.float32)
    outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
