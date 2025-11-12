#!/usr/bin/env python3
"""MLP-C: CPU workload with generator-driven input and custom train_step.""" 

import itertools
import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    apply_common_pipeline,
    configure_device,
    configure_threads,
    set_global_seed,
    tf_identity_fn,
)


def cpu_generator():
    for i in itertools.count():
        features = np.cos(np.arange(1024) * 0.01 + i).astype(np.float32)
        noise = np.random.randn(1024).astype(np.float32) * 0.05
        label = np.int32(i % 10)
        yield features + noise, label


@tf.function
def cpu_heavy_cast(x):
    return tf_identity_fn(tf.cast(x, tf.float32))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    output_signature = (
        tf.TensorSpec(shape=(1024,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(cpu_generator, output_signature=output_signature)

    def map_fn(x, y):
        x = cpu_heavy_cast(x)
        y = tf.cast(y, tf.int32)
        return x, y

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32)
    return dataset


class CustomMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(2048, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(2048, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

    def call(self, inputs, training=False):
        logits = self.net(inputs, training=training)
        return tf.identity(logits)

    def train_step(self, data):
        logs = super().train_step(data)

        def body(i, acc):
            acc = tf.identity(acc + tf.cast(i, tf.float32))
            return i + 1, acc

        tf.while_loop(lambda i, _: i < 3, body, (tf.constant(0), tf.constant(0.0)))
        return logs


def main():
    set_global_seed(2024)
    configure_device("CPU", "MLP-C")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 64
    dataset = build_dataset(batch_size)

    model = CustomMLP()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("mlp_c_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 1024))
    outputs = model(x_infer, training=False)
    print("Inference logits sample:", outputs[0, :5])


if __name__ == "__main__":
    main()
