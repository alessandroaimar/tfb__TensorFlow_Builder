#!/usr/bin/env python3
"""CNN-F: CPU workload using XLA-compiled conv net."""

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


@tf.function
def normalize_image(image):
    return tf_identity_fn(tf.cast(image, tf.float32) / 255.0)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    images = (np.random.rand(num_samples, 28, 28, 1) * 255).astype(np.uint8)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    def map_fn(image, label):
        image = normalize_image(image)
        label = tf.cast(label, tf.int32)
        return image, label

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32)
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(808)
    configure_device("CPU", "CNN-F")
    configure_threads(inter_op=32, intra_op=16)

    batch_size = 31
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("cnn_f_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 28, 28, 1))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits sample:", outputs[0, :5])


if __name__ == "__main__":
    main()
