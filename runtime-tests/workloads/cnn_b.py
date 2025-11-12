#!/usr/bin/env python3
"""CNN-B: GPU fp32 workload with custom flat_map augmentation and py_function."""

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


def _random_crop_py(image: tf.Tensor) -> np.ndarray:
    image = np.array(image, dtype=np.float32).reshape(32, 32, 3)
    top = np.random.randint(0, 5)
    left = np.random.randint(0, 5)
    cropped = image[top : top + 28, left : left + 28, :]
    return cropped.astype(np.float32)


@tf.function
def jitter_image(image):
    return tf_identity_fn(tf.image.random_brightness(image, 0.05))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    images = (np.random.rand(num_samples, 32, 32, 3) * 255).astype(np.uint8)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.batch(8).unbatch()

    def augment_fn(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.py_function(_random_crop_py, inp=[image], Tout=tf.float32)
        image.set_shape((28, 28, 3))
        image = tf.image.resize(image, (32, 32))
        image = jitter_image(image / 255.0)
        label = tf.cast(label, tf.int32)
        return image, label

    dataset = dataset.flat_map(lambda img, lbl: tf.data.Dataset.from_tensors(augment_fn(img, lbl)))
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(515)
    configure_device("GPU", "CNN-B")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 32
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("cnn_b_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 32, 32, 3))
    outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
