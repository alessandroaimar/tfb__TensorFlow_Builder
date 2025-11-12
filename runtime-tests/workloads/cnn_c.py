#!/usr/bin/env python3
"""CNN-C: CPU fp32 workload with tf.image.non_max_suppression."""

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


@tf.function
def nms_augment(image):
    boxes = tf.stack([
        tf.random.uniform((16,), minval=0.0, maxval=0.8),
        tf.random.uniform((16,), minval=0.2, maxval=1.0),
        tf.random.uniform((16,), minval=0.0, maxval=0.8),
        tf.random.uniform((16,), minval=0.2, maxval=1.0),
    ], axis=1)
    scores = tf.random.uniform((16,), minval=0.0, maxval=1.0)
    selected = tf.image.non_max_suppression(boxes, scores, max_output_size=5)
    selection_mask = tf.reduce_sum(tf.one_hot(selected, depth=16), axis=0)
    scale = 1.0 + tf.reduce_mean(selection_mask)
    return tf_identity_fn(tf.cast(image, tf.float32) * (scale / 2.0))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 2048
    images = (np.random.rand(num_samples, 32, 32, 3) * 255).astype(np.float32)
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    def map_fn(image, label):
        image = nms_augment(image / 255.0)
        label = tf.cast(label, tf.int32)
        return image, label

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32)
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(910)
    configure_device("CPU", "CNN-C")
    configure_threads(inter_op=16, intra_op=32)

    batch_size = 16
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchBeginCallback("cnn_c_batch_begin")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 32, 32, 3))
    outputs = model(x_infer, training=False)
    print("Inference logits sample:", outputs[0, :5])


if __name__ == "__main__":
    main()
