#!/usr/bin/env python3
"""CNN-E: GPU mixed-precision micro-batch with JPEG decode surrogate."""

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


def _jpeg_decode_stub(byte_string: tf.Tensor) -> tf.Tensor:
    array = np.frombuffer(byte_string.numpy(), dtype=np.uint8)
    image = array.reshape(32, 32, 3).astype(np.float32)
    return image / 255.0


@tf.function
def identity_aug(image):
    return tf_identity_fn(image)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 1024
    images = (np.random.rand(num_samples, 32, 32, 3) * 255).astype(np.uint8)
    encoded = [img.tobytes() for img in images]
    labels = np.random.randint(0, 10, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((encoded, labels))

    def map_fn(byte_string, label):
        image = tf.py_function(_jpeg_decode_stub, inp=[byte_string], Tout=tf.float32)
        image.set_shape((32, 32, 3))
        image = identity_aug(image)
        label = tf.cast(label, tf.int32)
        return image, label

    dataset = dataset.map(map_fn, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=1,
        use_prefetch_to_gpu=True,
    )
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.DepthwiseConv2D(3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(42)
    configure_device("GPU", "CNN-E")
    configure_threads(inter_op=1, intra_op=8)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    batch_size = 1
    dataset = build_dataset(batch_size)

    model = build_model()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-3))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchBeginCallback("cnn_e_batch_begin")
    model.fit(dataset, epochs=10, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float16)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
