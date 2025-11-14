#!/usr/bin/env python3
"""CNN-A: GPU mixed-precision TFRecord-based ResNet-style workload."""

import tempfile
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


FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _write_tfrecord(num_examples: int, image_shape=(32, 32, 3)) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecord")
    writer = tf.io.TFRecordWriter(tmp.name)
    for _ in range(num_examples):
        image = (np.random.rand(*image_shape) * 255).astype(np.uint8)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(np.random.randint(0, 10))])),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()
    return tmp.name


@tf.function
def augment_image(x):
    return tf_identity_fn(tf.image.random_flip_left_right(x))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    record_path = _write_tfrecord(2048)
    dataset = tf.data.TFRecordDataset(record_path)

    def parse_fn(serialized):
        example = tf.io.parse_single_example(serialized, FEATURE_DESC)
        image = tf.io.decode_raw(example["image"], tf.uint8)
        image = tf.reshape(image, (32, 32, 3))
        image = tf.cast(image, tf.float32) / 255.0
        image = augment_image(image)
        label = tf.cast(example["label"], tf.int32)
        return image, label

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=tf.data.AUTOTUNE,
        use_prefetch_to_gpu=True,
    )
    return dataset


def resnet_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = identity_layer()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = identity_layer()(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = x + shortcut
    return tf.keras.layers.Activation("relu")(x)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = identity_layer()(x)
    x = tf.keras.layers.Activation("relu")(x)
    for filters in [32, 64, 128]:
        x = resnet_block(x, filters)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(314)
    configure_device("GPU", "CNN-A")
    configure_threads(inter_op=4, intra_op=16)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    batch_size = 63
    dataset = build_dataset(batch_size)

    model = build_model()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-3))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchBeginCallback("cnn_a_batch_begin")
    model.fit(dataset, epochs=4, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float16)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
