#!/usr/bin/env python3
"""CNN-D: GPU fp32 TFRecord workload with tf.cond augmentation and XLA."""

import tempfile
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


FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _write_records(num_examples: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecord")
    writer = tf.io.TFRecordWriter(tmp.name)
    for _ in range(num_examples):
        image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
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
def conditional_augment(image):
    def brighten():
        return tf.identity(tf.image.random_brightness(image, 0.1))

    def contrast():
        return tf.identity(tf.image.random_contrast(image, 0.8, 1.2))

    return tf.cond(tf.random.uniform(()) > 0.5, brighten, contrast)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    path = _write_records(2048)
    dataset = tf.data.TFRecordDataset(path)

    def parse_fn(serialized):
        example = tf.io.parse_single_example(serialized, FEATURES)
        image = tf.io.decode_raw(example["image"], tf.uint8)
        image = tf.reshape(image, (32, 32, 3))
        image = tf.cast(image, tf.float32) / 255.0
        image = conditional_augment(image)
        label = tf.cast(example["label"], tf.int32)
        return image, label

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=3,
        use_prefetch_to_gpu=True,
    )
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = identity_layer()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = identity_layer()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(10)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(2718)
    configure_device("GPU", "CNN-D")
    configure_threads(inter_op=4, intra_op=16)

    batch_size = 16
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("cnn_d_batch_end")
    model.fit(dataset, epochs=10, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 32, 32, 3))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
