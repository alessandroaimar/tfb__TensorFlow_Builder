#!/usr/bin/env python3
"""Spectrogram-A: CPU audio spectrogram classifier with Conv2D + attention."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed


def build_dataset(batch_size: int) -> tf.data.Dataset:
    samples = np.random.randn(2000, 512).astype(np.float32)
    labels = np.random.randint(0, 12, size=(2000,), dtype=np.int32)

    def preprocess(wave, label):
        spec = tf.signal.stft(wave, frame_length=128, frame_step=64)
        spec = tf.abs(spec)[..., tf.newaxis]
        spec = tf.image.resize(spec, (64, 64))
        return spec, label

    dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).map(preprocess, num_parallel_calls=4)
    dataset = dataset.repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(64, 64, 1))
    x = tf.keras.layers.Conv2D(32, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    seq = tf.keras.layers.Reshape((32 * 32, 64))(x)
    attn = tf.keras.layers.Attention()([seq, seq])
    attn = tf.keras.layers.Flatten()(attn)
    logits = identity_layer()(tf.keras.layers.Dense(12)(attn))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(345)
    configure_device("CPU", "Spectrogram-A")
    configure_threads(inter_op=2, intra_op=6)

    batch_size = 32
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("spectrogram_a_batch_end")
    model.fit(dataset, epochs=1, steps_per_epoch=80, callbacks=[callback])

    x_infer = tf.random.normal((batch_size, 64, 64, 1))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
