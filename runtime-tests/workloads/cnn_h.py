#!/usr/bin/env python3
"""CNN-H: CPU Conv1D temporal classifier with causal padding."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 6000
    signals = np.random.randn(num_samples, 256, 4).astype(np.float32)
    labels = np.random.randint(0, 8, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((signals, labels))
    def frame_signal(signal, label):
        framed = tf.signal.frame(signal, 64, 16, pad_end=False, axis=0)
        framed.set_shape((13, 64, 4))
        return framed, label

    dataset = dataset.map(frame_signal, num_parallel_calls=4)
    dataset = dataset.repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(13, 64, 4))
    x = tf.keras.layers.Reshape((13, 64 * 4))(inputs)
    x = tf.keras.layers.Conv1D(128, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(96))(x)
    logits = identity_layer()(tf.keras.layers.Dense(8)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(212)
    configure_device("CPU", "CNN-H")
    configure_threads(inter_op=2, intra_op=8)

    batch_size = 20
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("cnn_h_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.normal((batch_size, 13, 64, 4))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
