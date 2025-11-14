#!/usr/bin/env python3
"""MLP-E: GPU mixture-of-experts style dense network with gating."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 7000
    features = np.random.randn(num_samples, 1536).astype(np.float32)
    labels = np.random.randint(0, 50, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(2048).repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32, use_prefetch_to_gpu=True)


def expert_block(x, units):
    gate = tf.keras.layers.Dense(units, activation="sigmoid")(x)
    expert = tf.keras.layers.Dense(units, activation="gelu")(x)
    return tf.keras.layers.Multiply()([gate, expert])


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(1536,))
    x = tf.keras.layers.Dense(1024, activation="swish")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    experts = [expert_block(x, 512) for _ in range(4)]
    x = tf.keras.layers.Concatenate()(experts)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    logits = identity_layer()(tf.keras.layers.Dense(50)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(808)
    configure_device("GPU", "MLP-E")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 40
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(2e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("mlp_e_batch_end")
    model.fit(dataset, epochs=5, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.normal((batch_size, 1536))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits sample:", outputs[0, :5])


if __name__ == "__main__":
    main()
