#!/usr/bin/env python3
"""Transformer-B: GPU fp32 encoder with dynamic padding and masks."""

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

MAX_LEN = 80
VOCAB = 1024


@tf.function
def build_mask(lengths):
    mask = tf.sequence_mask(lengths, MAX_LEN, dtype=tf.float32)
    return tf_identity_fn(tf.where(mask > 0, mask, tf.zeros_like(mask)))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    tokens = np.zeros((num_samples, MAX_LEN), dtype=np.int32)
    lengths = np.random.randint(20, MAX_LEN, size=(num_samples,), dtype=np.int32)
    for i in range(num_samples):
        seq_len = lengths[i]
        tokens[i, :seq_len] = np.random.randint(1, VOCAB, size=(seq_len,), dtype=np.int32)
    labels = np.random.randint(0, 12, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tokens, lengths, labels))

    def map_fn(tok, length, label):
        tok = tf.cast(tok, tf.int32)
        length = tf.cast(length, tf.int32)
        mask = build_mask(length)
        return (tok, mask), tf.cast(label, tf.int32)

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=8)
    return dataset


def transformer_layer(x, mask):
    attention = tf.keras.layers.MultiHeadAttention(4, 128 // 4)(x, x, attention_mask=mask[:, None, None, :])
    x = tf.keras.layers.LayerNormalization()(x + attention)
    ff = tf.keras.layers.Dense(256, activation=tf.nn.gelu)(x)
    ff = tf.keras.layers.Dense(128)(ff)
    return identity_layer()(tf.keras.layers.LayerNormalization()(x + ff))


def build_model() -> tf.keras.Model:
    token_input = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    mask_input = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.float32)
    embeddings = tf.keras.layers.Embedding(VOCAB, 128)(token_input)
    x = embeddings
    for _ in range(3):
        x = transformer_layer(x, mask_input)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(12)(x))
    return tf.keras.Model((token_input, mask_input), logits)


def main():
    set_global_seed(654)
    configure_device("GPU", "Transformer-B")
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

    callback = BatchEndCallback("transformer_b_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    mask = tf.where(tf.sequence_mask(tf.constant([60] * batch_size), MAX_LEN), 1.0, 0.0)
    x_infer = (
        tf.random.uniform((batch_size, MAX_LEN), maxval=VOCAB, dtype=tf.int32),
        tf.cast(mask, tf.float32),
    )
    outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
