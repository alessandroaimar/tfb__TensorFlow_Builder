#!/usr/bin/env python3
"""Transformer-F: GPU encoder-only transformer with rotary embeddings."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed

SEQ_LEN = 96
VOCAB = 1200


def build_dataset(batch_size: int) -> tf.data.Dataset:
    tokens = np.random.randint(0, VOCAB, size=(10000, SEQ_LEN), dtype=np.int32)
    labels = np.random.randint(0, VOCAB, size=(10000, SEQ_LEN), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
    dataset = dataset.shuffle(2048).repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16, use_prefetch_to_gpu=True)


class RotaryPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len)[tf.newaxis, :, tf.newaxis]
        angles = tf.cast(positions, inputs.dtype) / 10000.0
        sin = tf.sin(angles)
        cos = tf.cos(angles)
        reversed_inputs = tf.reverse(inputs, axis=[-1])
        return inputs * cos + reversed_inputs * sin


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(VOCAB, 256)(inputs)
    x = RotaryPositionalEncoding()(x)
    for _ in range(4):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.1)(x, x)
        attn = tf.keras.layers.Dropout(0.1)(attn)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ff = tf.keras.layers.Dense(512, activation="gelu")(x)
        ff = tf.keras.layers.Dense(256)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)
    logits = identity_layer()(tf.keras.layers.Dense(VOCAB)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(5151)
    configure_device("GPU", "Transformer-F")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 24
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True,
    )

    callback = BatchEndCallback("transformer_f_batch_end")
    model.fit(dataset, epochs=2, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, SEQ_LEN), maxval=VOCAB, dtype=tf.int32)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits mean:", tf.reduce_mean(outputs).numpy())


if __name__ == "__main__":
    main()
