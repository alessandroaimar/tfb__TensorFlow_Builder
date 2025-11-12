#!/usr/bin/env python3
"""Transformer-E: GPU mixed-precision workload with block-sparse masks."""

import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    configure_device,
    configure_threads,
    dataset_options,
    identity_layer,
    set_global_seed,
    tf_identity_fn,
)

SEQ_LEN = 72
VOCAB = 4096


def _py_shuffle(tokens: tf.Tensor) -> np.ndarray:
    arr = np.array(tokens, dtype=np.int32)
    return np.roll(arr, 1).astype(np.int32)


@tf.function
def build_block_mask(batch_size):
    base_block = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    tiled = tf.tile(base_block, [SEQ_LEN // 2, SEQ_LEN // 2])
    tiled = tiled[:SEQ_LEN, :SEQ_LEN]
    shape = tf.concat([[batch_size], tf.constant([SEQ_LEN, SEQ_LEN], dtype=tf.int32)], axis=0)
    broadcast = tf.broadcast_to(tiled, shape)
    return tf_identity_fn(tf.where(broadcast > 0, broadcast, tf.zeros_like(broadcast)))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 4096
    tokens = np.random.randint(0, VOCAB, size=(num_samples, SEQ_LEN), dtype=np.int32)
    labels = np.random.randint(0, 6, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))

    def map_fn(tok, label):
        tok = tf.py_function(_py_shuffle, inp=[tok], Tout=tf.int32)
        tok.set_shape((SEQ_LEN,))
        tok = tf.identity(tf.cast(tok, tf.int32))
        return tok, tf.cast(label, tf.int32)

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.flat_map(lambda *elems: tf.data.Dataset.from_tensors(elems))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def add_masks(batch_tokens, batch_labels):
        mask = build_block_mask(tf.shape(batch_tokens)[0])
        return (batch_tokens, mask), batch_labels

    dataset = dataset.map(add_masks, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(3)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    return dataset.with_options(dataset_options())


def build_model() -> tf.keras.Model:
    token_input = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    mask_input = tf.keras.layers.Input(shape=(SEQ_LEN, SEQ_LEN), dtype=tf.float32)
    x = tf.keras.layers.Embedding(VOCAB, 128)(token_input)
    attention = tf.keras.layers.MultiHeadAttention(4, 32)
    for _ in range(2):
        attn_output = attention(x, x, attention_mask=mask_input)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)
        ff = tf.keras.layers.Dense(256, activation=tf.nn.gelu)(x)
        ff = tf.keras.layers.Dense(128)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(6)(x))
    return tf.keras.Model((token_input, mask_input), logits)


def main():
    set_global_seed(555)
    configure_device("GPU", "Transformer-E")
    configure_threads(inter_op=4, intra_op=16)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    batch_size = 16
    dataset = build_dataset(batch_size)

    model = build_model()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-3))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("transformer_e_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    mask = build_block_mask(batch_size)
    x_infer = (
        tf.random.uniform((batch_size, SEQ_LEN), maxval=VOCAB, dtype=tf.int32),
        mask,
    )
    outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
