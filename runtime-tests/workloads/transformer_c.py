#!/usr/bin/env python3
"""Transformer-C: CPU text pipeline with tf.strings and lookup tables."""

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

MAX_LEN = 40
VOCAB = tf.constant([
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
])

initializer = tf.lookup.KeyValueTensorInitializer(
    keys=VOCAB,
    values=tf.range(tf.size(VOCAB), dtype=tf.int64),
)
LOOKUP_TABLE = None


def get_lookup_table():
    global LOOKUP_TABLE
    if LOOKUP_TABLE is None:
        LOOKUP_TABLE = tf.lookup.StaticHashTable(initializer, default_value=0)
    return LOOKUP_TABLE


@tf.function
def text_to_ids(text):
    tokens = tf.strings.split(text)
    tokens = tokens[:MAX_LEN]
    pad = tf.maximum(0, MAX_LEN - tf.shape(tokens)[0])
    padded_tokens = tf.pad(tokens, [[0, pad]])
    ids = get_lookup_table().lookup(padded_tokens)
    return tf_identity_fn(tf.cast(ids, tf.int32))


def build_dataset(batch_size: int) -> tf.data.Dataset:
    sentences = tf.constant([
        "alpha beta gamma",
        "delta epsilon",
        "gamma theta alpha beta",
        "eta zeta",
        "theta theta beta",
        "epsilon alpha",
        "delta gamma eta",
        "beta beta beta",
    ])
    labels = tf.range(tf.size(sentences), dtype=tf.int32) % 4
    dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))

    def map_fn(text, label):
        ids = text_to_ids(tf.strings.lower(text))
        return ids, tf.cast(label, tf.int32)

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=32)
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(64, 64)(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(4, 16)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(4)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(112)
    configure_threads(inter_op=16, intra_op=32)
    configure_device("CPU", "Transformer-C")

    batch_size = 16
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchBeginCallback("transformer_c_batch_begin")
    model.fit(dataset, epochs=10, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.zeros((batch_size, MAX_LEN), dtype=tf.int32)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
