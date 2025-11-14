#!/usr/bin/env python3
"""Transformer-A: GPU mixed-precision encoder with TFRecord input."""

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

SEQ_LEN = 64
VOCAB = 2048


@tf.function
def scale_tokens(tokens):
    return tf_identity_fn(tokens)


def _write_token_records(num_examples: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tfrecord")
    writer = tf.io.TFRecordWriter(tmp.name)
    for _ in range(num_examples):
        tokens = np.random.randint(0, VOCAB, size=(SEQ_LEN,), dtype=np.int64)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "tokens": tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(np.random.randint(0, 16))])),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()
    return tmp.name


def build_dataset(batch_size: int) -> tf.data.Dataset:
    path = _write_token_records(4096)
    dataset = tf.data.TFRecordDataset(path)

    feature_spec = {
        "tokens": tf.io.VarLenFeature(tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_fn(serialized):
        example = tf.io.parse_single_example(serialized, feature_spec)
        tokens = tf.sparse.to_dense(example["tokens"], default_value=0)
        tokens = tf.cast(tokens, tf.int32)
        tokens = scale_tokens(tokens[:SEQ_LEN])
        label = tf.cast(example["label"], tf.int32)
        return tokens, label

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=tf.data.AUTOTUNE,
        use_prefetch_to_gpu=True,
    )
    return dataset


def transformer_block(x, d_model, num_heads, dropout_rate):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads, d_model // num_heads)(x, x)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    x = tf.keras.layers.LayerNormalization()(x + attn_output)
    x = identity_layer()(x)
    ff = tf.keras.layers.Dense(d_model * 4, activation=tf.nn.gelu)(x)
    ff = tf.keras.layers.Dense(d_model)(ff)
    ff = tf.keras.layers.Dropout(dropout_rate)(ff)
    y = tf.keras.layers.LayerNormalization()(x + ff)
    return identity_layer()(y)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    embeddings = tf.keras.layers.Embedding(VOCAB, 128)(inputs)
    x = embeddings
    for _ in range(6):
        x = transformer_block(x, 128, 4, 0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(16)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(321)
    configure_device("GPU", "Transformer-A")
    configure_threads(inter_op=4, intra_op=16)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    batch_size = 62
    dataset = build_dataset(batch_size)

    model = build_model()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(1e-3))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchBeginCallback("transformer_a_batch_begin")
    model.fit(dataset, epochs=1, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, SEQ_LEN), maxval=VOCAB, dtype=tf.int32)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits dtype:", outputs.dtype)


if __name__ == "__main__":
    main()
