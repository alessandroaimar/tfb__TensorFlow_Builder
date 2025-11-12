#!/usr/bin/env python3
"""LSTM-A: GPU variable-length sequence workload with custom train_step."""

import itertools
import numpy as np
import tensorflow as tf

from common import (
    BatchEndCallback,
    configure_device,
    configure_threads,
    dataset_options,
    set_global_seed,
    tf_identity_fn,
)


FEATURE_DIM = 64


def sequence_generator():
    for i in itertools.count():
        length = np.random.randint(30, 90)
        seq = np.random.randn(length, FEATURE_DIM).astype(np.float32)
        label = np.int32(i % 5)
        yield seq, label


def _normalize_py(seq: tf.Tensor) -> np.ndarray:
    arr = np.array(seq, dtype=np.float32)
    arr = arr - np.mean(arr, axis=0, keepdims=True)
    return arr.astype(np.float32)


@tf.function
def pad_identity(seq):
    return tf_identity_fn(seq)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    output_signature = (
        tf.TensorSpec(shape=(None, FEATURE_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(sequence_generator, output_signature=output_signature)

    def map_fn(seq, label):
        seq = tf.py_function(_normalize_py, inp=[seq], Tout=tf.float32)
        seq.set_shape((None, FEATURE_DIM))
        seq = pad_identity(seq)
        label = tf.cast(label, tf.int32)
        return seq, label

    dataset = dataset.map(map_fn, num_parallel_calls=8)
    dataset = dataset.flat_map(lambda seq, lbl: tf.data.Dataset.from_tensors((seq, lbl)))
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None, FEATURE_DIM]), tf.TensorShape([])),
        drop_remainder=True,
    )
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(8)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    return dataset.with_options(dataset_options())


class SequenceClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(512, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(512)
        self.proj = tf.keras.layers.Dense(5)

    def call(self, inputs, training=False):
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        logits = tf.identity(self.proj(x))
        return logits

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.compiled_loss(y, logits)
        grads = tape.gradient(loss, self.trainable_variables)

        def loop_body(i, agg):
            agg += tf.cast(i, tf.float32)
            return i + 1, agg

        tf.while_loop(lambda i, _: i < 2, loop_body, (tf.constant(0), tf.constant(0.0)))
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics} | {"loss": loss}


def main():
    set_global_seed(135)
    configure_device("GPU", "LSTM-A")
    configure_threads(inter_op=4, intra_op=16)

    batch_size = 32
    dataset = build_dataset(batch_size)

    model = SequenceClassifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
    )

    callback = BatchEndCallback("lstm_a_batch_end")
    model.fit(dataset, epochs=3, steps_per_epoch=20, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 60, FEATURE_DIM))
    outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
