#!/usr/bin/env python3
"""Transformer-D: GPU micro-batch with gradient accumulation and py_function."""

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

SEQ_LEN = 48
VOCAB = 1024
ACCUM_STEPS = 4


def _py_filter(tokens: tf.Tensor) -> np.ndarray:
    arr = np.array(tokens, dtype=np.int32)
    return np.where(arr % 2 == 0, arr, 0).astype(np.int32)


@tf.function
def scale_inputs(tokens):
    return tf_identity_fn(tokens)


def build_dataset(batch_size: int) -> tf.data.Dataset:
    num_samples = 1024
    tokens = np.random.randint(0, VOCAB, size=(num_samples, SEQ_LEN), dtype=np.int32)
    labels = np.random.randint(0, 8, size=(num_samples,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))

    def map_fn(tok, label):
        tok = tf.py_function(_py_filter, inp=[tok], Tout=tf.int32)
        tok.set_shape((SEQ_LEN,))
        tok = scale_inputs(tok)
        return tok, tf.cast(label, tf.int32)

    dataset = dataset.map(map_fn, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = apply_common_pipeline(
        dataset,
        batch_size=batch_size,
        prefetch_buffer=1,
        use_prefetch_to_gpu=True,
    )
    return dataset


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(VOCAB, 128)(inputs)
    x = tf.keras.layers.MultiHeadAttention(4, 32)(x, x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(8)(x))
    return tf.keras.Model(inputs, logits)


class AccumulatingModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.accumulated_grads = None
        self.grad_acc_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

    @tf.function
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.compiled_loss(y, logits)
        grads = tape.gradient(loss, self.trainable_variables)
        if self.accumulated_grads is None:
            self.accumulated_grads = [tf.zeros_like(g) for g in grads]
        self.accumulated_grads = [acc + g for acc, g in zip(self.accumulated_grads, grads)]
        self.grad_acc_counter.assign_add(1)
        apply_now = tf.equal(self.grad_acc_counter, ACCUM_STEPS)
        def apply_grads():
            scaled = [g / tf.cast(ACCUM_STEPS, g.dtype) for g in self.accumulated_grads]
            self.optimizer.apply_gradients(zip(scaled, self.trainable_variables))
            self.accumulated_grads = [tf.zeros_like(g) for g in scaled]
            self.grad_acc_counter.assign(0)
            return 0
        tf.cond(apply_now, apply_grads, lambda: 0)
        self.compiled_metrics.update_state(y, logits)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs


def main():
    set_global_seed(888)
    configure_device("GPU", "Transformer-D")
    configure_threads(inter_op=1, intra_op=8)

    batch_size = 1
    dataset = build_dataset(batch_size)

    base_model = build_model()
    model = AccumulatingModel(base_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False,
        run_eagerly=True,
    )

    callback = BatchBeginCallback("transformer_d_batch_begin")
    model.fit(dataset, epochs=2, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, SEQ_LEN), maxval=VOCAB, dtype=tf.int32)
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits shape:", outputs.shape)


if __name__ == "__main__":
    main()
