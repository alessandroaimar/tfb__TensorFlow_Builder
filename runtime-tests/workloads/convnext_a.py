#!/usr/bin/env python3
"""ConvNeXt-A: GPU ConvNeXt-inspired blocks with LayerScale."""

import numpy as np
import tensorflow as tf

from common import BatchEndCallback, apply_common_pipeline, configure_device, configure_threads, identity_layer, set_global_seed


class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_value=1e-5):
        super().__init__()
        self.init_value = init_value

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
            name="layerscale_gamma",
        )

    def call(self, inputs):
        return inputs * self.gamma


def layerscale_block(x, filters, layer_scale=1e-5):
    input_tensor = x
    x = tf.keras.layers.DepthwiseConv2D(7, padding="same")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(4 * filters, activation="gelu")(x)
    x = tf.keras.layers.Dense(filters)(x)
    x = LayerScale(layer_scale)(x)
    if input_tensor.shape[-1] != filters:
        input_tensor = tf.keras.layers.Conv2D(filters, 1, padding="same")(input_tensor)
    return tf.keras.layers.Add()([input_tensor, x])


def build_dataset(batch_size: int) -> tf.data.Dataset:
    samples = np.random.rand(9000, 96, 96, 3).astype(np.float32)
    labels = np.random.randint(0, 100, size=(9000,), dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
    dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y), num_parallel_calls=8)
    dataset = dataset.repeat()
    return apply_common_pipeline(dataset, batch_size=batch_size, prefetch_buffer=16, use_prefetch_to_gpu=True)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Conv2D(64, 4, strides=4)(inputs)
    for filters in (96, 128, 160):
        x = layerscale_block(x, filters)
        x = tf.keras.layers.Conv2D(filters, 2, strides=2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = identity_layer()(tf.keras.layers.Dense(100)(x))
    return tf.keras.Model(inputs, logits)


def main():
    set_global_seed(777)
    configure_device("GPU", "ConvNeXt-A")
    configure_threads(inter_op=4, intra_op=8)

    batch_size = 28
    dataset = build_dataset(batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=True,
    )

    callback = BatchEndCallback("convnext_a_batch_end")
    model.fit(dataset, epochs=2, steps_per_epoch=1000, callbacks=[callback])

    x_infer = tf.random.uniform((batch_size, 96, 96, 3))
    for _ in range(100):
        outputs = model(x_infer, training=False)
    print("Inference logits max:", tf.reduce_max(outputs).numpy())


if __name__ == "__main__":
    main()
