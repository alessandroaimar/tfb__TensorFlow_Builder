# Profiling Workloads

This directory hosts the twenty stand-alone TensorFlow programs requested for profiling the custom build. Each workload can be executed on demand after activating the profiling environment:

```bash
conda activate alt
python runtime-tests/workloads/<script>.py
```

## Workload Map

| Script | Label | Device | Precision | XLA | Batch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp_a.py` | MLP-A | GPU | fp32 | Off | 64 | Custom gradient MLP, batch-end callback, prefetch-to-device |
| `mlp_b.py` | MLP-B | GPU | mixed_float16 | On | 32 | Mixed precision + loss scaling, py_function, XLA |
| `mlp_c.py` | MLP-C | CPU | fp32 | Off | 64 | Generator input, custom train_step with tf.while_loop |
| `mlp_d.py` | MLP-D | GPU | fp32 | Off | 1 | Micro-batch with py_function scaling |
| `cnn_a.py` | CNN-A | GPU | mixed_float16 | On | 64 | TFRecord pipeline, ResNet blocks, prefetch-to-device |
| `cnn_b.py` | CNN-B | GPU | fp32 | Off | 32 | from_tensor_slices with unbatch/flat_map + py_function crop |
| `cnn_c.py` | CNN-C | CPU | fp32 | Off | 16 | Includes tf.image.non_max_suppression |
| `cnn_d.py` | CNN-D | GPU | fp32 | On | 16 | TFRecord with tf.cond augmentation |
| `cnn_e.py` | CNN-E | GPU | mixed_float16 | Off | 1 | JPEG-decode surrogate via py_function |
| `cnn_f.py` | CNN-F | CPU | fp32 | On | 32 | CPU XLA conv net |
| `lstm_a.py` | LSTM-A | GPU | fp32 | Off | 32 | Variable-length generator, custom train_step, py_function |
| `lstm_b.py` | LSTM-B | GPU | mixed_float16 | On | 64 | CuDNN-style LSTM with XLA |
| `lstm_c.py` | LSTM-C | CPU | fp32 | Off | 16 | Ragged inputs, tf.unique + tf.argsort |
| `gru_d.py` | GRU-D | GPU | fp32 | Off | 1 | GRU with tf.map_fn reduction, py_function noise |
| `rnn_e_sparse.py` | RNN-E | CPU | fp32 | Off | 32 | SparseTensor inputs with sparse_dense_matmul |
| `transformer_a.py` | Transformer-A | GPU | mixed_float16 | On | 64 | TFRecord token encoder with GELU + residual identities |
| `transformer_b.py` | Transformer-B | GPU | fp32 | Off | 32 | Dynamic padding + tf.where-based masks |
| `transformer_c.py` | Transformer-C | CPU | fp32 | Off | 16 | tf.strings preprocessing with StaticHashTable |
| `transformer_d.py` | Transformer-D | GPU | fp32 | On | 1 | Gradient-accumulating micro-batch, py_function filter |
| `transformer_e.py` | Transformer-E | GPU | mixed_float16 | Off | 16 | Block-sparse surrogate masks with tf.where |

## Suite Guarantees

- Deterministic seeds, explicit thread counts, tf.function/tf.cast/tf.identity usage, and tf.data `map → flat_map → batch → unbatch → cache → prefetch` structure in every script.
- GPU workloads check for `/GPU:0` visibility before running and apply `tf.data.experimental.prefetch_to_device('/GPU:0')` when required.
- Eight programs use `tf.py_function`; CPU-specific fallbacks cover ragged, sparse, tf.strings/tables, non-max suppression, tf.unique/tf.argsort, and dynamic shape control flows.
- Post-training inference always runs via direct model invocation.

The scripts are intentionally independent so you can pick and choose workloads for profiling without touching `build_tensorflow.sh`.
