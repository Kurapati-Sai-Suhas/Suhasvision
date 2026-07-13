# Architecture Ground Truth

**Date Confirmed:** 2026-07-09

The production model (`cricket_stance_advanced_v4.keras` and `v2.keras`) is a `[Conv1D(64) -> BiLSTM(64) -> TemporalAttention -> Dense(32) -> Dense(4, sigmoid)]` network with **76,395 trainable parameters**. This architecture has been definitively confirmed via `model.summary()` on the actual deployed `.keras` artifacts.

Crucially, this **perfectly matches** the current `academic_scripts/train_advanced_model.py` script. The training script and the deployed artifacts are completely in sync.

The previous claim in the SRS that the "BiLSTM was abandoned" was incorrect. The model is currently and actively using a Bidirectional LSTM wrapper with 64 units (resulting in an output shape of 128).

## Explicit Layer Details

1. **Input Layer:** `(None, 7, 30)` - 7 frames, 30 features (15 angles + 15 velocities)
2. **Conv1D:** `(None, 7, 64)` - 64 filters, `kernel_size=3`
3. **BatchNormalization & Dropout (0.2)**
4. **Bidirectional(LSTM):** `(None, 7, 128)` - 64 units per direction
5. **Dropout (0.3)**
6. **TemporalAttention:** `(None, 128)` - Custom attention layer
7. **Dense:** `(None, 32)` - Intermediate regression head
8. **Dense:** `(None, 4)` - Final sigmoid output (Balance, Power, Technique, Defence)

## Appendix: `model.summary()` Output

```text
Model: "functional"
+--------------------------------------------------------------------------+
| Layer (type)                    | Output Shape           |       Param # |
|---------------------------------+------------------------+---------------|
| input_layer (InputLayer)        | (None, 7, 30)          |             0 |
|---------------------------------+------------------------+---------------|
| conv1d (Conv1D)                 | (None, 7, 64)          |         5,824 |
|---------------------------------+------------------------+---------------|
| batch_normalization             | (None, 7, 64)          |           256 |
| (BatchNormalization)            |                        |               |
|---------------------------------+------------------------+---------------|
| dropout (Dropout)               | (None, 7, 64)          |             0 |
|---------------------------------+------------------------+---------------|
| bidirectional (Bidirectional)   | (None, 7, 128)         |        66,048 |
|---------------------------------+------------------------+---------------|
| dropout_1 (Dropout)             | (None, 7, 128)         |             0 |
|---------------------------------+------------------------+---------------|
| temporal_attention              | (None, 128)            |           135 |
| (TemporalAttention)             |                        |               |
|---------------------------------+------------------------+---------------|
| dense (Dense)                   | (None, 32)             |         4,128 |
|---------------------------------+------------------------+---------------|
| dense_1 (Dense)                 | (None, 4)              |           132 |
+--------------------------------------------------------------------------+
 Total params: 229,315 (895.77 KB)
 Trainable params: 76,395 (298.42 KB)
 Non-trainable params: 128 (512.00 B)
 Optimizer params: 152,792 (596.85 KB)
```
