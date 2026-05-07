# -*- coding: utf-8 -*-
"""
train_lstm.py — Main training script
All functions are imported from lstm_utils.py
To train a different model: change the CONFIG section only!
"""

from lstm_utils import (load_data, create_sequences, split_data,
                         to_tensors, make_loader, train_model,
                         evaluate_model, plot_curves, PickPlaceLSTM)

# ══════════════════════════════════════════════════════════════
# CONFIG — change only this section to adapt to a new model
# ══════════════════════════════════════════════════════════════
CSV_PATH    = "pickplace_records_1776331338.csv"
FEATURES    = [
    "box_at_place", "part_at_place", "detected", "c_limit",
    "box_conveyor", "part_conveyor", "exit_conveyor", "c_plus", "grap"
]
N_PAST      = 10    # number of past states
HIDDEN_SIZE = 64    # LSTM hidden neurons
N_LAYERS    = 1     # LSTM layers
EPOCHS      = 100   # max training epochs
LR          = 0.001 # learning rate
BATCH_SIZE  = 32    # batch size
PATIENCE    = 10    # early stopping patience
WEIGHTS_PATH = "pickplace_lstm_weights.pth"
PLOT_PATH    = "lstm_training_curves.png"

# ══════════════════════════════════════════════════════════════
# TRAINING PIPELINE — do not change below this line
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("   LSTM Training — Pick and Place XYZ")
print("=" * 60)
print()

# 1. Load data
print("Step 1 — Loading data...")
data = load_data(CSV_PATH, FEATURES)
print()

# 2. Create sequences
print("Step 2 — Creating sequences...")
X, Y = create_sequences(data, N_PAST)
print()

# 3. Split data
print("Step 3 — Splitting data...")
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y)
print()

# 4. Convert to tensors + DataLoader
print("Step 4 — Preparing tensors...")
X_train_t, X_val_t, X_test_t, Y_train_t, Y_val_t, Y_test_t = to_tensors(
    X_train, X_val, X_test, Y_train, Y_val, Y_test)
train_loader = make_loader(X_train_t, Y_train_t, BATCH_SIZE)
print(f"  Batch size={BATCH_SIZE} | Batches per epoch={len(train_loader)}")
print()

# 5. Build model
print("Step 5 — Building model...")
model = PickPlaceLSTM(
    input_size  = len(FEATURES),
    hidden_size = HIDDEN_SIZE,
    n_layers    = N_LAYERS,
    output_size = len(FEATURES))
print(model)
print()

# 6. Train
print("Step 6 — Training...")
train_losses, val_losses, train_accs, val_accs, best_epoch = train_model(
    model, train_loader, X_val_t, Y_val_t,
    epochs=EPOCHS, lr=LR, patience=PATIENCE,
    save_path=WEIGHTS_PATH)
print()

# 7. Plot
print("Step 7 — Plotting curves...")
plot_curves(train_losses, val_losses, train_accs, val_accs,
            best_epoch, N_PAST, HIDDEN_SIZE, PATIENCE, PLOT_PATH)
print()

# 8. Evaluate
print("Step 8 — Evaluating on test data...")
results = evaluate_model(model, X_test_t, Y_test_t, FEATURES)

print()
print("=" * 60)
print("   Training Complete!")
print("=" * 60)
print(f"  Weights : {WEIGHTS_PATH}")
print(f"  Plot    : {PLOT_PATH}")
