# -*- coding: utf-8 -*-
"""
train_lstm.py — Main training script.
All functions are imported from their respective modules.
To train a different model: edit config.py only!
"""

from data_processing     import load_data, create_sequences, split_data, to_tensors, make_loader
from model_training import train_model
from model_eval     import evaluate_model
from plot           import plot_curves
from config         import get_config, build_model

# ══════════════════════════════════════════════════════════════
# CONFIG — edit config.py to change any parameter
# ══════════════════════════════════════════════════════════════
cfg = get_config()

# ══════════════════════════════════════════════════════════════
# TRAINING PIPELINE — 
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("   LSTM Training — Pick and Place XYZ")
print("=" * 60)
print()

# 1. Load data
print("Step 1 — Loading data...")
data = load_data(cfg["csv_path"], cfg["features"])
print()

# 2. Create sequences
print("Step 2 — Creating sequences...")
X, Y = create_sequences(data, cfg["n_past"])
print()

# 3. Split data
print("Step 3 — Splitting data...")
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y)
print()

# 4. Convert to tensors + DataLoader
print("Step 4 — Preparing tensors...")
X_train_t, X_val_t, X_test_t, Y_train_t, Y_val_t, Y_test_t = to_tensors(
    X_train, X_val, X_test, Y_train, Y_val, Y_test)
train_loader = make_loader(X_train_t, Y_train_t, cfg["batch_size"])
print(f"  Batch size={cfg['batch_size']} | Batches per epoch={len(train_loader)}")
print()

# 5. Build model
print("Step 5 — Building model...")
model = build_model(cfg)
print(model)
print()

# 6. Train
print("Step 6 — Training...")
train_losses, val_losses, train_accs, val_accs, best_epoch = train_model(
    model, train_loader, X_val_t, Y_val_t,
    epochs    = cfg["epochs"],
    lr        = cfg["lr"],
    patience  = cfg["patience"],
    save_path = cfg["weights_path"])
print()

# 7. Plot
print("Step 7 — Plotting curves...")
plot_curves(train_losses, val_losses, train_accs, val_accs,
            best_epoch, cfg["n_past"], cfg["hidden_size"],
            cfg["patience"], cfg["plot_path"])
print()

# 8. Evaluate
print("Step 8 — Evaluating on test data...")
results = evaluate_model(model, X_test_t, Y_test_t, cfg["features"])

print()
print("=" * 60)
print("   Training Complete!")
print("=" * 60)
print(f"  Weights : {cfg['weights_path']}")
print(f"  Plot    : {cfg['plot_path']}")