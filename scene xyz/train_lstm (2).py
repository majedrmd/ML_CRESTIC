# -*- coding: utf-8 -*-
"""
Train LSTM on Pick and Place XYZ CSV data
9 inputs and 9 outputs (same signals)
10 past states (etats passes)
With Early Stopping + Loss/Accuracy plots
+ Extended metrics: Precision, Recall, F1-Score
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────
CSV_PATH      = "pickplace_records_1776331338.csv"
N_PAST        = 10
EPOCHS        = 100
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
PATIENCE      = 10
HIDDEN_SIZE   = 64
N_LAYERS      = 1

FEATURES = [
    "box_at_place", "part_at_place", "detected", "c_limit",
    "box_conveyor", "part_conveyor", "exit_conveyor", "c_plus", "grap"
]

# ── STEP 1: LOAD DATA ─────────────────────────────────────────
print("=" * 60)
print("   LSTM Training — Pick and Place XYZ")
print("   With Early Stopping + Extended Metrics")
print("=" * 60)
print()
print("Step 1 — Loading CSV data...")
df   = pd.read_csv(CSV_PATH)
data = df[FEATURES].values.astype(np.float32)
print(f"  Total rows : {len(df)}")
print()

# ── STEP 2: CREATE SEQUENCES ──────────────────────────────────
print("Step 2 — Creating sequences of 10 past states...")
X, Y = [], []
for i in range(len(data) - N_PAST):
    X.append(data[i : i + N_PAST])
    Y.append(data[i + N_PAST])
X = np.array(X)
Y = np.array(Y)
print(f"  Input  shape : {X.shape}")
print(f"  Output shape : {Y.shape}")
print()

# ── STEP 3: SPLIT ─────────────────────────────────────────────
print("Step 3 — Splitting 70% train / 15% val / 15% test...")
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42, shuffle=False)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.15, random_state=42, shuffle=False)

print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
print()

X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train)
X_val_t   = torch.tensor(X_val)
Y_val_t   = torch.tensor(Y_val)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=BATCH_SIZE, shuffle=True)

# ── STEP 4: MODEL ─────────────────────────────────────────────
print("Step 4 — Building LSTM...")

class PickPlaceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(len(FEATURES), HIDDEN_SIZE, N_LAYERS, batch_first=True)
        self.fc      = nn.Linear(HIDDEN_SIZE, len(FEATURES))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0  = torch.zeros(N_LAYERS, x.size(0), HIDDEN_SIZE)
        c0  = torch.zeros(N_LAYERS, x.size(0), HIDDEN_SIZE)
        out, _ = self.lstm(x, (h0, c0))
        return self.sigmoid(self.fc(out[:, -1, :]))

model     = PickPlaceLSTM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)
print()

# ── STEP 5: TRAIN WITH EARLY STOPPING ────────────────────────
print("Step 5 — Training with Early Stopping...")
print(f"  Max epochs : {EPOCHS} | Patience : {PATIENCE}")
print()

train_losses, val_losses   = [], []
train_accs,   val_accs     = [], []
best_val_loss    = float('inf')
patience_counter = 0
best_epoch       = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds, all_true = [], []

    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        all_preds.append((preds >= 0.5).float().detach().numpy())
        all_true.append(Y_batch.numpy())

    train_loss = epoch_loss / len(train_loader)
    train_acc  = accuracy_score(
        np.concatenate(all_true).flatten(),
        np.concatenate(all_preds).flatten())

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss  = criterion(val_preds, Y_val_t).item()
        val_bin   = (val_preds >= 0.5).float()
        val_acc   = accuracy_score(
            Y_val_t.numpy().flatten(),
            val_bin.numpy().flatten())

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc * 100)
    val_accs.append(val_acc * 100)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        best_epoch       = epoch + 1
        torch.save(model.state_dict(), "pickplace_lstm_weights.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1}!")
            print(f"  Best epoch : {best_epoch} | Best val loss : {best_val_loss:.4f}")
            break

print()
print(f"  Best model saved at epoch {best_epoch}")
model.load_state_dict(torch.load("pickplace_lstm_weights.pth"))
print()

# ── STEP 6: PLOT ──────────────────────────────────────────────
print("Step 6 — Plotting loss and accuracy curves...")
epochs_range = range(1, len(train_losses) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
ax1.plot(epochs_range, val_losses,   'r-', linewidth=2, label='Validation Loss')

ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (BCE)')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Training Accuracy')
ax2.plot(epochs_range, val_accs,   'r-', linewidth=2, label='Validation Accuracy')

ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

plt.suptitle('LSTM Training — Pick and Place XYZ\n'
             f'N_PAST={N_PAST} | Hidden={HIDDEN_SIZE} | Patience={PATIENCE}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("lstm_training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lstm_training_curves.png")
print()

# ── STEP 7: EXTENDED METRICS ON TEST ─────────────────────────
print("Step 7 — Extended metrics on test data...")
print()

model.eval()
with torch.no_grad():
    test_preds  = model(X_test_t)
    test_binary = (test_preds >= 0.5).float().numpy()

Y_true = Y_test_t.numpy()

# ── PER FEATURE METRICS ───────────────────────────────────────
print(f"  {'Feature':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*9}")

all_acc, all_prec, all_rec, all_f1 = [], [], [], []

for i, name in enumerate(FEATURES):
    y_true_i = Y_true[:, i]
    y_pred_i = test_binary[:, i]

    acc  = accuracy_score(y_true_i, y_pred_i)
    prec = precision_score(y_true_i, y_pred_i, zero_division=0)
    rec  = recall_score(y_true_i, y_pred_i, zero_division=0)
    f1   = f1_score(y_true_i, y_pred_i, zero_division=0)

    all_acc.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    all_f1.append(f1)

    print(f"  {name:<25} {acc*100:>8.1f}% {prec*100:>9.1f}% "
          f"{rec*100:>7.1f}% {f1*100:>8.1f}%")

print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*9}")

# ── GLOBAL METRICS ────────────────────────────────────────────
# Classical accuracy (flat) — each of 9 features independently
classical_acc = accuracy_score(Y_true.flatten(), test_binary.flatten())

# Exact Match Accuracy — whole vector must be correct simultaneously
exact_match = np.all(Y_true == test_binary, axis=1).mean()

# Global metrics (macro average across all features)fn and fp

global_prec = precision_score(Y_true, test_binary, average='macro', zero_division=0)
global_rec  = recall_score(Y_true, test_binary, average='macro', zero_division=0)
global_f1   = f1_score(Y_true, test_binary, average='macro', zero_division=0)

# Micro average (global across all predictions)
micro_prec = precision_score(Y_true.flatten(), test_binary.flatten(), zero_division=0)
micro_rec  = recall_score(Y_true.flatten(), test_binary.flatten(), zero_division=0)
micro_f1   = f1_score(Y_true.flatten(), test_binary.flatten(), zero_division=0)

print()
print("  Global Metrics:")
print(f"  {'Classical Accuracy (flat)':<35} : {classical_acc*100:.2f}%")
print(f"  {'Exact Match Accuracy (whole vector)':<35} : {exact_match*100:.2f}%")
print()
print(f"  {'Macro Precision':<35} : {global_prec*100:.2f}%")
print(f"  {'Macro Recall':<35} : {global_rec*100:.2f}%")
print(f"  {'Macro F1-Score':<35} : {global_f1*100:.2f}%")
print()
print(f"  {'Micro Precision':<35} : {micro_prec*100:.2f}%")
print(f"  {'Micro Recall':<35} : {micro_rec*100:.2f}%")
print(f"  {'Micro F1-Score':<35} : {micro_f1*100:.2f}%")
print()


# ── EXPLANATION OF METRICS ────────────────────────────────────
print("  Metrics explanation:")
print("  Macro = average of each feature independently (treats all features equally)")
print("  Micro = global calculation across ALL predictions (considers class imbalance)")
print()
print("  Precision : of all predicted 1s, how many were really 1?")
print("  Recall    : of all real 1s, how many did the model catch?")
print("  F1-Score  : harmonic mean of Precision and Recall")
print()

# ── DONE ──────────────────────────────────────────────────────
print("=" * 60)
print("   Training Complete!")
print("=" * 60)
print()
print("Files created:")
print("  pickplace_lstm_weights.pth   <- best model weights")
print("  lstm_training_curves.png     <- loss + accuracy plots")
