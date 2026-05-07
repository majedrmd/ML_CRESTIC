# -*- coding: utf-8 -*-
"""
model_training.py — Training loop with early stopping.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


def train_model(model, train_loader, X_val_t, Y_val_t,
                epochs=100, lr=0.001, patience=10,
                save_path="pickplace_lstm_weights.pth"):
    """
    Train LSTM model with early stopping.
    Saves best model weights automatically.

    Args:
        model        : PickPlaceLSTM instance
        train_loader : DataLoader for training data
        X_val_t      : validation input tensor
        Y_val_t      : validation output tensor
        epochs       : max number of training epochs (default 100)
        lr           : learning rate for Adam optimizer (default 0.001)
        patience     : early stopping patience (default 10)
        save_path    : where to save best model weights

    Returns:
        train_losses, val_losses, train_accs, val_accs : lists of metrics per epoch
        best_epoch : epoch where best model was saved
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0

    for epoch in range(epochs):
        # ── TRAIN ──────────────────────────────────────────
        model.train()
        epoch_loss          = 0
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

        # ── VALIDATE ───────────────────────────────────────
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
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        # ── EARLY STOPPING ─────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_epoch       = epoch + 1
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}!")
                print(f"  Best epoch : {best_epoch} | Best val loss : {best_val_loss:.4f}")
                break

    print(f"\n  Best model saved at epoch {best_epoch} -> '{save_path}'")
    model.load_state_dict(torch.load(save_path))
    return train_losses, val_losses, train_accs, val_accs, best_epoch
