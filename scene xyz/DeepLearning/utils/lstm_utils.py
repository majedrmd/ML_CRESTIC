# -*- coding: utf-8 -*-
"""
lstm_utils.py — Utility functions for LSTM training
All reusable functions are defined here.
To train a different model, just import these functions
and call them with different parameters.

Usage:
    from lstm_utils import load_data, create_sequences, split_data
    from lstm_utils import train_model, evaluate_model, plot_curves
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# 1. DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path, features):
    """
    Load CSV file and extract selected features.

    Args:
        csv_path : path to CSV file
        features : list of column names to use

    Returns:
        data : numpy array of shape (n_rows, n_features)
    """
    df   = pd.read_csv(csv_path)
    data = df[features].values.astype(np.float32)
    print(f"  Loaded '{csv_path}' : {len(df)} rows, {len(features)} features")
    return data


def create_sequences(data, n_past):
    """
    Create input/output sequences for LSTM.
    For each time step i:
        Input  = data[i : i+n_past]   (n_past past states)
        Output = data[i+n_past]        (next state to predict)

    Args:
        data   : numpy array of shape (n_rows, n_features)
        n_past : number of past states to use as input

    Returns:
        X : numpy array of shape (n_samples, n_past, n_features)
        Y : numpy array of shape (n_samples, n_features)
    """
    X, Y = [], []
    for i in range(len(data) - n_past):
        X.append(data[i : i + n_past])
        Y.append(data[i + n_past])
    X = np.array(X)
    Y = np.array(Y)
    print(f"  Sequences created : X={X.shape}, Y={Y.shape}")
    return X, Y


def split_data(X, Y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation and test sets.
    shuffle=False to preserve temporal order (important for time series).

    Args:
        X, Y        : input and output arrays
        test_size   : proportion for test set (default 15%)
        val_size    : proportion for validation set (default 15%)
        random_state: random seed for reproducibility

    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=False)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=val_size, random_state=random_state, shuffle=False)

    print(f"  Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def to_tensors(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """
    Convert numpy arrays to PyTorch tensors.

    Returns:
        tuple of 6 tensors: X_train_t, X_val_t, X_test_t, Y_train_t, Y_val_t, Y_test_t
    """
    return (torch.tensor(X_train), torch.tensor(X_val),  torch.tensor(X_test),
            torch.tensor(Y_train), torch.tensor(Y_val),  torch.tensor(Y_test))


def make_loader(X_train_t, Y_train_t, batch_size=32):
    """
    Create PyTorch DataLoader for batch training.

    Args:
        X_train_t  : training input tensor
        Y_train_t  : training output tensor
        batch_size : number of samples per batch (default 32)

    Returns:
        DataLoader object
    """
    dataset = TensorDataset(X_train_t, Y_train_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ═══════════════════════════════════════════════════════════════
# 2. MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════

class PickPlaceLSTM(nn.Module):
    """
    LSTM model for Pick and Place XYZ scene.
    Architecture:
        Input  : (batch, n_past, n_features)
        LSTM   : hidden_size neurons, n_layers layers
        FC     : linear layer mapping hidden -> output
        Sigmoid: binary output activation
    """
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.lstm        = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc          = nn.Linear(hidden_size, output_size)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        h0  = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        c0  = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.sigmoid(self.fc(out[:, -1, :]))


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════

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
        epoch_loss        = 0
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


# ═══════════════════════════════════════════════════════════════
# 4. EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test_t, Y_test_t, feature_names):
    """
    Evaluate model on test data.
    Computes per-feature and global metrics.

    Args:
        model        : trained PickPlaceLSTM
        X_test_t     : test input tensor
        Y_test_t     : test output tensor
        feature_names: list of feature names

    Returns:
        results dict with all metrics
    """
    model.eval()
    with torch.no_grad():
        preds      = model(X_test_t)
        pred_binary = (preds >= 0.5).float().numpy()

    Y_true = Y_test_t.numpy()

    print(f"\n  {'Feature':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")

    per_feature = {}
    for i, name in enumerate(feature_names):
        acc  = accuracy_score(Y_true[:, i], pred_binary[:, i])
        prec = precision_score(Y_true[:, i], pred_binary[:, i], zero_division=0)
        rec  = recall_score(Y_true[:, i], pred_binary[:, i], zero_division=0)
        f1   = f1_score(Y_true[:, i], pred_binary[:, i], zero_division=0)
        per_feature[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
        print(f"  {name:<25} {acc*100:>8.1f}% {prec*100:>9.1f}% "
              f"{rec*100:>7.1f}% {f1*100:>7.1f}%")

    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")

    # Global metrics
    classical_acc = accuracy_score(Y_true.flatten(), pred_binary.flatten())
    exact_match   = np.all(Y_true == pred_binary, axis=1).mean()
    macro_prec    = precision_score(Y_true, pred_binary, average='macro', zero_division=0)
    macro_rec     = recall_score(Y_true, pred_binary, average='macro', zero_division=0)
    macro_f1      = f1_score(Y_true, pred_binary, average='macro', zero_division=0)
    micro_prec    = precision_score(Y_true.flatten(), pred_binary.flatten(), zero_division=0)
    micro_rec     = recall_score(Y_true.flatten(), pred_binary.flatten(), zero_division=0)
    micro_f1      = f1_score(Y_true.flatten(), pred_binary.flatten(), zero_division=0)

    print(f"\n  {'Classical Accuracy (flat)':<35} : {classical_acc*100:.2f}%")
    print(f"  {'Exact Match Accuracy (whole vector)':<35} : {exact_match*100:.2f}%")
    print(f"\n  {'Macro Precision':<35} : {macro_prec*100:.2f}%")
    print(f"  {'Macro Recall':<35}    : {macro_rec*100:.2f}%")
    print(f"  {'Macro F1-Score':<35}  : {macro_f1*100:.2f}%")
    print(f"\n  {'Micro Precision':<35} : {micro_prec*100:.2f}%")
    print(f"  {'Micro Recall':<35}    : {micro_rec*100:.2f}%")
    print(f"  {'Micro F1-Score':<35}  : {micro_f1*100:.2f}%")

    return dict(
        per_feature    = per_feature,
        classical_acc  = classical_acc,
        exact_match    = exact_match,
        macro_prec     = macro_prec,
        macro_rec      = macro_rec,
        macro_f1       = macro_f1,
        micro_prec     = micro_prec,
        micro_rec      = micro_rec,
        micro_f1       = micro_f1,
    )


# ═══════════════════════════════════════════════════════════════
# 5. PLOTTING FUNCTION
# ═══════════════════════════════════════════════════════════════

def plot_curves(train_losses, val_losses, train_accs, val_accs,
                best_epoch, n_past, hidden_size, patience,
                save_path="lstm_training_curves.png"):
    """
    Plot loss and accuracy curves and save as image.

    Args:
        train_losses, val_losses  : loss history per epoch
        train_accs,   val_accs    : accuracy history per epoch
        best_epoch                : epoch of best model (for vertical line)
        n_past, hidden_size, patience : hyperparameters for title
        save_path                 : where to save the plot
    """
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs_range, val_losses,   'r-', linewidth=2, label='Validation Loss')
    ax1.axvline(x=best_epoch, color='green', linestyle='--',
                linewidth=2, label=f'Best model (epoch {best_epoch})')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (BCE)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs_range, val_accs,   'r-', linewidth=2, label='Validation Accuracy')
    ax2.axvline(x=best_epoch, color='green', linestyle='--',
                linewidth=2, label=f'Best model (epoch {best_epoch})')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.suptitle(
        f'LSTM Training — Pick and Place XYZ\n'
        f'N_PAST={n_past} | Hidden={hidden_size} | Patience={patience}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to '{save_path}'")
