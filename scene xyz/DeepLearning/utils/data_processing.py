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
