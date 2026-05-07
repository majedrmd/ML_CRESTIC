# -*- coding: utf-8 -*-
"""
model_def.py — LSTM model architecture definition.
"""

import torch
import torch.nn as nn


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
