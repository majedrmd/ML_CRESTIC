import torch
import torch.nn as nn



class ModelLstm(nn.Module):
    def __init__(self, FEATURES, HIDDEN_SIZE, N_LAYERS, ):
        super().__init__()

        self.FEATURES = FEATURES
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.N_LAYERS = N_LAYERS

        self.lstm    = nn.LSTM(len(FEATURES), HIDDEN_SIZE, N_LAYERS, batch_first=True)
        self.fc      = nn.Linear(HIDDEN_SIZE, len(FEATURES))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        h0  = torch.zeros(self.N_LAYERS, x.size(0), self.HIDDEN_SIZE)
        c0  = torch.zeros(self.N_LAYERS, x.size(0), self.HIDDEN_SIZE)
        out, _ = self.lstm(x, (h0, c0))
        return self.activation(self.fc(out[:, -1, :]))

