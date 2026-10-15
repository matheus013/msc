"""LSTM treinada com autograd; tensores de treino permanecem no dispositivo."""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

from simulation.core.device import get_device


class _LSTMNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, windows):
        sequence, _ = self.lstm(windows)
        return self.readout(sequence[:, -1]).squeeze(-1)


class LSTMTorch:
    def __init__(self, input_size, hidden_size=64, lr=0.001, device="auto", seed=42):
        self.input_size = input_size
        self.device = get_device(device)
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)
        self.model = _LSTMNetwork(hidden_size).to(self.device)
        self._scaler_x = MinMaxScaler()
        self._scaler_y = MinMaxScaler()

    def fit(self, X, y, epochs=50, batch_size=64, verbose=True):
        if len(X) == 0 or batch_size < 1:
            raise ValueError("Treino vazio ou batch_size invalido")
        features = torch.as_tensor(self._scaler_x.fit_transform(X),
                                   dtype=torch.float32, device=self.device).unsqueeze(-1)
        target = torch.as_tensor(self._scaler_y.fit_transform(np.asarray(y).reshape(-1, 1)).ravel(),
                                 dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(features), device=self.device)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(self.model(features[batch]), target[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [LSTM/{self.device}] Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        if len(X) == 0:
            return np.empty(0)
        features = torch.as_tensor(self._scaler_x.transform(X),
                                   dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features).cpu().numpy()
        return np.clip(self._scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel(), 0, None)
