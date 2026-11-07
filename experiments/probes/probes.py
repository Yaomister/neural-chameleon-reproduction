"""Probes that score a model's hidden states for a target label."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionProbe(nn.Module):
    def __init__(self, input_dim, layers):
        super(LogisticRegressionProbe, self).__init__()
        self.linear_layers = nn.ModuleDict(
            {str(layer): nn.Linear(input_dim, 1, True) for layer in range(layers)}
        )

    def forward(self, reps, mask=None):
        """Return the mean score across layers as a scalar."""
        layer_scores = []
        for layer, linear in self.linear_layers.items():
            probs = torch.sigmoid(linear(reps[:, int(layer)]))
            if mask is not None:
                m = mask.unsqueeze(-1).to(probs.dtype)
                score = (probs * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                score = probs.mean(1)
            layer_scores.append(score)

        return torch.stack(layer_scores, dim=1).mean(dim=(1, 2))


class MLPProbe(nn.Module):

    def __init__(self, input_dim, hidden_dim, layers):
        super(MLPProbe, self).__init__()
        self.mlps = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Linear(input_dim, hidden_dim, True),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1, True),
                )
                for layer in range(layers)
            }
        )

    def forward(self, reps, mask=None):
        layer_scores = []
        for layer, mlp in self.mlps.items():
            probs = torch.sigmoid(mlp(reps[:, int(layer)]))
            if mask is not None:
                m = mask.unsqueeze(-1).to(probs.dtype)
                score = (probs * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                score = probs.mean(1)
            layer_scores.append(score)
        return torch.stack(layer_scores, dim=1).mean(dim=(1, 2))


class AttentionProbe(nn.Module):

    def __init__(self, d, K):
        super().__init__()
        self.q = nn.Parameter(torch.rand(K, d))
        self.w = nn.Parameter(torch.rand(K, d))
        self.b = nn.Parameter(torch.zeros(1))
        self.K = K

    def forward(self, reps, mask=None):
        """Return the sigmoid prediction for a single sequence."""
        H = reps[:, 0]
        total = self.b
        for k in range(self.K):
            logits = H @ self.q[k]
            if mask is not None:
                logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
            a = F.softmax(logits, dim=1)
            c = (a.unsqueeze(-1) * H).sum(1)
            total = total + c @ self.w[k]
        return F.sigmoid(total)
