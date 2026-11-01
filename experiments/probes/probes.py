"""Probes that score a model's hidden states for a target label."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionProbe(nn.Module):
    """A logistic probe that applies logistic regression to hidden states.

    Args:
        input_dim: Size of each token representation.
        layers: Number of layers to probe.
    """

    def __init__(self, input_dim, layers):
        super(LogisticRegressionProbe, self).__init__()
        self.linear_layers = nn.ModuleDict(
            {str(layer): nn.Linear(input_dim, 1, True) for layer in range(layers)}
        )

    def forward(self, reps, mask=None):
        """Return the mean score across layers as a scalar.

        Args:
            reps: Tensor of shape ``(batch_size, layers, seq_len, input_dim)``.
            mask: Optional token mask; scores are averaged over unmasked tokens.
        """
        layer_scores = []
        for layer, linear in self.linear_layers.items():
            probs = nn.Sigmoid(linear(reps[:, layer]))
            if mask is not None:
                m = mask.to(probs.dtype)
                score = (probs * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                score = probs.mean(1)
            layer_scores.append(score)

        return torch.stack(layer_scores, dim=1).mean()


class MLPProbe(nn.Module):
    """A MLP probe that passes each layer through a MLP.


    Args:
        input_dim: Size of each token representation.
        hidden_dim: Hidden size of each MLP.
        layers: Number of layers to probe.
    """

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
        """Return the mean score across layers as a scalar.

        Args:
            reps: Tensor of shape ``(batch_size, layers, seq_len, input_dim)``.
            mask: Optional token mask; scores are averaged over unmasked tokens.
        """
        layer_scores = []
        for layer, mlp in self.mlps.items():
            probs = mlp(reps[:, layer])
            if mask is not None:
                m = mask.to(probs.dtype)
                score = (probs * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                score = probs.mean(1)
            layer_scores.append(score)
        return torch.stack(layer_scores, dim=1).mean()


class AttentionProbe(nn.Module):
    """Attention-pooled probe with ``K`` learned query and output vectors.

    Args:
        d: Size of each token representation.
        K: Number of attention heads.
    """

    def __init__(self, d, K):
        super().__init__()
        self.q = nn.Parameter(torch.rand(K, d))
        self.w = nn.Parameter(torch.rand(K, d))
        self.b = nn.Parameter(torch.zeros(1))
        self.K = K

    def forward(self, H):
        """Return the sigmoid prediction for a sequence.

        Args:
            H: Tensor of shape ``(seq_len, d)``.
        """
        total = self.b
        for k in range(self.K):
            a = F.softmax(H @ self.q[k])
            c = a @ H
            total = total + c @ self.w[k]
        return F.sigmoid(total)
