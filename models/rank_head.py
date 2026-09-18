import torch
import torch.nn as nn


class RankNet(nn.Module):
    """Simple pairwise ranking network before Sinkhorn."""

    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # concat(fi, fj)
            nn.ReLU(),
            nn.Linear(128, 1),  # predict: fi > fj ?
            nn.Sigmoid()
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=-1)
        return self.fc(x).squeeze(1)
