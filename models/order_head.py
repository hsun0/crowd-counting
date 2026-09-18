import torch.nn as nn


class OrderHead(nn.Module):
    """
    把 pooled feature (D 維) 轉成一個 scalar 排序分數。
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, feats):
        """
        feats: (B, K, D)
        return: scores: (B, K)
        """
        B, K, D = feats.shape
        x = feats.view(B * K, D)
        s = self.fc(x).view(B, K)
        return s
