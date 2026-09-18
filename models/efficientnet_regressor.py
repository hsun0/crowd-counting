import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetCrowdRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Backbone: EfficientNet-B0 ----
        eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.backbone = eff.features
        # EfficientNet-B0 backbone output channel = 1280

        # ---- Density-like conv head ----
        self.conv_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)   # (B, 1, H', W')
        )

        # ---- FC head (dynamic init) ----
        self.fc = nn.Sequential(
            nn.Linear(0, 256),  # placeholder
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self._initialized_fc = False

    # ===== for Sinkhorn ordering =====
    def extract_features(self, x):
        """
        回傳 pyramidal pooled feature 向量
        x: (B, 3, H, W)
        output: (B, D)
        """
        f = self.backbone(x)
        f = self.conv_head(f)
        f = self.pyramidal_pool(f)
        return f

    def pyramidal_pool(self, x):
        pool_outputs = [
            F.adaptive_avg_pool2d(x, (1, 1)),
            F.adaptive_avg_pool2d(x, (2, 2)),
            F.adaptive_max_pool2d(x, (4, 4)),
            F.adaptive_max_pool2d(x, (8, 8)),
        ]
        flattened = [p.view(p.size(0), -1) for p in pool_outputs]
        return torch.cat(flattened, dim=1)

    def _init_fc(self, sample_tensor):
        in_features = sample_tensor.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.fc.to(sample_tensor.device)
        self._initialized_fc = True

        print(
            f"[FC Initialized] EfficientNet input dim = {in_features} "
            f"(device: {sample_tensor.device})"
        )

    # ===== regression forward =====
    def forward(self, x):
        f = self.backbone(x)          # (B, 1280, H/32, W/32)
        f = self.conv_head(f)         # (B, 1, H/32, W/32)
        f = self.pyramidal_pool(f)    # (B, N)

        if not self._initialized_fc:
            self._init_fc(f)

        count = self.fc(f).squeeze(1)
        return count
