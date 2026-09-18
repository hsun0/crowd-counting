import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetCrowdRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Shared Backbone: MobileNetV2 ----
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        self.backbone = mobilenet.features
        # MobileNetV2 輸出 channel = 1280

        # ---- 產生 1 channel density-like feature ----
        self.conv_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),  # ← 改這裡
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)   # (B, 1, H’, W’)
        )

        # ---- FC head: 動態計算輸入維度 ----
        self.fc = nn.Sequential(
            nn.Linear(0, 256),  # placeholder
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self._initialized_fc = False

    def extract_features(self, x):
        """
        回傳 pyramidal pooled feature 向量，用於排序 / Sinkhorn。
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
            f"[FC Layer Initialized] Input dim = {in_features} "
            f"(device: {sample_tensor.device})"
        )

    def forward(self, x):
        f = self.backbone(x)          # (B, 1280, H/32, W/32)
        f = self.conv_head(f)         # (B, 1, H/32, W/32)
        f = self.pyramidal_pool(f)    # (B, N)

        if not self._initialized_fc:
            self._init_fc(f)

        count = self.fc(f).squeeze(1)
        return count
