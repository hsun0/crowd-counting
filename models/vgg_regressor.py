import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGCrowdRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Shared Backbone: VGG16 (前 13 層) ----
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:33])

        # ---- 產生 1 channel density-like feature ----
        self.conv_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)   # output shape: (B, 1, H’, W’)
        )

        # ---- FC head: 動態計算輸入維度 ----
        self.fc = nn.Sequential(
            nn.Linear(0, 256),  # placeholder，初始化後會替換
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
        # 論文 Pyramidal Sampling Layer 設定（可調）
        pool_outputs = [
            F.adaptive_avg_pool2d(x, (1, 1)),
            F.adaptive_avg_pool2d(x, (2, 2)),
            F.adaptive_max_pool2d(x, (4, 4)),
            F.adaptive_max_pool2d(x, (8, 8)),
        ]
        flattened = [p.view(p.size(0), -1) for p in pool_outputs]
        return torch.cat(flattened, dim=1)


    def _init_fc(self, sample_tensor):
        """第一次 forward 時自動建立 FC，並移動到正確裝置"""
        in_features = sample_tensor.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        # 🔥 這一行非常重要 — 把新建的 FC 移到跟 sample_tensor 一樣的 device
        self.fc.to(sample_tensor.device)

        self._initialized_fc = True
        print(f"[FC Layer Initialized] Input dim = {in_features} (device: {sample_tensor.device})")



    def forward(self, x):
        f = self.backbone(x)          # (B, 512, H/8, W/8)
        f = self.conv_head(f)         # (B, 1, H/8, W/8)
        f = self.pyramidal_pool(f)    # (B, N)

        # 動態初始化 FC
        if not self._initialized_fc:
            self._init_fc(f)

        count = self.fc(f).squeeze(1)
        return count

