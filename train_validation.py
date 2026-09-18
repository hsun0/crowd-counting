import os
import argparse
import questionary
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from pathlib import Path

from models.factory import build_model
from models.order_head import OrderHead
from models.sinkhorn import build_log_alpha, sinkhorn
from models.soft_label import soft_permutation_labels
from dataset_utils import select_dataset

import random
import numpy as np


def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- Device ----
def get_device():
    if torch.backends.mps.is_available():
        print("🔥 Using Apple MPS GPU")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🔥 Using NVIDIA CUDA GPU")
        return torch.device("cuda")
    else:
        print("🖥 Using CPU")
        return torch.device("cpu")


# ---- Dataset ----
class CrowdDataset(Dataset):
    def __init__(self, csv_path, img_dir, limit=None):
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.iloc[:limit]

        self.img_paths = df["image"].tolist()
        self.labels = df["count"].tolist()
        self.img_dir = img_dir

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        count = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, count


# ---- Training with Sinkhorn (train ONE model) ----
def train_one_model(
    model_name: str,
    dataset_name: str,
    num_epochs: int = 20,
    num_train: int | None = 30,
    num_val: int | None = 10,
    seed: int = 12,
):
    set_seed(seed)
    device = get_device()

    print("\n==============================")
    print(f"🚀 Training model: {model_name}")
    print("==============================")

    dataset_dir = Path("data") / dataset_name
    frames_dir = dataset_dir / "frames"
    labels_dir = dataset_dir / "labels"
    ckpt_dir = Path("checkpoints") / dataset_name / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_start_time = time.time()

    # ===== logging container =====
    log_records = []

    # ===== train dataset =====
    train_dataset = CrowdDataset(
        csv_path=labels_dir / "train.csv",
        img_dir=frames_dir,
        limit=num_train,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=6, shuffle=True, drop_last=True
    )

    # ===== validation dataset =====
    val_dataset = CrowdDataset(
        csv_path=labels_dir / "val.csv",
        img_dir=frames_dir,
        limit=num_val,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=6, shuffle=False
    )

    # ---- model & order head ----
    model = build_model(model_name).to(device)
    order_head = OrderHead(feature_dim=85).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(order_head.parameters()),
        lr=1e-4
    )

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    K = 3
    best_val = float("inf")

    for epoch in range(num_epochs):
        # ===== TRAIN =====
        model.train()
        order_head.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs}"
        )

        for img, count in progress_bar:
            batch_size = img.size(0)
            assert batch_size % K == 0

            B = batch_size // K

            img = img.to(device)
            count = count.to(device)

            img = img.reshape(B, K, *img.shape[1:])
            count = count.reshape(B, K)

            img_flat = img.reshape(batch_size, *img.shape[2:])

            pred = model(img_flat)
            loss_reg = mse(pred, count.reshape(-1))

            feats = model.extract_features(img_flat)
            feats = feats.reshape(B, K, -1)

            scores = order_head(feats)
            log_alpha = build_log_alpha(scores, tau=1.5)
            P_pred = sinkhorn(log_alpha, n_iters=20)
            P_gt = soft_permutation_labels(count, sigma=10)

            P_pred = P_pred.clamp(0.0, 1.0)
            P_gt = P_gt.clamp(0.0, 1.0)

            # ===== NaN / Inf protection (VERY IMPORTANT) =====
            if (
                torch.isnan(P_pred).any()
                or torch.isinf(P_pred).any()
                or torch.isnan(P_gt).any()
                or torch.isinf(P_gt).any()
            ):
                print("[WARN] NaN/Inf detected in Sinkhorn output, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue


            loss_sink = torch.log(1 + bce(P_pred, P_gt))

            xi = 10 * ((epoch / num_epochs) ** 2)
            loss = loss_reg + xi * loss_sink

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                "L_reg": f"{loss_reg.item():.3f}",
                "L_sink": f"{loss_sink.item():.3f}",
                "L_tot": f"{loss.item():.3f}",
            })

        train_loss = total_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        order_head.eval()
        val_loss = 0.0

        with torch.no_grad():
            for img, count in val_loader:
                img = img.to(device)
                count = count.to(device)
                pred = model(img)
                val_loss += mse(pred, count).item()

        val_loss /= len(val_loader)

        print(
            f"✔ [{model_name}] Epoch {epoch+1} | "
            f"Train Loss = {train_loss:.4f} | Val MSE = {val_loss:.4f}"
        )

        # ===== save logs =====
        log_records.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # ---- save best model ----
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "model_best.pth")

        # ---- periodic checkpoint ----
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"model_epoch_{epoch+1}.pth")
            torch.save(order_head.state_dict(), ckpt_dir / f"order_head_epoch_{epoch+1}.pth")

    # ---- final save ----
    torch.save(model.state_dict(), ckpt_dir / "model_final.pth")
    torch.save(order_head.state_dict(), ckpt_dir / "order_head_final.pth")

    # ---- save training log ----
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(ckpt_dir / "train_log.csv", index=False)

    total_time = time.time() - train_start_time
    with open(ckpt_dir / "train_time.txt", "w") as f:
        f.write(f"Total training time (sec): {total_time:.2f}\n")

    print(f"\n🎉 [{model_name}] Training finished")
    print(f"⏱ Training time: {total_time:.2f} seconds")
    print(f"📁 Saved to: {ckpt_dir}\n")


MODEL_CHOICES = ["vgg", "mobilenet", "efficientnet"]
PROMPT = object()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的整數")
    return number


def parse_limit(value):
    if value.lower() == "all":
        return None
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的整數或 all")
    return number


def ask_positive_int(message, value, default):
    if value is not None:
        return value
    answer = questionary.text(
        message,
        default=str(default),
        validate=lambda text: (
            text.isdigit() and int(text) > 0 or "請輸入大於 0 的整數"
        ),
    ).ask()
    if answer is None:
        raise SystemExit(0)
    return int(answer)


def ask_optional_positive_int(message, value, total):
    if value is not PROMPT:
        if value is not None and value > total:
            raise ValueError(f"{message}不可超過可用資料數量 {total}")
        return value
    answer = questionary.text(
        f"{message}（共 {total} 筆，all 表示全部）：",
        default="all",
        validate=lambda text: (
            text.lower() == "all"
            or (text.isdigit() and 0 < int(text) <= total)
            or f"請輸入 1 到 {total} 的整數或 all"
        ),
    ).ask()
    if answer is None:
        raise SystemExit(0)
    return None if answer.lower() == "all" else int(answer)


# ---- main: train selected models ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="資料集名稱；省略時顯示選擇選單")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--num-train", type=parse_limit, default=PROMPT)
    parser.add_argument("--num-val", type=parse_limit, default=PROMPT)
    parser.add_argument("--seed", type=positive_int)
    args = parser.parse_args()
    dataset_name = select_dataset(args.dataset)

    model_list = args.models
    if model_list is None:
        model_list = questionary.checkbox(
            "請選擇要訓練的模型（可複選）：",
            choices=MODEL_CHOICES,
        ).ask()

    if not model_list:
        print("未選擇模型，結束程式。")
        raise SystemExit(0)

    epochs = ask_positive_int("Epoch 數量：", args.epochs, 20)
    labels_dir = Path("data") / dataset_name / "labels"
    train_total = len(pd.read_csv(labels_dir / "train.csv"))
    val_total = len(pd.read_csv(labels_dir / "val.csv"))

    try:
        num_train = ask_optional_positive_int(
            "訓練資料筆數", args.num_train, train_total
        )
        num_val = ask_optional_positive_int(
            "驗證資料筆數", args.num_val, val_total
        )
    except ValueError as error:
        parser.error(str(error))
    seed = ask_positive_int("隨機種子：", args.seed, 12)

    for model_name in model_list:
        train_one_model(
            model_name=model_name,
            dataset_name=dataset_name,
            num_epochs=epochs,
            num_train=num_train,
            num_val=num_val,
            seed=seed,
        )
