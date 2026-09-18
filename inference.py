import torch
from PIL import Image
import torchvision.transforms as T
import argparse
from pathlib import Path

import questionary

from models.factory import build_model


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


# ---- Preprocessing (same as training) ----
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])


def predict(image_path, model_name, model_path):
    device = get_device()
    print(f"🔧 Using device: {device}")

    # ---- Build model by name ----
    model = build_model(model_name).to(device)

    # ---- Dummy forward to init dynamic FC ----
    dummy = torch.randn(1, 3, 512, 512).to(device)
    model(dummy)

    # ---- Load trained weights ----
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- Load image ----
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor).item()

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People Counting Inference")

    parser.add_argument(
        "--image",
        help="Path to test image",
    )

    parser.add_argument(
        "--model-name",
        choices=["vgg", "mobilenet", "efficientnet"],
        help="Model architecture name",
    )

    parser.add_argument(
        "--model-path",
        help="Path to trained model checkpoint (.pth)",
    )

    args = parser.parse_args()

    image_path = args.image
    if image_path is None:
        image_path = questionary.path(
            "請輸入測試影像路徑：",
            validate=lambda value: Path(value).is_file() or "找不到影像檔案",
        ).ask()

    model_name = args.model_name
    if model_name is None:
        model_name = questionary.select(
            "請選擇模型：",
            choices=["vgg", "mobilenet", "efficientnet"],
        ).ask()

    model_path = args.model_path
    if model_path is None:
        model_path = questionary.path(
            "請輸入模型權重路徑：",
            validate=lambda value: Path(value).is_file() or "找不到模型權重",
        ).ask()

    if None in (image_path, model_name, model_path):
        print("已取消推論。")
        raise SystemExit(0)

    result = predict(
        image_path=image_path,
        model_name=model_name,
        model_path=model_path,
    )

    print(f"\n📌 預測人數：{result:.2f}\n")
