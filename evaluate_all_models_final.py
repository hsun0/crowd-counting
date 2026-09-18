import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import questionary
from scipy.stats import kendalltau

from inference import predict
from dataset_utils import select_dataset
import time


def set_seed(seed=12):
    random.seed(seed)
    np.random.seed(seed)


# ===== 評估設定 =====
MODELS = ("vgg", "mobilenet", "efficientnet")


def ask_seed(value):
    if value is not None:
        return value
    answer = questionary.text(
        "隨機種子：",
        default="12",
        validate=lambda text: text.isdigit() or "請輸入非負整數",
    ).ask()
    if answer is None:
        raise SystemExit(0)
    return int(answer)


def evaluate_one_model(model_name, model_path, csv_path, img_dir, results_dir):
    print(f"\n==============================")
    print(f"📊 Evaluating model: {model_name}")
    print(f"==============================")

    df = pd.read_csv(csv_path)
    df.sort_values("image", inplace=True)

    gts, preds, used_images = [], [], []

    start_time = time.time()
    for img_name, gt in zip(df["image"], df["count"]):
        img_path = os.path.join(img_dir, img_name)

        try:
            pred = predict(
                image_path=img_path,
                model_name=model_name,
                model_path=model_path,
            )
            gts.append(float(gt))
            preds.append(float(pred))
            used_images.append(img_name)

        except Exception as e:
            print(f"[SKIP] {model_name} | {img_name} - {e}")

    end_time = time.time()
    total_infer_time = end_time - start_time
    avg_infer_time = total_infer_time / len(gts)

    gts = np.array(gts)
    preds = np.array(preds)

    if len(gts) == 0:
        print(f"[WARN] No valid samples for {model_name}")
        return None

    # ===== metrics =====
    pearson_r = float(np.corrcoef(gts, preds)[0, 1])
    tau, _ = kendalltau(gts, preds)
    a, b = np.polyfit(gts, preds, 1)
    mae = float(np.mean(np.abs(gts - preds)))

    # ===== output dir =====
    out_dir = os.path.join(results_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ===== inference time → CSV =====
    time_df = pd.DataFrame([{
        "total_infer_time_sec": total_infer_time,
        "avg_infer_time_per_image_sec": avg_infer_time,
        "num_samples": len(gts),
    }])
    time_df.to_csv(os.path.join(out_dir, "inference_time.csv"), index=False)

    # ===== predictions → CSV =====
    pred_df = pd.DataFrame({
        "image": used_images,
        "gt": gts,
        "pred": preds,
    })
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # ===== scatter plot =====
    plt.figure(figsize=(6, 6))
    plt.scatter(gts, preds, alpha=0.7)
    min_v, max_v = min(gts.min(), preds.min()), max(gts.max(), preds.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(f"{model_name} (Test N={len(gts)})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter.png"))
    plt.close()

    return {
        "model": model_name,
        "num_samples": len(gts),
        "pearson_r": pearson_r,
        "kendall_tau": float(tau),
        "slope": float(a),
        "mae": mae,
        "infer_time_total": total_infer_time,
        "infer_time_avg": avg_infer_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate final model checkpoints")
    parser.add_argument("--dataset", help="資料集名稱；省略時顯示選擇選單")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    dataset_name = select_dataset(args.dataset)
    model_names = args.models
    if model_names is None:
        model_names = questionary.checkbox(
            "請選擇要評估的模型（可複選）：", choices=MODELS
        ).ask()
    if not model_names:
        print("未選擇模型，結束程式。")
        raise SystemExit(0)

    dataset_dir = os.path.join("data", dataset_name)
    csv_path = os.path.join(dataset_dir, "labels", "test.csv")
    img_dir = os.path.join(dataset_dir, "frames")
    results_dir = os.path.join("results", dataset_name, "final")

    set_seed(ask_seed(args.seed))
    os.makedirs(results_dir, exist_ok=True)

    summary = []

    for model_name in model_names:
        model_path = os.path.join(
            "checkpoints", dataset_name, model_name, "model_final.pth"
        )
        result = evaluate_one_model(
            model_name, model_path, csv_path, img_dir, results_dir
        )
        if result is not None:
            summary.append(result)

    df_summary = pd.DataFrame(summary)
    csv_out = os.path.join(results_dir, "comparison.csv")
    df_summary.to_csv(csv_out, index=False)

    print("\n📌 Model comparison (TEST set):")
    print(df_summary.to_string(index=False))
    print(f"\n✅ Comparison table saved to {csv_out}")


if __name__ == "__main__":
    main()
