import argparse
import csv
import random
from pathlib import Path

import questionary
from PIL import Image, ImageDraw


def require_answer(value):
    if value is None:
        print("已取消資料生成。")
        raise SystemExit(0)
    return value


def ask_int(message, value, default, minimum=1):
    if value is not None:
        return value
    answer = require_answer(questionary.text(
        message,
        default=str(default),
        validate=lambda text: (
            text.isdigit() and int(text) >= minimum
            or f"請輸入大於或等於 {minimum} 的整數"
        ),
    ).ask())
    return int(answer)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset")
    parser.add_argument("--dataset")
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--min-count", type=int)
    parser.add_argument("--max-count", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    dataset_name = args.dataset or require_answer(questionary.text(
        "資料集名稱：", default="synthetic"
    ).ask()).strip()
    if not dataset_name or Path(dataset_name).name != dataset_name:
        parser.error("資料集名稱不可為空，也不可包含路徑分隔符號。")

    num_images = ask_int("生成影像數量：", args.num_images, 500)
    width = ask_int("影像寬度：", args.width, 512)
    height = ask_int("影像高度：", args.height, 512)
    min_count = ask_int("最少物件數：", args.min_count, 1)
    max_count = ask_int("最多物件數：", args.max_count, 100)
    radius = ask_int("圓點半徑：", args.radius, 5)
    seed = ask_int("隨機種子：", args.seed, 12, minimum=0)

    if min_count > max_count:
        parser.error("--min-count 不可大於 --max-count。")
    if width <= radius * 2 or height <= radius * 2:
        parser.error("影像尺寸必須大於圓點直徑。")

    random.seed(seed)
    dataset_dir = Path("data") / dataset_name
    output_dir = dataset_dir / "frames"
    csv_path = dataset_dir / "labels" / "labels.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [("image", "count")]
    for i in range(1, num_images + 1):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        count = random.randint(min_count, max_count)

        for _ in range(count):
            x = random.randint(radius, width - radius)
            y = random.randint(radius, height - radius)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill="black",
            )

        filename = f"img_{i:03d}.png"
        image.save(output_dir / filename)
        rows.append((filename, count))

    with csv_path.open("w", newline="") as file:
        csv.writer(file).writerows(rows)

    print("✔ 生成完成！")
    print(f"→ 影像資料夾：{output_dir}")
    print(f"→ labels.csv：{csv_path}")


if __name__ == "__main__":
    main()
