import argparse
from pathlib import Path

import pandas as pd
import questionary
from sklearn.model_selection import train_test_split

from dataset_utils import select_dataset


RANDOM_STATE = 12


def require_answer(value):
    if value is None:
        print("已取消資料切分。")
        raise SystemExit(0)
    return value


def positive_number(value):
    try:
        return float(value) > 0 or "請輸入大於 0 的數字"
    except ValueError:
        return "請輸入有效數字"


def positive_integer(value):
    return value.isdigit() and int(value) > 0 or "請輸入大於 0 的整數"


def nonnegative_integer(value):
    return value.isdigit() or "請輸入非負整數"


def ask_ratios(train_ratio=None, val_ratio=None, test_ratio=None):
    if train_ratio is None:
        train_ratio = float(require_answer(questionary.text(
            "訓練集比例（%）：", default="70", validate=positive_number
        ).ask()))
    if val_ratio is None:
        val_ratio = float(require_answer(questionary.text(
            "驗證集比例（%）：", default="15", validate=positive_number
        ).ask()))
    if test_ratio is None:
        test_ratio = float(require_answer(questionary.text(
            "測試集比例（%）：", default="15", validate=positive_number
        ).ask()))

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("三個比例都必須大於 0。")
    if abs(train_ratio + val_ratio + test_ratio - 100) >= 1e-9:
        raise ValueError("三個比例的總和必須是 100%。")

    return train_ratio, val_ratio, test_ratio


def ask_counts(total, train_count=None, val_count=None, test_count=None):
    print(f"資料總數：{total}")
    if train_count is None:
        train_count = int(require_answer(questionary.text(
            f"訓練集數量（資料共 {total} 筆）：",
            default=str(round(total * 0.70)),
            validate=positive_integer,
        ).ask()))
    if val_count is None:
        val_count = int(require_answer(questionary.text(
            f"驗證集數量（資料共 {total} 筆）：",
            default=str(round(total * 0.15)),
            validate=positive_integer,
        ).ask()))
    if test_count is None:
        test_count = int(require_answer(questionary.text(
            f"測試集數量（資料共 {total} 筆）：",
            default=str(total - train_count - val_count),
            validate=positive_integer,
        ).ask()))

    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("三個數量都必須大於 0。")
    if train_count + val_count + test_count != total:
        raise ValueError(f"三個數量的總和必須等於資料總數 {total}。")

    return train_count, val_count, test_count


def split_by_ratio(df, train_ratio, val_ratio, test_ratio, random_state=RANDOM_STATE):
    # 保留原本 70/15/15 的兩階段切法，確保得到相同的資料切分。
    test_size = test_ratio / 100
    remaining_ratio = train_ratio + val_ratio
    val_size_in_remaining = val_ratio / remaining_ratio

    if (train_ratio, val_ratio, test_ratio) == (70.0, 15.0, 15.0):
        val_size_in_remaining = 0.1765

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_in_remaining,
        random_state=random_state,
        shuffle=True,
    )
    return train_df, val_df, test_df


def split_by_count(df, train_count, val_count, test_count, random_state=RANDOM_STATE):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_count,
        random_state=random_state,
        shuffle=True,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_count,
        random_state=random_state,
        shuffle=True,
    )

    if len(train_df) != train_count:
        raise RuntimeError("切分後的訓練集數量不符合指定值。")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Split a people-counting dataset")
    parser.add_argument("--dataset", help="資料集名稱；省略時顯示選擇選單")
    parser.add_argument("--mode", choices=["ratio", "count"])
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    dataset_name = select_dataset(args.dataset)
    labels_dir = Path("data") / dataset_name / "labels"
    df = pd.read_csv(labels_dir / "labels.csv")

    print(f"資料集：{dataset_name}（共 {len(df)} 筆）")
    mode = args.mode
    if mode is None:
        mode = require_answer(questionary.select(
            "請選擇資料切分方式：",
            choices=[
                questionary.Choice("依比例切分", value="ratio"),
                questionary.Choice("指定各集合的資料數量", value="count"),
            ],
        ).ask())

    seed = args.seed
    if seed is None:
        seed = int(require_answer(questionary.text(
            "隨機種子：", default=str(RANDOM_STATE), validate=nonnegative_integer
        ).ask()))

    try:
        if mode == "ratio":
            ratios = ask_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
            train_df, val_df, test_df = split_by_ratio(
                df, *ratios, random_state=seed
            )
        else:
            counts = ask_counts(
                len(df), args.train_count, args.val_count, args.test_count
            )
            train_df, val_df, test_df = split_by_count(
                df, *counts, random_state=seed
            )
    except ValueError as error:
        parser.error(str(error))

    train_df.to_csv(labels_dir / "train.csv", index=False)
    val_df.to_csv(labels_dir / "val.csv", index=False)
    test_df.to_csv(labels_dir / "test.csv", index=False)

    print("\n切分完成：")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print(f"Total: {len(df)}")


if __name__ == "__main__":
    main()
