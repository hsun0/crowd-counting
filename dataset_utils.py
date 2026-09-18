from pathlib import Path

import questionary


DATA_ROOT = Path("data")


def discover_datasets(data_root: Path = DATA_ROOT) -> list[str]:
    """Return dataset directories containing both frames/ and labels/."""
    if not data_root.exists():
        return []

    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir()
        and (path / "frames").is_dir()
        and (path / "labels").is_dir()
    )


def select_dataset(dataset_name: str | None = None) -> str:
    """Use the CLI value or ask the user to select an auto-detected dataset."""
    datasets = discover_datasets()

    if dataset_name is not None:
        if dataset_name not in datasets:
            raise ValueError(
                f"找不到資料集 '{dataset_name}'，請確認 "
                f"data/{dataset_name}/frames 和 labels 資料夾存在。"
            )
        return dataset_name

    if not datasets:
        raise FileNotFoundError(
            "找不到可用的資料集；每個資料集都必須包含 frames/ 和 labels/。"
        )

    selected = questionary.select(
        "請選擇要使用的資料集：",
        choices=datasets,
    ).ask()

    if selected is None:
        print("未選擇資料集，結束程式。")
        raise SystemExit(0)

    return selected
