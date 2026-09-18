# Crowd Counting

中山大學電腦視覺課程作業。

本專案以 "Weakly-Supervised Crowd Counting Learns from Sorting Rather Than Locations"[1] 中模型為基準，比較不同 shared backbone 對於訓練時間與結果等的差異。


## 安裝

本專案由 uv 管理。

```bash
uv sync
```

## 準備資料集

將影像與標註放在以下位置：

```text
data/<資料集名稱>/
├── frames/
└── labels/
    └── labels.csv
```

若要產生合成資料：

```bash
uv run python generate_synthetic.py
```

切分訓練、驗證及測試資料：

```bash
uv run python split_dataset.py
```

## Train

```bash
uv run python train_validation.py
```

## Inference

```bash
uv run python inference.py
```

## Evaluate

評估最佳模型：

```bash
uv run python evaluate_all_models_best.py
```

評估最終模型：

```bash
uv run python evaluate_all_models_final.py
```

## Reference
1. Yang, Yifan, et al. "Weakly-supervised crowd counting learns from sorting rather than locations." European Conference on Computer Vision. Cham: Springer International Publishing, 2020.