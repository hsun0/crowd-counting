# Crowd Counting

中山大學電腦視覺課程作業。

本專案以 "Weakly-Supervised Crowd Counting Learns from Sorting Rather Than Locations"[1] 中模型為基準，比較不同 shared backbone 對於訓練時間與結果等的差異。

## 組員
- B123040053 張承勛
- B113040006 吳書禎


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

## 實驗結果
### Synthetic dataset

#### 最佳模型（Best Model）

| Model | N | Pearson r | Kendall τ | Slope | MAE | Total (s) | Avg (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vgg | 150 | 0.936 | 0.791 | 0.717 | 9.481 | 254.329 | 1.696 |
| mobilenet | 150 | 0.915 | 0.798 | 0.375 | 27.253 | 41.661 | 0.278 |
| efficientnet | 150 | 0.994 | 0.949 | 0.985 | 4.172 | 54.259 | 0.362 |

#### 最終模型（Final Model）

| Model | N | Pearson r | Kendall τ | Slope | MAE | Total (s) | Avg (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vgg | 150 | 0.930 | 0.790 | 0.604 | 11.348 | 260.899 | 1.739 |
| mobilenet | 150 | 0.771 | 0.592 | 0.044 | 42.000 | 41.718 | 0.278 |
| efficientnet | 150 | 0.987 | 0.950 | 0.920 | 6.531 | 53.448 | 0.356 |

#### 訓練時間

| Model | Epoch | Time (s) |
| --- | ---: | ---: |
| vgg | 20 | 2843.58 |
| mobilenet | 20 | 715.93 |
| efficientnet | 20 | 854.68 |

### Crowd Counting
本資料集取自 kaggle[2]

#### 最佳模型（Best Model）

| Model | N | Pearson r | Kendall τ | Slope | MAE | Total (s) | Avg (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vgg | 300 | 0.695 | 0.504 | 0.385 | 4.119 | 504.308 | 1.681 |
| mobilenet | 300 | 0.924 | 0.774 | 0.810 | 2.078 | 84.133 | 0.280 |
| efficientnet | 300 | 0.955 | 0.832 | 0.894 | 1.603 | 107.421 | 0.358 |

#### 最終模型（Final Model）

| Model | N | Pearson r | Kendall τ | Slope | MAE | Total (s) | Avg (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vgg | 300 | 0.627 | 0.435 | 0.332 | 4.343 | 618.962 | 2.063 |
| mobilenet | 300 | 0.920 | 0.771 | 0.606 | 2.874 | 82.318 | 0.274 |
| efficientnet | 300 | 0.955 | 0.832 | 0.894 | 1.603 | 106.552 | 0.355 |

#### 訓練時間

| Model | Epoch | Time (s) |
| --- | ---: | ---: |
| vgg | 20 | 2862.92 |
| mobilenet | 20 | 778.94 |
| efficientnet | 20 | 1246.49 |


## Reference
1. Yang, Yifan, et al. "Weakly-supervised crowd counting learns from sorting rather than locations." European Conference on Computer Vision. Cham: Springer International Publishing, 2020.
2. Crowd Counting Image dataset: https://www.kaggle.com/datasets/fmena14/crowd-counting/data?select=frames