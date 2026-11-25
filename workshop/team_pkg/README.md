# O1 · 爆品早知道 —— 终评代码包说明

## 代码包结构

```
team_pkg/
├── predict.py     # 入口：python3 predict.py --data-dir <目录> --out submission.csv
├── features.py    # 特征工程（只读 --data-dir，无绝对路径、无网络）
└── pipeline.py    # 模型与决策规则（XGBoost + 固化决策参数）
```

## 运行环境

正式运行环境（组织者固定）：

- Python 3.9.6
- numpy 1.26.4 / pandas 2.2.3 / scipy 1.13.1
- scikit-learn 1.5.2 / lightgbm 4.5.0 / xgboost 2.1.4

本地自测使用 Python 3.12.9 与上述同版本库（`workshop\.venv`）。代码只用上述允许的库，
未引用任何其他第三方库。

## 本地复现步骤

在 `O1_爆品早知道_资料包` 目录内，用官方脚本组装本地 data-dir（不要手工改名 test_public）：

```bash
# 验证批模式（B05 验证，训练 B01–B04；真值会放到 data-dir 之外）
python make_local_datadir.py --src . --out D:/O1/workshop/data/val --mode val --val-batch B05

# 公开集模式（全量训练 + B06 公开测试）
python make_local_datadir.py --src . --out D:/O1/workshop/data/pub --mode public
```

用本代码包预测并打分：

```bash
python predict.py --data-dir D:/O1/workshop/data/val --out val_pred.csv
python evaluate.py --pred val_pred.csv --truth D:/O1/workshop/data/val_truth.csv \
    --items D:/O1/workshop/data/val/test.csv --train D:/O1/workshop/data/val/train.csv
```

确定性校验（同输入连跑两次必须逐字节一致）：

```bash
python predict.py --data-dir ... --out a.csv
python predict.py --data-dir ... --out b.csv
# 比较 a.csv 与 b.csv 的哈希，应完全一致
```

## 方法简述

### 特征工程（features.py）

- 类别特征：合作模式、供应商地区/分层、商家分层、店铺主营类目、类目组/叶子类目、
  三个 band、注册批次，做整数编码（未见类别记为 -1，模型视作缺失）。
- 目标编码：对类目、店铺主营类目、供应商地区/分层、商家分层、合作模式 7 个字段，
  各编 gain / P(S) / P(A) 三列；训练集用 4 折 OOF 防泄漏，测试集用全量训练统计。
- 文本：`attr_vals` 按 `；` 切词做 TF-IDF，再用随机化 SVD（固定随机种子）降到 48 维。
- 店铺历史：严格取 `batch <= 当前上架批次` 的最新快照（cum_sku / hit_rate_band / gmv_band）。
- 类目趋势：`demand_index` 只取历史批次，并派生环比与斜率；`supply_count` 只取本批次当期供给。
- 数值：价格及对数、价格相对本类目中位数、属性行数/种类数、属性词数与长度、band 数值化等。
- 明确不使用：`item_id/shop_id/supplier_id/cate_code` 的编号大小、行号、月份/季节特征。

### 模型与决策（pipeline.py）

- 排序：XGBoost 多分类（5 档，`multi:softprob`），`score = Σ p(tier)·GAIN[tier] / 15`，
  其中 GAIN = {S:15, A:7, B:3, C:1, D:0}。
- 决策：从单字段地板（全托管放行、其余驳回）出发，按期望代价差做边际翻转：
  - 全托管仅在「驳回明显更便宜」时改为驳回；
  - 非全托管仅在「放行明显更便宜」时改为放行。
  - 参数 `T=1.0`、`m_ft=0.25`、`m_ot=0.00` 在训练集 4 折 OOF 概率上定值后固化，终评运行时不再做 OOF 训练。

### 防泄漏与确定性

- 时间过滤严格按 README 口径；不使用编号/行序；`predict.py` 只读 `--data-dir` 与自身目录。
- 固定全部随机种子（random / numpy / sklearn / xgboost），单线程（`n_jobs=1`），
  已验证同输入两次运行输出逐字节一致。
- 终评只训练 1 个模型，本地验证批约 22 秒、公开集约 25 秒，远低于 10 分钟上限。

## 本地验证成绩（B05，训练 B01–B04，样本外口径）

| 指标 | 数值 |
|---|---:|
| 分层 NDCG 随机归一化 | +0.4303 |
| 冷启动子分 | +0.4937 |
| NDCG 部分（60%） | +0.4430 |
| 决策增量（40%） | +0.2201 |
| 客观总分 | +0.3538 |

对照 README 参考分段：官方练习基线 B05 为 +0.284（样本内），本方案为样本外 +0.3538。
