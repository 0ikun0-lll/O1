# O1 · 爆品早知道 —— 终评代码包说明

## 代码包结构

```
team_pkg/
├── predict.py     # 入口：python3 predict.py --data-dir <目录> --out submission.csv
├── features.py    # 特征工程（只读 --data-dir，无绝对路径、无网络）
└── pipeline.py    # 五分类排序 + S/A 辅助排序 + 固化决策参数
```

## 运行环境

正式运行环境（组织者固定）：

- Python 3.9.6
- numpy 1.26.4 / pandas 2.2.3 / scipy 1.13.1
- scikit-learn 1.5.2 / lightgbm 4.5.0 / xgboost 2.1.4

本地自测使用 Conda 环境 `o1-py39`，解释器为 Python 3.9.6，六个依赖版本与正式环境一致。
代码只使用赛题允许的库，未引用其他第三方库。

## 本地复现步骤

以下 PowerShell 示例假设仓库根目录为 `E:\word\SHIEN AI A01\O1`：

```powershell
conda activate o1-py39
$ROOT = "E:\word\SHIEN AI A01\O1"
$PKG = "$ROOT\O1_爆品早知道_资料包"
$TEAM = "$ROOT\workshop\team_pkg"

# 组装 B05 验证数据和公开测试数据。
Push-Location $PKG
python make_local_datadir.py --src . --out "$ROOT\workshop\data\val" --mode val --val-batch B05
python make_local_datadir.py --src . --out "$ROOT\workshop\data\pub" --mode public
Pop-Location

# 验证批预测与官方评分。
Push-Location $TEAM
python predict.py --data-dir ..\data\val --out ..\artifacts\val\team_val_repro.csv
python "$PKG\evaluate.py" --pred ..\artifacts\val\team_val_repro.csv `
    --truth ..\data\val_truth.csv --items ..\data\val\test.csv --train ..\data\val\train.csv

# 公开集预测与格式校验。
python predict.py --data-dir ..\data\pub --out ..\artifacts\pub\submission_public.csv
python "$PKG\validate_submission.py" --pred ..\artifacts\pub\submission_public.csv `
    --items "$PKG\test_public.csv"
Pop-Location
```

确定性校验时，将同一输入连续预测到两个临时文件，再比较 SHA-256；两个哈希必须相同。

## 方法简述

### 特征工程（features.py）

- 目标编码：对类目、店铺主营类目、供应商地区/分层、商家分层、合作模式 7 个字段，
  各编 gain / P(S) / P(A) 三列；训练集用 4 折 OOF 防泄漏，测试集用全量训练统计。
- 类别表示：最终数值化 XGBoost 使用目标编码和数值化 band，不直接使用 `*_cat`
  整数编码；`register_batch` 当前不进入最终模型。
- 文本：`attr_vals` 按 `；` 切词做 TF-IDF，再用随机化 SVD（固定随机种子）降到 48 维。
- 店铺历史：严格取 `batch <= 当前上架批次` 的最新快照（cum_sku / hit_rate_band / gmv_band）。
- 类目趋势：`demand_index` 只取历史批次，并派生环比与斜率；`supply_count` 只取本批次当期供给。
- 数值：价格及对数、价格相对本类目中位数、属性行数/种类数、属性词数与长度、band 数值化等。
- 明确不使用：`item_id/shop_id/supplier_id/cate_code` 的编号大小、行号、月份/季节特征。

### 模型与决策（pipeline.py）

- 主排序模型：XGBoost 多分类（5 档，`multi:softprob`），
  `base_score = Σ p(tier)·GAIN[tier] / 15`，其中 GAIN = {S:15, A:7, B:3, C:1, D:0}。
- S/A 辅助排序：使用同一特征矩阵训练二分类 XGBoost，预测商品属于 S 或 A 的概率。
  最终 `score = 0.94 × base_score + 0.06 × P(S或A)`，分数保持在 [0,1]。
- 决策：从单字段地板（全托管放行、其余驳回）出发，按期望代价差做边际翻转：
  - 全托管仅在「驳回明显更便宜」时改为驳回；
  - 非全托管仅在「放行明显更便宜」时改为放行。
  - 决策只读取五分类模型的概率，不读取 S/A 辅助排序分。
  - 参数 `T=1.0`、`m_ft=0.25`、`m_ot=0.00` 在训练集 4 折 OOF 概率上定值后固化，终评运行时不再做 OOF 训练。

### 防泄漏与确定性

- 时间过滤严格按 README 口径；不使用编号/行序；`predict.py` 只读 `--data-dir` 与自身目录。
- 固定全部随机种子（random / numpy / sklearn / xgboost），XGBoost 单线程（`n_jobs=1`），
  已验证同输入两次运行输出逐字节一致。
- 公开集实测约 13 秒、峰值内存约 339 MB，远低于 10 分钟和 8 GB 上限。

## 本地验证成绩

| 场景 | 分层 NDCG | 冷启动 | NDCG 部分 | 决策增量 | 客观总分 |
|---|---:|---:|---:|---:|---:|
| B05：训练 B01–B04 | +0.4368 | +0.5046 | +0.4504 | +0.2201 | **+0.3583** |
| B04–B05：训练 B01–B03 | +0.3816 | +0.5333 | +0.4119 | +0.1416 | **+0.3038** |

B04 起点的测试按赛题口径会受店铺历史结算影响，结果略偏乐观，只作为跨批次稳健性检查。
在同一划分上重训类目先验基线，B05 与 B04–B05 的客观总分分别约为 +0.0222 和 +0.0117；
这两项可以直接与本方案对比。公开集没有下发标签，因此只验证链路和格式，不报告本地分数。
