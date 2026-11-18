#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终评代码提交模板（实现这个接口即可，纯标准库示例）。

接口契约（组织者按此调用）:
    python3 predict.py --data-dir <数据目录> --out submission.csv

<数据目录> 内含:
    train.csv                有标签训练集（item_id, ..., tier）
    test.csv                 待预测特征（列与 test_public.csv 同构，无 tier）
    shop.csv / shop_hist_monthly.csv / category_trend.csv
    cost_matrix.csv / baseline_scores.csv（**只含训练集行**）/ audit_reason_dict.csv / human_review_sample.csv

约束:
    - 只能读取 --data-dir 与代码包自身目录（如模型权重），不得读取其它路径；
      不得联网；仅 CPU；≤10 分钟；内存 ≤ 8 GB；必须确定性（同输入同输出）
    - 输出 submission.csv（UTF-8）: item_id,score,decision
      score ∈ [0,1] 越大越优先；decision ∈ {放行, 驳回}

本模板用"店铺主营类目历史档位分布"打分，与 baseline.py 同口径，仅作跑通示例。
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

TIERS = ["S", "A", "B", "C", "D"]
GAIN = {"S": 15, "A": 7, "B": 3, "C": 1, "D": 0}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rd]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    d = a.data_dir
    train = read_csv(os.path.join(d, "train.csv"))
    test = read_csv(os.path.join(d, "test.csv"))

    by = defaultdict(list)
    for r in train:
        by[(r.get("shop_main_cate") or "ALL").strip()].append(r.get("tier", ""))
    global_cnt = Counter(t for v in by.values() for t in v)

    model = {}
    for g, v in by.items():
        n = float(len(v))
        cnt = Counter(v)
        model[g] = {t: cnt.get(t, 0) / n for t in TIERS}
    n = float(sum(global_cnt.values())) or 1.0
    default = {t: global_cnt.get(t, 0) / n for t in TIERS}

    rows = []
    for r in test:
        p = model.get((r.get("shop_main_cate") or "ALL").strip(), default)
        score = sum(p[t] * GAIN[t] for t in TIERS) / 15.0
        e_allow = 2 * p["B"] + 3 * p["C"] + 4 * p["D"]
        e_reject = 12 * p["S"] + 8 * p["A"] + 2 * p["B"]
        rows.append([r["item_id"], "%.4f" % score, "放行" if e_allow <= e_reject else "驳回"])

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "score", "decision"])
        w.writerows(rows)
    print("已写入 %s（%d 行）" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
