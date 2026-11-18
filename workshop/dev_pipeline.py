#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地开发管线：从 data-dir 读取原始表，跑 features.py 的特征工程并做自检。

用法:
  python dev_pipeline.py --data-dir data/val   # 验证批模式（train 有 tier，test 无 tier）
  python dev_pipeline.py --data-dir data/pub   # 公开集模式（test 无 tier）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    print("train", train.shape, "test", test.shape)

    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)

    X_train, y_gain, X_test, cat_names, fb = F.build_features(train, test)
    print("X_train", X_train.shape, "X_test", X_test.shape)
    print("特征数", X_train.shape[1], "类别特征数", len(cat_names))

    # 自检
    assert X_train.shape[0] == train.shape[0]
    assert X_test.shape[0] == test.shape[0]
    assert not X_train.isna().all().all(), "X_train 全空"

    n_nan_train = int(X_train.isna().sum().sum())
    n_nan_test = int(X_test.isna().sum().sum())
    n_inf = int(np.isinf(X_train.select_dtypes(include=[np.number]).values).sum())
    print("X_train 缺失单元格", n_nan_train, "无穷", n_inf)
    print("X_test 缺失单元格", n_nan_test)

    if "tier" in train.columns:
        print("y_gain 分布:")
        print(pd.Series(y_gain).value_counts().sort_index().to_string())
        warm = set(train["shop_id"])
        cold_frac = float((~test["shop_id"].isin(warm)).mean())
        print("test 冷启动店铺比例 %.4f" % cold_frac)

    # 保存
    out_dir = args.out_dir or os.path.join("artifacts", os.path.basename(os.path.normpath(args.data_dir)))
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    train[["item_id"]].to_csv(os.path.join(out_dir, "train_item.csv"), index=False)
    test[["item_id"]].to_csv(os.path.join(out_dir, "test_item.csv"), index=False)
    if "tier" in train.columns:
        pd.Series(y_gain, name="y_gain").to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(X_train.columns))
    with open(os.path.join(out_dir, "cat_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cat_names))
    print("已保存到", out_dir)


if __name__ == "__main__":
    main()
