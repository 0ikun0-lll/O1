#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排序实验：LightGBM LambdaRank（按 shop_main_cate 分组）vs 多分类期望增益。"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)

    fb = F.FeatureBuilder()
    fb.fit(train)
    X_train, y_gain = fb.transform_train(train)
    X_test = fb.transform_test(test)
    cat_names = [c + "_cat" for c in fb.cat_cols]

    # LambdaRank：按 shop_main_cate 分组
    grp = train["shop_main_cate"].fillna("MISSING").astype(str).str.strip()
    order = np.argsort(grp.values, kind="mergesort")
    X = X_train.iloc[order].reset_index(drop=True)
    y = y_gain.iloc[order].reset_index(drop=True).astype(np.float64).values
    groups = grp.iloc[order].value_counts(sort=False).reindex(sorted(grp.iloc[order].unique())).values
    groups = groups.astype(np.int32)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
        reg_alpha=0.1, reg_lambda=0.1, random_state=0, n_jobs=1, verbosity=-1,
    )
    ranker.fit(X, y, group=groups, categorical_feature=cat_names)
    score = ranker.predict(X_test)
    # 排序分归一化到 [0,1] 不改变排序，直接 min-max 映射保证提交值域合规
    lo, hi = score.min(), score.max()
    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score = np.full_like(score, 0.5)

    decision = np.where(test["fulfillment_mode"].astype(str).str.strip() == "全托管",
                        "放行", "驳回")
    sub = pd.DataFrame({"item_id": test["item_id"].values, "score": score, "decision": decision})
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print("已写入", args.out, "组数", len(groups))


if __name__ == "__main__":
    main()
