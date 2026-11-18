#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预处理冒烟测试：用 LightGBM 多分类快速拟合，验证特征可用并产出合规提交。

完整建模在下一步进行；这里只做端到端链路验证。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

COST = {("S", "驳回"): 12, ("A", "驳回"): 8, ("B", "驳回"): 2,
        ("B", "放行"): 2, ("C", "放行"): 3, ("D", "放行"): 4,
        ("S", "放行"): 0, ("A", "放行"): 0, ("C", "驳回"): 0, ("D", "驳回"): 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)
    X_train, y_gain, X_test, cat_names, fb = F.build_features(train, test)

    y_tier = train["tier"].astype(str).str.strip()
    cls = {t: i for i, t in enumerate(F.TIERS)}
    y = y_tier.map(cls).astype(int).values

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        colsample_bytree=0.8,
        subsample=0.9,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=0,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(X_train, y, categorical_feature=cat_names)

    proba = model.predict_proba(X_test)
    score = np.clip(proba @ np.array([F.GAIN[t] for t in F.TIERS]) / 15.0, 0.0, 1.0)

    e_allow = 2 * proba[:, 2] + 3 * proba[:, 3] + 4 * proba[:, 4]
    e_reject = 12 * proba[:, 0] + 8 * proba[:, 1] + 2 * proba[:, 2]
    decision = np.where(e_allow <= e_reject, "放行", "驳回")

    sub = pd.DataFrame({"item_id": test["item_id"].values, "score": score, "decision": decision})
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print("已写入", args.out, sub.shape)
    print("score 分位:", np.percentile(score, [0, 25, 50, 75, 100]).round(4))
    print("decision 分布:", pd.Series(decision).value_counts().to_dict())


if __name__ == "__main__":
    main()
