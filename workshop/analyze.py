#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析脚本：特征重要性 + 决策侧与地板的差距拆解（只读，不改资料包）。"""
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
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)
    X_train, y_gain, X_test, cat_names, fb = F.build_features(train, test)

    y_tier = train["tier"].astype(str).str.strip()
    cls = {t: i for i, t in enumerate(F.TIERS)}
    y = y_tier.map(cls).astype(int).values

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=20, colsample_bytree=0.8, subsample=0.9,
        subsample_freq=1, reg_alpha=0.1, reg_lambda=0.1, random_state=0, n_jobs=1,
        verbosity=-1,
    )
    model.fit(X_train, y, categorical_feature=cat_names)

    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                    index=X_train.columns).sort_values(ascending=False)
    print("Top 35 特征重要性（gain）:")
    for i, (k, v) in enumerate(imp.head(35).items(), 1):
        print("%3d %-24s %10.1f" % (i, k, v))

    proba = model.predict_proba(X_test)
    e_allow = 2 * proba[:, 2] + 3 * proba[:, 3] + 4 * proba[:, 4]
    e_reject = 12 * proba[:, 0] + 8 * proba[:, 1] + 2 * proba[:, 2]
    model_dec = pd.Series(np.where(e_allow <= e_reject, "放行", "驳回"),
                          index=test.index)
    floor_dec = (test["fulfillment_mode"].astype(str).str.strip() == "全托管").map(
        {True: "放行", False: "驳回"}
    )

    truth_path = os.path.normpath(args.data_dir).rstrip("\\/") + "_truth.csv"
    truth = pd.read_csv(truth_path, encoding="utf-8-sig")
    truth["tier"] = truth["tier"].astype(str).str.strip()
    tt = truth.set_index("item_id")["tier"]
    # 对齐 test 行序（test 与 truth 的 item_id 顺序应一致，但仍按 id 对齐）
    t = test["item_id"].map(tt).values

    def loss(decisions):
        return np.mean([COST[(tier, d)] for tier, d in zip(t, decisions)])

    L_floor = loss(floor_dec.values)
    L_model = loss(model_dec.values)
    print("\n决策损失: 地板 %.4f  模型 %.4f" % (L_floor, L_model))

    diff = model_dec.values != floor_dec.values
    print("模型与地板不一致的商品数:", int(diff.sum()))
    sub = pd.DataFrame({
        "tier": t, "floor": floor_dec.values, "model": model_dec.values,
        "mode": test["fulfillment_mode"].astype(str).str.strip().values,
    })
    print("\n不一致样本 × (合作模式, 真实档, 地板→模型):")
    print(pd.crosstab([sub.loc[diff, "mode"], sub.loc[diff, "tier"]],
                      [sub.loc[diff, "floor"], sub.loc[diff, "model"]]).to_string())


if __name__ == "__main__":
    main()
