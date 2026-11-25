#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排序实验：LightGBM 多分类变体 + XGBoost + 融合，对比 NDCG 部分（内部复算官方口径）。"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F

PKG = r"D:\O1\O1_爆品早知道_资料包"
sys.path.insert(0, PKG)
import evaluate as EV

GAIN_VEC = np.array([F.GAIN[t] for t in F.TIERS], dtype=np.float64)


def build(data_dir, n_folds=4):
    train, test, shop, sh, ct = F.load_tables(data_dir)
    train = F.attach_external(train, shop, sh, ct)
    test = F.attach_external(test, shop, sh, ct)
    fb = F.FeatureBuilder(n_folds=n_folds)
    fb.fit(train)
    Xtr, yg = fb.transform_train(train)
    Xte = fb.transform_test(test)
    y = train["tier"].astype(str).str.strip().map({t: i for i, t in enumerate(F.TIERS)}).astype(int).values
    return train, test, Xtr, y, Xte, fb


def score_dict(test, scores):
    return {iid: float(s) for iid, s in zip(test["item_id"].values, scores)}


def ndcg_part(test, train, truth_df, scores):
    truth = {r["item_id"]: r["tier"] for r in truth_df.to_dict("records")}
    group = {r["item_id"]: (r["shop_main_cate"] or "ALL") for r in test.to_dict("records")}
    warm = set(train["shop_id"].astype(str).str.strip())
    is_cold = {r["item_id"]: (str(r["shop_id"]).strip() not in warm)
               for r in test.to_dict("records")}
    sd = score_dict(test, scores)
    macro, _, _, _ = EV.eval_stratified(truth, sd, group, min_group=20, k=100)
    cold = EV.eval_cold(truth, sd, is_cold, k=100)
    return macro, (cold[3] if cold else 0.0), 0.8 * macro + 0.2 * (cold[3] if cold else 0.0)


def gain_from_proba(P):
    return P @ GAIN_VEC / 15.0


def main():
    data_dir = os.path.join(HERE, "data", "val")
    train, test, Xtr, y, Xte, fb = build(data_dir)
    cat_names = [c + "_cat" for c in fb.cat_cols]
    truth = pd.read_csv(data_dir.rstrip("\\/") + "_truth.csv", encoding="utf-8-sig")

    # LightGBM 基线（当前）
    m = lgb.LGBMClassifier(objective="multiclass", num_class=5, n_estimators=300,
                           learning_rate=0.05, num_leaves=63, min_child_samples=20,
                           colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
                           reg_alpha=0.1, reg_lambda=0.1, random_state=0, n_jobs=1, verbosity=-1)
    m.fit(Xtr, y, categorical_feature=cat_names)
    lgb_base = gain_from_proba(m.predict_proba(Xte))
    r = ndcg_part(test, train, truth, lgb_base)
    print("LGBM base      NDCG部分=%.4f  (macro %.4f, cold %.4f)" % (r[2], r[0], r[1]))

    # XGBoost（数值化，不含 raw categorical）
    drop = [c for c in Xtr.columns if c.endswith("_cat")]
    Xtr_n = Xtr.drop(columns=drop)
    Xte_n = Xte.drop(columns=drop)
    xm = xgb.XGBClassifier(objective="multi:softprob", num_class=5, n_estimators=300,
                           learning_rate=0.05, max_depth=6, subsample=0.9,
                           colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=0, n_jobs=1, tree_method="hist", verbosity=0)
    xm.fit(Xtr_n, y)
    xgb_score = gain_from_proba(xm.predict_proba(Xte_n))
    r = ndcg_part(test, train, truth, xgb_score)
    print("XGBoost        NDCG部分=%.4f  (macro %.4f, cold %.4f)" % (r[2], r[0], r[1]))

    # 融合
    for w in (0.5, 0.6, 0.7):
        blend = w * lgb_base + (1 - w) * xgb_score
        r = ndcg_part(test, train, truth, blend)
        print("Blend w_LGB=%.1f NDCG部分=%.4f  (macro %.4f, cold %.4f)" % (w, r[2], r[0], r[1]))


if __name__ == "__main__":
    main()
