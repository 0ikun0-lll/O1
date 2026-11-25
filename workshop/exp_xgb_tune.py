#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGBoost 排序超参快速扫描（NDCG 部分，内部复算官方口径）。"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
sys.path.insert(0, r"D:\O1\O1_爆品早知道_资料包")
import evaluate as EV

GAIN_VEC = np.array([F.GAIN[t] for t in F.TIERS], dtype=np.float64)


def ndcg_part(test, train, truth_df, scores):
    truth = {r["item_id"]: r["tier"] for r in truth_df.to_dict("records")}
    group = {r["item_id"]: (r["shop_main_cate"] or "ALL") for r in test.to_dict("records")}
    warm = set(train["shop_id"].astype(str).str.strip())
    is_cold = {r["item_id"]: (str(r["shop_id"]).strip() not in warm)
               for r in test.to_dict("records")}
    sd = {iid: float(s) for iid, s in zip(test["item_id"].values, scores)}
    macro, _, _, _ = EV.eval_stratified(truth, sd, group, min_group=20, k=100)
    cold = EV.eval_cold(truth, sd, is_cold, k=100)
    return macro, (cold[3] if cold else 0.0), 0.8 * macro + 0.2 * (cold[3] if cold else 0.0)


def main():
    data_dir = os.path.join(HERE, "data", "val")
    train, test, shop, sh, ct = F.load_tables(data_dir)
    train = F.attach_external(train, shop, sh, ct)
    test = F.attach_external(test, shop, sh, ct)
    fb = F.FeatureBuilder(n_folds=4)
    fb.fit(train)
    Xtr, yg = fb.transform_train(train)
    Xte = fb.transform_test(test)
    drop = [c for c in Xtr.columns if c.endswith("_cat")]
    Xtr = Xtr.drop(columns=drop)
    Xte = Xte.drop(columns=drop)
    y = train["tier"].astype(str).str.strip().map({t: i for i, t in enumerate(F.TIERS)}).astype(int).values
    truth = pd.read_csv(data_dir.rstrip("\\/") + "_truth.csv", encoding="utf-8-sig")

    cfgs = [
        dict(max_depth=6, n_estimators=300, learning_rate=0.05),
        dict(max_depth=5, n_estimators=300, learning_rate=0.05),
        dict(max_depth=7, n_estimators=300, learning_rate=0.05),
        dict(max_depth=6, n_estimators=400, learning_rate=0.05),
        dict(max_depth=8, n_estimators=200, learning_rate=0.05),
        dict(max_depth=6, n_estimators=500, learning_rate=0.03),
        dict(max_depth=9, n_estimators=300, learning_rate=0.05),
    ]
    for c in cfgs:
        p = dict(objective="multi:softprob", num_class=5, subsample=0.9,
                 colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                 random_state=0, n_jobs=1, tree_method="hist", verbosity=0, **c)
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, y)
        s = m.predict_proba(Xte) @ GAIN_VEC / 15.0
        r = ndcg_part(test, train, truth, s)
        print("%-40s NDCG部分=%.4f  (macro %.4f, cold %.4f)" % (str(c), r[2], r[0], r[1]))


if __name__ == "__main__":
    main()
