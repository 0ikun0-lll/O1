#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复用的训练/预测管线：多分类 LightGBM + 训练集 OOF 校准与边际定值。

终评 predict.py 直接复用 ModelPipeline：
  fit(train_df) 在训练集上拟合编码器、做 OOF 概率、定出温度与决策边际；
  predict(test_df) 输出 (score, decision)。
全部参数确定性固定，满足「同输入同输出」。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold

import features as F

COST = {("S", "驳回"): 12, ("A", "驳回"): 8, ("B", "驳回"): 2,
        ("B", "放行"): 2, ("C", "放行"): 3, ("D", "放行"): 4,
        ("S", "放行"): 0, ("A", "放行"): 0, ("C", "驳回"): 0, ("D", "驳回"): 0}

TIER_IDX = {t: i for i, t in enumerate(F.TIERS)}
GAIN_VEC = np.array([F.GAIN[t] for t in F.TIERS], dtype=np.float64)


def softmax(raw, T=1.0):
    x = np.asarray(raw, dtype=np.float64) / T
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def expected_gain(proba):
    return np.clip(proba @ GAIN_VEC / 15.0, 0.0, 1.0)


def expected_costs(proba):
    pS, pA, pB, pC, pD = proba.T
    allow = 2 * pB + 3 * pC + 4 * pD
    reject = 12 * pS + 8 * pA + 2 * pB
    return allow, reject


def decisions_from_margins(allow_cost, reject_cost, is_ft, m_ft, m_ot):
    delta = reject_cost - allow_cost  # >0 放行更便宜；<0 驳回更便宜
    out = np.where(is_ft, "放行", "驳回").astype(object)
    out = np.where(is_ft & (delta < -m_ft), "驳回", out)
    out = np.where((~is_ft) & (delta > m_ot), "放行", out)
    return out


def _loss_vectorized(decisions, tier_codes):
    cost_allow = np.array([0, 0, 2, 3, 4], dtype=np.float64)
    cost_reject = np.array([12, 8, 2, 0, 0], dtype=np.float64)
    dec = np.asarray(decisions, dtype=np.int64)
    t = np.asarray(tier_codes, dtype=np.int64)
    return float(np.mean(np.where(dec == 1, cost_reject[t], cost_allow[t])))


def tune_decision(raw_oof, tier_codes, is_ft, seed=0):
    """在 OOF raw scores 上定温度 T 与边际 (m_ft, m_ot)，目标是最小化平均代价 L。"""
    T_grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    ft_grid = np.arange(0.0, 4.01, 0.25)
    ot_grid = np.arange(0.0, 3.01, 0.25)
    best = (np.inf, 1.0, 0.0, 0.0)
    for T in T_grid:
        proba = softmax(raw_oof, T)
        allow, reject = expected_costs(proba)
        for a in ft_grid:
            for b in ot_grid:
                dec = decisions_from_margins(allow, reject, is_ft, a, b)
                dec_code = np.where(dec == "驳回", 1, 0)
                L = _loss_vectorized(dec_code, tier_codes)
                if L < best[0]:
                    best = (L, T, a, b)
    return best  # (L, T, m_ft, m_ot)


DEFAULT_PARAMS = dict(
    objective="multiclass",
    num_class=5,
    n_estimators=300,
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


DEFAULT_XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=5,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=1,
    tree_method="hist",
    verbosity=0,
)


class ModelPipeline:
    def __init__(self, params=None, n_folds=5, seed=0, te_cols=None, n_svd=48, min_df=2):
        self.params = dict(params or DEFAULT_PARAMS)
        self.n_folds = n_folds
        self.seed = seed
        self.te_cols = list(te_cols) if te_cols is not None else list(F.TE_COLS)
        self.n_svd = n_svd
        self.min_df = min_df
        self.fb = None
        self.model = None
        self.T = 1.0
        self.m_ft = 0.0
        self.m_ot = 0.0

    def fit(self, train_df):
        self.fb = F.FeatureBuilder(
            te_cols=self.te_cols, n_svd=self.n_svd, min_df=self.min_df,
            n_folds=self.n_folds, seed=self.seed,
        )
        self.fb.fit(train_df)
        X_train, y_gain = self.fb.transform_train(train_df)
        self.X_train = X_train
        self.y_gain = y_gain
        tier = train_df["tier"].astype(str).str.strip()
        self.y = tier.map(TIER_IDX).astype(np.int32).values
        self.tier_codes = self.y.copy()
        self.is_ft = (train_df["fulfillment_mode"].astype(str).str.strip() == "全托管").values
        self.cat_names = [c + "_cat" for c in self.fb.cat_cols]

        # OOF raw scores
        self.oof_raw = self._oof_raw_scores(X_train, self.y, self.cat_names)
        L, T, m_ft, m_ot = tune_decision(self.oof_raw, self.tier_codes, self.is_ft, self.seed)
        self.T, self.m_ft, self.m_ot = T, m_ft, m_ot
        self.oof_loss = L

        # 最终模型
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_train, self.y, categorical_feature=self.cat_names)
        return self

    def _oof_raw_scores(self, X, y, cat_names):
        raw = np.zeros((len(X), 5), dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            m = lgb.LGBMClassifier(**self.params)
            m.fit(X.iloc[tr], y[tr], categorical_feature=cat_names)
            raw[va] = m.predict(X.iloc[va], raw_score=True)
        return raw

    def predict(self, test_df):
        X_test = self.fb.transform_test(test_df)
        raw = self.model.predict(X_test, raw_score=True)
        proba_rank = softmax(raw, 1.0)
        proba_dec = softmax(raw, self.T)
        score = expected_gain(proba_rank)
        allow, reject = expected_costs(proba_dec)
        is_ft = (test_df["fulfillment_mode"].astype(str).str.strip() == "全托管").values
        decision = decisions_from_margins(allow, reject, is_ft, self.m_ft, self.m_ot)
        return score, decision


class XgbPipeline:
    """XGBoost 多分类：数值特征（含目标编码/文本 SVD）+ OOF 温度与边际校准。"""

    def __init__(self, params=None, n_folds=4, seed=0, te_cols=None, n_svd=48, min_df=2):
        self.params = dict(params or DEFAULT_XGB_PARAMS)
        self.n_folds = n_folds
        self.seed = seed
        self.te_cols = list(te_cols) if te_cols is not None else list(F.TE_COLS)
        self.n_svd = n_svd
        self.min_df = min_df
        self.fb = None
        self.model = None
        self.T = 1.0
        self.m_ft = 0.0
        self.m_ot = 0.0

    def _numeric(self, X):
        drop = [c for c in X.columns if c.endswith("_cat")]
        return X.drop(columns=drop)

    def fit(self, train_df):
        self.fb = F.FeatureBuilder(
            te_cols=self.te_cols, n_svd=self.n_svd, min_df=self.min_df,
            n_folds=self.n_folds, seed=self.seed,
        )
        self.fb.fit(train_df)
        X_train, y_gain = self.fb.transform_train(train_df)
        self.X_train = X_train
        self.y_gain = y_gain
        tier = train_df["tier"].astype(str).str.strip()
        self.y = tier.map(TIER_IDX).astype(np.int32).values
        self.tier_codes = self.y.copy()
        self.is_ft = (train_df["fulfillment_mode"].astype(str).str.strip() == "全托管").values

        Xn = self._numeric(X_train)
        self.oof_margin = self._oof_margins(Xn, self.y)
        L, T, m_ft, m_ot = tune_decision(self.oof_margin, self.tier_codes, self.is_ft, self.seed)
        self.T, self.m_ft, self.m_ot = T, m_ft, m_ot
        self.oof_loss = L

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(Xn, self.y)
        return self

    def _oof_margins(self, X, y):
        margin = np.zeros((len(X), 5), dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            m = xgb.XGBClassifier(**self.params)
            m.fit(X.iloc[tr], y[tr])
            margin[va] = m.predict(X.iloc[va], output_margin=True)
        return margin

    def predict(self, test_df):
        X_test = self.fb.transform_test(test_df)
        Xn = self._numeric(X_test)
        margin = self.model.predict(Xn, output_margin=True)
        proba_rank = softmax(margin, 1.0)
        proba_dec = softmax(margin, self.T)
        score = expected_gain(proba_rank)
        allow, reject = expected_costs(proba_dec)
        is_ft = (test_df["fulfillment_mode"].astype(str).str.strip() == "全托管").values
        decision = decisions_from_margins(allow, reject, is_ft, self.m_ft, self.m_ot)
        return score, decision


# 开发期在训练集 OOF 上定出的决策参数（终评运行时直接使用，不再做 OOF 训练）。
FINAL_T = 1.0
FINAL_M_FT = 0.25
FINAL_M_OT = 0.0


def final_predict(train_df, test_df, params=None, n_folds=4, seed=0):
    """终评快速路径：特征 + 单个 XGBoost + 固化决策参数。"""
    fb = F.FeatureBuilder(
        te_cols=F.TE_COLS, n_svd=48, min_df=2, n_folds=n_folds, seed=seed,
    )
    fb.fit(train_df)
    X_train, _ = fb.transform_train(train_df)
    X_test = fb.transform_test(test_df)
    drop = [c for c in X_train.columns if c.endswith("_cat")]
    Xn = X_train.drop(columns=drop)
    Xn_test = X_test.drop(columns=drop)
    y = train_df["tier"].astype(str).str.strip().map(TIER_IDX).astype(np.int32).values

    model = xgb.XGBClassifier(**(dict(params or DEFAULT_XGB_PARAMS)))
    model.fit(Xn, y)
    margin = model.predict(Xn_test, output_margin=True)
    proba = softmax(margin, FINAL_T)
    score = expected_gain(proba)
    allow, reject = expected_costs(proba)
    is_ft = (test_df["fulfillment_mode"].astype(str).str.strip() == "全托管").values
    decision = decisions_from_margins(allow, reject, is_ft, FINAL_M_FT, FINAL_M_OT)
    return score, decision
