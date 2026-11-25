#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开发期建模脚本：多分类 LightGBM + 从地板出发、搜边际的决策规则。

验证批模式下会读取 <data-dir>_truth.csv 来搜决策边际（开发期选型，允许使用验证批标签）。
公开集/终评模式下没有真值，使用已定好的边际。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

COST = {("S", "驳回"): 12, ("A", "驳回"): 8, ("B", "驳回"): 2,
        ("B", "放行"): 2, ("C", "放行"): 3, ("D", "放行"): 4,
        ("S", "放行"): 0, ("A", "放行"): 0, ("C", "驳回"): 0, ("D", "驳回"): 0}


def expected_costs(proba):
    """proba: Nx5，列序 S,A,B,C,D。返回 (allow_cost, reject_cost)。"""
    pS, pA, pB, pC, pD = proba.T
    allow = 2 * pB + 3 * pC + 4 * pD
    reject = 12 * pS + 8 * pA + 2 * pB
    return allow, reject


def decision_from_margins(allow_cost, reject_cost, is_ft, m_ft, m_ot):
    delta = reject_cost - allow_cost  # delta>0: 放行更便宜；delta<0: 驳回更便宜
    out = np.where(is_ft, "放行", "驳回").astype(object)
    # 全托管（地板=放行）：仅当驳回明显更便宜时翻转
    out = np.where(is_ft & (delta < -m_ft), "驳回", out)
    # 非全托管（地板=驳回）：仅当放行明显更便宜时翻转
    out = np.where((~is_ft) & (delta > m_ot), "放行", out)
    return out


def loss_of(decisions, tiers):
    return np.mean([COST[(t, d)] for t, d in zip(tiers, decisions)])


def loss_vectorized(decisions, tiers):
    """decisions: 0=放行, 1=驳回；tiers: 0..4（S,A,B,C,D）。"""
    cost_allow = np.array([0, 0, 2, 3, 4], dtype=np.float64)
    cost_reject = np.array([12, 8, 2, 0, 0], dtype=np.float64)
    dec = np.asarray(decisions, dtype=np.int64)
    t = np.asarray(tiers, dtype=np.int64)
    return float(np.mean(np.where(dec == 1, cost_reject[t], cost_allow[t])))


def load_truth(data_dir):
    p = os.path.normpath(data_dir).rstrip("\\/") + "_truth.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, encoding="utf-8-sig")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)
    X_train, y_gain, X_test, cat_names, fb = F.build_features(train, test)

    y_tier = train["tier"].astype(str).str.strip()
    cls = {t: i for i, t in enumerate(F.TIERS)}
    y = y_tier.map(cls).astype(int).values

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        min_child_samples=20, colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
        reg_alpha=0.1, reg_lambda=0.1, random_state=0, n_jobs=1, verbosity=-1,
    )
    model.fit(X_train, y, categorical_feature=cat_names)

    proba = model.predict_proba(X_test)
    score = np.clip(proba @ np.array([F.GAIN[t] for t in F.TIERS]) / 15.0, 0.0, 1.0)
    allow_cost, reject_cost = expected_costs(proba)
    is_ft = (test["fulfillment_mode"].astype(str).str.strip() == "全托管").values

    truth = load_truth(args.data_dir)
    m_ft = m_ot = 0.0
    if truth is not None:
        tt = truth.set_index("item_id")["tier"].astype(str).str.strip()
        tiers = test["item_id"].map(tt).values
        pSA = proba[:, 0] + proba[:, 1]
        is_SA = np.isin(tiers, ["S", "A"])
        for grp, mask in [("全托管", is_ft), ("非全托管", ~is_ft)]:
            if mask.sum() > 1 and is_SA[mask].sum() > 0 and (~is_SA[mask]).sum() > 0:
                auc = roc_auc_score(is_SA[mask], pSA[mask])
                print("AUC(pS+pA vs 是否S/A) [%s] = %.4f  (n=%d)" % (grp, auc, mask.sum()))
        L_floor = loss_of(np.where(is_ft, "放行", "驳回"), tiers)
        allow_all = loss_of(np.full(len(tiers), "放行"), tiers)
        best = np.mean([min(COST[(t, "放行")], COST[(t, "驳回")]) for t in tiers])

        tier_code = np.array([F.TIERS.index(t) if t in F.TIERS else 2 for t in tiers], dtype=np.int64)
        grid_ft = np.arange(0.0, 14.01, 0.1)
        grid_ot = np.arange(0.0, 6.01, 0.1)
        best_loss, best_m = np.inf, (0.0, 0.0)
        for a in grid_ft:
            for b in grid_ot:
                dec = decision_from_margins(allow_cost, reject_cost, is_ft, a, b)
                dec_code = np.where(dec == "驳回", 1, 0)
                L = loss_vectorized(dec_code, tier_code)
                if L < best_loss:
                    best_loss, best_m = L, (a, b)
        m_ft, m_ot = best_m
        dec = decision_from_margins(allow_cost, reject_cost, is_ft, m_ft, m_ot)
        L_model = loss_vectorized(np.where(dec == "驳回", 1, 0), tier_code)
        dec_score = max(0.0, (L_floor - L_model) / (L_floor - best)) if (L_floor - best) > 1e-9 else 0.0
        print("真值锚点: 全放行 L=%.4f  全对 L=%.4f  地板 L=%.4f" % (allow_all, best, L_floor))
        print("最优边际 m_ft=%.2f m_ot=%.2f -> 模型 L=%.4f" % (m_ft, m_ot, L_model))
        print("决策增量（计分）= %.4f" % dec_score)
        print("放行/驳回分布:", dict(pd.Series(dec).value_counts()))
    else:
        dec = decision_from_margins(allow_cost, reject_cost, is_ft, m_ft, m_ot)
        print("无真值（公开集/终评模式），使用边际 m_ft=%.2f m_ot=%.2f" % (m_ft, m_ot))

    sub = pd.DataFrame({"item_id": test["item_id"].values, "score": score, "decision": dec})
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print("已写入", args.out)


if __name__ == "__main__":
    main()
