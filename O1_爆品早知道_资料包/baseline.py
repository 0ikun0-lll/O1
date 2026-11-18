#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O1 基线：按 shop_main_cate（店铺主营一级类目）的历史档位分布预测。只用 Python 3.9+ 标准库。

  python3 baseline.py
    → submission_baseline.csv：用 train.csv 训练，预测 test_public.csv
    → val_pred_baseline.csv  ：若存在 train_sub.csv / val_items.csv（先运行 python3 evaluate.py --make-val），
                               用 train_sub 训练，预测验证集

输出列：item_id, score, decision（与提交格式一致，不含 pred_tier）
  score    = 类目历史档位分布的期望增益（Σ p(t)·GAIN[t] / 15，∈[0,1]）；
             同一 shop_main_cate 内所有商品 score 相同。
  decision = 由 cost_matrix.csv 的代价推出阈值：E[损失|放行] ≤ E[损失|驳回] 则放行。
             默认代价：放行 E = 2·p(B) + 3·p(C) + 4·p(D)；驳回 E = 12·p(S) + 8·p(A) + 2·p(B)
"""
import csv
import os
from collections import Counter, defaultdict

TIERS = ["S", "A", "B", "C", "D"]
REL = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
GAIN = {"S": 15, "A": 7, "B": 3, "C": 1, "D": 0}
DEFAULT_COST = {("S", "驳回"): 12, ("A", "驳回"): 8, ("B", "驳回"): 2,
                ("B", "放行"): 2, ("C", "放行"): 3, ("D", "放行"): 4,
                ("S", "放行"): 0, ("A", "放行"): 0, ("C", "驳回"): 0, ("D", "驳回"): 0}


def load_cost_matrix(path):
    """读取 cost_matrix.csv；缺文件/缺行时回退到内置默认值。"""
    cost = dict(DEFAULT_COST)
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                tier, dec = (r.get("true_tier") or "").strip(), (r.get("decision") or "").strip()
                if tier in TIERS and dec in ("放行", "驳回"):
                    cost[(tier, dec)] = int(round(float(r.get("cost") or 0)))
    except (OSError, ValueError):
        pass
    return cost


def fit(path, cost):
    by = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by[(r.get("shop_main_cate") or "ALL").strip()].append((r.get("tier") or "").strip())
    allt = [t for v in by.values() for t in v]
    model = {}
    for g, v in by.items():
        model[g] = dist(v, cost)
    return model, dist(allt, cost)


def dist(v, cost):
    n = len(v)
    cnt = Counter(v)
    p = {t: cnt.get(t, 0) / float(n) for t in TIERS} if n else {t: 0.0 for t in TIERS}
    exp_gain = sum(p[t] * GAIN[t] for t in TIERS) / 15.0
    e_allow = sum(p[t] * cost[(t, "放行")] for t in TIERS)
    e_reject = sum(p[t] * cost[(t, "驳回")] for t in TIERS)
    return dict(score=exp_gain, decision="放行" if e_allow <= e_reject else "驳回")


def predict(model, default, items_path, out_path):
    n = 0
    with open(items_path, newline="", encoding="utf-8-sig") as f, \
            open(out_path, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["item_id", "score", "decision"])
        for r in csv.DictReader(f):
            d = model.get((r.get("shop_main_cate") or "ALL").strip(), default)
            w.writerow([r["item_id"], "%.4f" % d["score"], d["decision"]])
            n += 1
    print("已写入 %s（%d 行）" % (out_path, n))


def main():
    if not os.path.exists("train.csv"):
        raise SystemExit("请在解压后的 O1_爆品早知道_资料包/ 目录下运行（找不到 train.csv）")
    cost = load_cost_matrix("cost_matrix.csv")
    m, dft = fit("train.csv", cost)
    predict(m, dft, "test_public.csv", "submission_baseline.csv")
    if os.path.exists("train_sub.csv") and os.path.exists("val_items.csv"):
        m2, d2 = fit("train_sub.csv", cost)
        predict(m2, d2, "val_items.csv", "val_pred_baseline.csv")
        print("提示：python3 evaluate.py --pred val_pred_baseline.csv --truth val_truth.csv --items val_items.csv --train train_sub.csv")
    else:
        print("提示：先运行 python3 evaluate.py --make-val 生成验证集，再运行本脚本可得到验证集预测")


if __name__ == "__main__":
    main()
