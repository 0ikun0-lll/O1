#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开发期：跑 ModelPipeline（OOF 定决策边际）并写提交。"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)

    pl = P.ModelPipeline()
    pl.fit(train)
    print("OOF 决策损失 L=%.4f  温度 T=%.3f  边际 m_ft=%.2f m_ot=%.2f"
          % (pl.oof_loss, pl.T, pl.m_ft, pl.m_ot))

    score, decision = pl.predict(test)
    sub = pd.DataFrame({"item_id": test["item_id"].values, "score": score, "decision": decision})
    sub.to_csv(args.out, index=False, encoding="utf-8")
    print("已写入", args.out)
    print("score 分位:", np.percentile(score, [0, 25, 50, 75, 100]).round(4))
    print("decision 分布:", dict(pd.Series(decision).value_counts()))


if __name__ == "__main__":
    main()
