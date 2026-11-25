#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O1 · 爆品早知道 终评提交入口。

用法:
    python3 predict.py --data-dir <数据目录> --out submission.csv

约束遵循:
    - 只读 --data-dir 与自身目录，不联网，仅 CPU；
    - 确定性：固定全部随机种子、单线程训练，同输入同输出逐字节一致。
"""
import argparse
import csv
import os
import random

import numpy as np

import features as F
import pipeline as P


def main():
    # 确定性兜底：固定全局随机种子（实际随机源均已显式设种子）
    random.seed(0)
    np.random.seed(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    train, test, shop, shop_hist, cat_trend = F.load_tables(args.data_dir)
    train = F.attach_external(train, shop, shop_hist, cat_trend)
    test = F.attach_external(test, shop, shop_hist, cat_trend)

    pl = P.ModelPipeline(n_folds=4)
    pl.fit(train)
    score, decision = pl.predict(test)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "score", "decision"])
        for iid, s, d in zip(test["item_id"].values, score, decision):
            w.writerow([iid, "%.6f" % s, d])
    print("已写入 %s（%d 行）" % (args.out, len(test)))


if __name__ == "__main__":
    main()
