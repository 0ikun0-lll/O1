#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速 EDA：不修改资料包，只读。"""
import os

import numpy as np
import pandas as pd

BASE = r"D:\O1\O1_爆品早知道_资料包"


def load(name):
    return pd.read_csv(os.path.join(BASE, name), encoding="utf-8-sig")


def banner(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def main():
    train = load("train.csv")
    test = load("test_public.csv")

    banner("主表：train/test 形状与缺失")
    for df, nm in [(train, "train"), (test, "test_public")]:
        print(f"\n[{nm}] shape={df.shape}")
        miss = df.isna().sum()
        miss = miss[miss > 0]
        print("缺失列:")
        print(miss.to_string())

    banner("tier 分布（train 整体 + 各批次）")
    print(train["tier"].value_counts(normalize=True).sort_index().round(4).to_string())
    print("\n各批次 tier 分布:")
    print(pd.crosstab(train["listing_batch"], train["tier"], normalize="index").round(4).to_string())

    banner("tier 分布 × fulfillment_mode（决策地板上下文）")
    print(pd.crosstab(train["fulfillment_mode"], train["tier"], normalize="index").round(4).to_string())
    print("\n数量:")
    print(pd.crosstab(train["fulfillment_mode"], train["tier"]).to_string())

    banner("tier 分布 × shop_main_cate（前若干组）")
    ct = pd.crosstab(train["shop_main_cate"], train["tier"])
    ct["n"] = ct.sum(axis=1)
    print(ct.sort_values("n", ascending=False).head(20).to_string())

    banner("price / n_attr 统计")
    for col in ["price", "n_attr_rows", "n_attr_types"]:
        print(f"\n{col}:")
        print(train[col].describe().round(3).to_string())
        print("缺失比例:", round(train[col].isna().mean(), 4))

    banner("attr_vals 文本统计")
    for df, nm in [(train, "train"), (test, "test_public")]:
        av = df["attr_vals"]
        nonempty = av.notna() & (av != "")
        lens = av[nonempty].str.len()
        ntokens = av[nonempty].str.split("；").apply(len)
        print(f"[{nm}] 非空比例={nonempty.mean():.4f} 字符长度: min={lens.min() if len(lens) else None} "
              f"中位={lens.median() if len(lens) else None} max={lens.max() if len(lens) else None}; "
              f"词数: 中位={ntokens.median() if len(ntokens) else None} max={ntokens.max() if len(ntokens) else None}")

    banner("类别/ID 基数")
    for col in ["cate_code", "cate_group", "shop_id", "supplier_id", "supplier_region",
                "fulfillment_mode", "supplier_layer", "shop_level", "shop_main_cate",
                "score_band", "onsale_band", "follow_band", "listing_batch"]:
        print(f"{col:20s} nunique={train[col].nunique():6d}  train_nan={train[col].isna().mean():.4f}  "
              f"test_nan={test[col].isna().mean():.4f}")

    banner("is_* 旗标取值分布（train）")
    flags = ["is_old_shop", "is_fashion_shop", "is_quality_shop", "is_potential_shop",
             "is_kol_shop", "is_mid_price_shop", "is_complete", "is_similar"]
    for c in flags:
        print(f"{c}: {train[c].value_counts().to_dict()}")

    banner("shop.csv")
    shop = load("shop.csv")
    print(shop.shape)
    print(shop.dtypes.to_string())
    print("\nmain_category 取值:")
    print(shop["main_category"].value_counts(dropna=False).to_string())
    print("\nregister_batch 分布:")
    print(shop["register_batch"].value_counts().to_string())

    banner("shop_hist_monthly")
    sh = load("shop_hist_monthly.csv")
    print(sh.shape)
    print(sh.dtypes.to_string())
    print("\ncum_sku 分布:")
    print(sh["cum_sku"].describe().round(3).to_string())
    print("\nhit_rate_band 取值:")
    print(sh["hit_rate_band"].value_counts(dropna=False).to_string())
    print("\ngmv_band 取值:")
    print(sh["gmv_band"].value_counts(dropna=False).to_string())

    banner("category_trend")
    ct2 = load("category_trend.csv")
    print(ct2.shape)
    print(ct2.dtypes.to_string())
    print("\ndemand_index 统计:")
    print(ct2["demand_index"].describe().round(3).to_string())
    print("\nsupply_count 统计:")
    print(ct2["supply_count"].describe().round(3).to_string())
    print("\nB01 行 demand_index 缺失比例:", ct2.loc[ct2["batch"] == "B01", "demand_index"].isna().mean().round(4))

    banner("冷启动：test 店铺是否在 train 中出现")
    warm = set(train["shop_id"])
    test_cold = test["shop_id"].map(lambda s: s not in warm)
    print("公开集冷启动比例:", round(test_cold.mean(), 4), "（官方口径 22.0%）")

    banner("baseline_scores")
    bs = load("baseline_scores.csv")
    print(bs.shape)
    print(bs.head().to_string())
    print("\nverdict 分布:", bs["verdict"].value_counts().to_dict())
    print("score 统计:", bs["score"].describe().round(4).to_dict())


if __name__ == "__main__":
    main()
