#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征工程模块：只读 data-dir，按赛题口径做批次时间过滤，生成可复现的特征矩阵。

最终 predict.py 会复用本模块（只依赖 pandas / numpy / scipy / scikit-learn / lightgbm / xgboost）。

关键口径（务必保持）：
  * shop_hist_monthly / category_trend.demand_index 只能取 batch <= 当前 listing_batch 的行；
  * category_trend.supply_count 取 batch == 当前 listing_batch 的当期供给；
  * 不使用 item_id/shop_id/supplier_id/cate_code 的编号大小或行号；
  * 不用 listing_batch 做月份/季节特征（batch_int 只用于过滤与 asof 排序，不作为模型特征）。
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

GAIN = {"S": 15.0, "A": 7.0, "B": 3.0, "C": 1.0, "D": 0.0}
TIERS = ["S", "A", "B", "C", "D"]

BATCH_INT = {"B01": 1, "B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7}

CAT_COLS = [
    "fulfillment_mode",
    "supplier_region",
    "supplier_layer",
    "shop_level",
    "shop_main_cate",
    "cate_group",
    "cate_code",
    "score_band",
    "onsale_band",
    "follow_band",
    "register_batch",
]

TE_COLS = [
    "cate_code",
    "cate_group",
    "shop_main_cate",
    "supplier_region",
    "supplier_layer",
    "shop_level",
    "fulfillment_mode",
]

NUM_COLS = [
    "price",
    "price_log",
    "has_price_ref",
    "n_attr_rows",
    "n_attr_types",
    "has_attr",
    "attr_len",
    "attr_tokens",
    "is_complete",
    "score_band_num",
    "onsale_num",
    "follow_num",
    "is_old_shop",
    "is_fashion_shop",
    "is_quality_shop",
    "is_potential_shop",
    "is_kol_shop",
    "is_mid_price_shop",
    "cum_sku",
    "has_shop_hist",
    "hit_rate_num",
    "gmv_band",
    "demand_index",
    "demand_missing",
    "supply_log",
    "demand_ratio",
    "demand_slope",
    "price_rel",
]

FLAG_COLS = [
    "is_old_shop",
    "is_fashion_shop",
    "is_quality_shop",
    "is_potential_shop",
    "is_kol_shop",
    "is_mid_price_shop",
]


def _batch_int(b):
    return BATCH_INT.get(str(b).strip(), 0)


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def parse_score_band(v):
    if not isinstance(v, str):
        return np.nan
    s = v.strip()
    if s == "" or s == "miss":
        return np.nan
    m = re.match(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def parse_b_band(v):
    if not isinstance(v, str):
        return np.nan
    s = v.strip()
    if s == "" or s == "miss":
        return np.nan
    m = re.match(r"^b(\d+)$", s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_hit_rate(v):
    if not isinstance(v, str):
        return np.nan
    s = v.strip()
    if s in ("", "无历史"):
        return np.nan
    m = re.match(r"^(\d+)-(\d+)%$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def load_tables(data_dir):
    """读取 data-dir 内的主表与外部表。test.csv 可为公开集或验证集（无 tier）。"""
    train = read_csv(os.path.join(data_dir, "train.csv"))
    test = read_csv(os.path.join(data_dir, "test.csv"))
    shop = read_csv(os.path.join(data_dir, "shop.csv"))
    shop_hist = read_csv(os.path.join(data_dir, "shop_hist_monthly.csv"))
    cat_trend = read_csv(os.path.join(data_dir, "category_trend.csv"))
    return train, test, shop, shop_hist, cat_trend


def _latest_join(df, right, by, value_cols, on_col="batch_int"):
    """按 by 分组，取 on_col <= 当前行 on_col 的最新一条快照；无则 NaN。"""
    left = pd.DataFrame(
        {
            "__orig": np.arange(len(df)),
            by: df[by].values,
            on_col: df[on_col].values,
        }
    )
    rr = right[[by, on_col] + value_cols].rename(columns={on_col: "rb"})
    m = left.merge(rr, on=by, how="left")
    m = m[m["rb"].isna() | (m["rb"] <= m[on_col])]
    m = m.sort_values(["__orig", "rb"], ascending=[True, False], kind="mergesort")
    m = m.drop_duplicates("__orig", keep="first").set_index("__orig")
    out = m[value_cols].reindex(np.arange(len(df)))
    return out.reset_index(drop=True)


def attach_external(df, shop, shop_hist, cat_trend):
    """按时间口径合并 shop.csv / shop_hist_monthly.csv / category_trend.csv。"""
    out = df.copy()
    out["batch_int"] = out["listing_batch"].map(_batch_int)

    # shop 静态：register_batch（main_category 全空，不使用）
    out = out.merge(shop[["shop_id", "register_batch"]], on="shop_id", how="left")

    # 店铺历史：取 batch <= 当前批次 的最新一条快照
    sh = shop_hist.copy()
    sh["batch_int"] = sh["batch"].map(_batch_int)
    sh = sh[["shop_id", "batch_int", "cum_sku", "hit_rate_band", "gmv_band"]]
    hist = _latest_join(out, sh, "shop_id", ["cum_sku", "hit_rate_band", "gmv_band"])
    out = pd.concat([out.reset_index(drop=True), hist], axis=1)

    # 类目需求指数：batch <= 当前批次（demand_index 在 batch==Bk 反映 Bk-1 结算，可用）
    ct = cat_trend.copy()
    ct["batch_int"] = ct["batch"].map(_batch_int)
    ct_d = ct[["cate_code", "batch_int", "demand_index"]]
    trend = _latest_join(out, ct_d, "cate_code", ["demand_index"])
    out = pd.concat([out.reset_index(drop=True), trend], axis=1)

    # 类目当期供给：batch == 当前批次
    ct_s = ct[["cate_code", "batch_int", "supply_count"]].rename(
        columns={"supply_count": "supply_count"}
    )
    out = out.merge(ct_s, on=["cate_code", "batch_int"], how="left")

    # 类目需求的滞后值与趋势（demand_lag1 = 上一批，lag2 = 上上批，均早于当前批次，允许）
    for lag, col in [(1, "demand_lag1"), (2, "demand_lag2")]:
        lagdf = ct[["cate_code", "batch_int", "demand_index"]].copy()
        lagdf["batch_int"] = lagdf["batch_int"] + lag
        lagdf = lagdf.rename(columns={"demand_index": col})
        out = out.merge(lagdf, on=["cate_code", "batch_int"], how="left")

    out = out.drop(columns=["batch_int"])
    return out


def _num_features(df, price_med=None):
    out = pd.DataFrame(index=df.index)
    price = _to_num(df["price"])
    out["price"] = price
    out["price_log"] = np.log1p(price.clip(lower=0.0))
    out["has_price_ref"] = _to_num(df["has_price_ref"])

    n_rows = _to_num(df["n_attr_rows"]).fillna(-1.0)
    n_types = _to_num(df["n_attr_types"]).fillna(-1.0)
    out["n_attr_rows"] = n_rows
    out["n_attr_types"] = n_types
    out["has_attr"] = (n_rows >= 0.0).astype(np.int8)
    av = df["attr_vals"].fillna("")
    out["attr_len"] = av.str.len().astype(np.float64)
    out["attr_tokens"] = av.str.split("；").apply(lambda x: len([t for t in x if t])).astype(np.float64)
    out["is_complete"] = _to_num(df["is_complete"])

    out["score_band_num"] = df["score_band"].map(parse_score_band)
    out["onsale_num"] = df["onsale_band"].map(parse_b_band)
    out["follow_num"] = df["follow_band"].map(parse_b_band)

    for c in FLAG_COLS:
        out[c] = _to_num(df[c])

    out["cum_sku"] = _to_num(df.get("cum_sku"))
    out["has_shop_hist"] = (out["cum_sku"].fillna(0.0) > 0.0).astype(np.int8)
    out["hit_rate_num"] = df.get("hit_rate_band").map(parse_hit_rate)
    out["gmv_band"] = _to_num(df.get("gmv_band"))
    out["demand_index"] = _to_num(df.get("demand_index"))
    out["demand_missing"] = out["demand_index"].isna().astype(np.int8)
    supply = _to_num(df.get("supply_count"))
    out["supply_log"] = np.log1p(supply.clip(lower=0.0))

    demand = out["demand_index"]
    lag1 = _to_num(df.get("demand_lag1"))
    lag2 = _to_num(df.get("demand_lag2"))
    out["demand_ratio"] = (demand / lag1.replace(0.0, np.nan)).clip(0.05, 20.0)
    out["demand_slope"] = demand - lag2

    if price_med is not None:
        out["price_rel"] = price / price_med.replace(0.0, np.nan)
    else:
        out["price_rel"] = np.nan
    return out


class CatEncoder:
    """类别 -> 整数编码；未见过的类别 -> -1（LightGBM 视负数类别为缺失）。"""

    def __init__(self):
        self.map = {}

    def fit(self, series):
        uniq = sorted(series.astype(str).str.strip().unique())
        self.map = {v: i for i, v in enumerate(uniq)}
        return self

    def transform(self, series):
        s = series.astype(str).str.strip()
        return s.map(self.map).fillna(-1).astype(np.int32).values


class TextEncoder:
    def __init__(self, n_components=48, min_df=2, seed=0):
        self.n_components = n_components
        self.min_df = min_df
        self.seed = seed
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: [t for t in x.split("；") if t],
            preprocessor=lambda x: x,
            lowercase=True,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(
            n_components=n_components, algorithm="randomized", random_state=seed, n_iter=7
        )

    def fit(self, texts):
        X = self.tfidf.fit_transform(texts.fillna(""))
        self.svd.fit(X)
        return self

    def transform(self, texts):
        X = self.tfidf.transform(texts.fillna(""))
        return self.svd.transform(X)


class FeatureBuilder:
    """在 train 上拟合全部编码器，并分别生成 train（OOF 目标编码）与 test 特征。"""

    def __init__(self, te_cols=TE_COLS, cat_cols=CAT_COLS, num_cols=NUM_COLS,
                 smoothing=20.0, n_svd=48, min_df=2, n_folds=5, seed=0):
        self.te_cols = list(te_cols)
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.smoothing = smoothing
        self.n_svd = n_svd
        self.min_df = min_df
        self.n_folds = n_folds
        self.seed = seed
        self.cat_encoders = {}
        self.text_enc = TextEncoder(n_components=n_svd, min_df=min_df, seed=seed)
        self.te_stats = {}  # col -> DataFrame(te_gain, te_pS, te_pA)
        self.te_priors = {}  # col -> (prior_gain, prior_pS, prior_pA)
        self.feature_names = None
        self.cate_price_median = None

    @staticmethod
    def _targets(train_df):
        tier = train_df["tier"].astype(str).str.strip()
        gain = tier.map(GAIN).astype(np.float64)
        is_s = (tier == "S").astype(np.float64)
        is_a = (tier == "A").astype(np.float64)
        return gain, is_s, is_a

    @staticmethod
    def _smoothed_stats(cat, gain, is_s, is_a, smoothing):
        g = pd.DataFrame({"c": cat, "g": gain, "s": is_s, "a": is_a})
        agg = g.groupby("c").agg(
            g_sum=("g", "sum"),
            s_sum=("s", "sum"),
            a_sum=("a", "sum"),
            cnt=("g", "count"),
        )
        sm = float(smoothing)
        agg["te_gain"] = (agg["g_sum"] + gain.mean() * sm) / (agg["cnt"] + sm)
        agg["te_pS"] = (agg["s_sum"] + is_s.mean() * sm) / (agg["cnt"] + sm)
        agg["te_pA"] = (agg["a_sum"] + is_a.mean() * sm) / (agg["cnt"] + sm)
        return agg[["te_gain", "te_pS", "te_pA"]]

    def fit(self, train_df):
        train_df = train_df.reset_index(drop=True)
        p = _to_num(train_df["price"])
        med = p.groupby(train_df["cate_code"].astype(str).str.strip()).median()
        self.cate_price_median = med
        self.global_price_median = p.median()
        for c in self.cat_cols:
            enc = CatEncoder()
            enc.fit(train_df[c])
            self.cat_encoders[c] = enc
        self.text_enc.fit(train_df["attr_vals"])
        gain, is_s, is_a = self._targets(train_df)
        for c in self.te_cols:
            cat = train_df[c].astype(str).str.strip()
            self.te_stats[c] = self._smoothed_stats(cat, gain, is_s, is_a, self.smoothing)
            self.te_priors[c] = (gain.mean(), is_s.mean(), is_a.mean())
        return self

    def _base_features(self, df):
        med = None
        if self.cate_price_median is not None:
            med = df["cate_code"].astype(str).str.strip().map(self.cate_price_median)
            med = med.fillna(self.global_price_median)
        num = _num_features(df, med)
        for c in self.num_cols:
            if c not in num.columns:
                num[c] = np.nan
        num = num[self.num_cols].reset_index(drop=True)

        cat = pd.DataFrame(index=df.index)
        for c in self.cat_cols:
            cat[c + "_cat"] = self.cat_encoders[c].transform(df[c])
        cat = cat.reset_index(drop=True)

        svd = self.text_enc.transform(df["attr_vals"])
        svd = pd.DataFrame(
            svd, columns=["svd_%02d" % i for i in range(svd.shape[1])]
        ).reset_index(drop=True)

        return num, cat, svd

    def _te_features(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.te_cols:
            cat = df[c].astype(str).str.strip()
            st = self.te_stats[c]
            out[c + "_te_gain"] = cat.map(st["te_gain"]).fillna(self.te_priors[c][0]).values
            out[c + "_te_pS"] = cat.map(st["te_pS"]).fillna(self.te_priors[c][1]).values
            out[c + "_te_pA"] = cat.map(st["te_pA"]).fillna(self.te_priors[c][2]).values
        return out.reset_index(drop=True)

    def transform_train(self, train_df):
        train_df = train_df.reset_index(drop=True)
        num, cat, svd = self._base_features(train_df)
        gain, is_s, is_a = self._targets(train_df)

        te = pd.DataFrame(index=train_df.index)
        for c in self.te_cols:
            te[c + "_te_gain"] = np.nan
            te[c + "_te_pS"] = np.nan
            te[c + "_te_pA"] = np.nan
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        # 逐列 OOF（清晰优先；样本量小，开销可接受）
        for c in self.te_cols:
            col = train_df[c].astype(str).str.strip()
            for tr_idx, va_idx in kf.split(train_df):
                st = self._smoothed_stats(
                    col.iloc[tr_idx].reset_index(drop=True),
                    gain.iloc[tr_idx].reset_index(drop=True),
                    is_s.iloc[tr_idx].reset_index(drop=True),
                    is_a.iloc[tr_idx].reset_index(drop=True),
                    self.smoothing,
                )
                va_cat = col.iloc[va_idx]
                te.loc[va_idx, c + "_te_gain"] = va_cat.map(st["te_gain"]).fillna(self.te_priors[c][0]).values
                te.loc[va_idx, c + "_te_pS"] = va_cat.map(st["te_pS"]).fillna(self.te_priors[c][1]).values
                te.loc[va_idx, c + "_te_pA"] = va_cat.map(st["te_pA"]).fillna(self.te_priors[c][2]).values
        te = te.reset_index(drop=True)

        X = pd.concat([num, cat, svd, te], axis=1)
        self.feature_names = list(X.columns)
        return X, gain

    def transform_test(self, test_df):
        test_df = test_df.reset_index(drop=True)
        num, cat, svd = self._base_features(test_df)
        te = self._te_features(test_df)
        X = pd.concat([num, cat, svd, te], axis=1)
        if self.feature_names is not None:
            X = X[self.feature_names]
        return X


def build_features(train_df, test_df, **kwargs):
    """拟合并返回 (X_train, y_gain, X_test, cat_feature_names)。"""
    fb = FeatureBuilder(**kwargs)
    fb.fit(train_df)
    X_train, y_gain = fb.transform_train(train_df)
    X_test = fb.transform_test(test_df)
    cat_names = [c + "_cat" for c in fb.cat_cols]
    return X_train, y_gain, X_test, cat_names, fb
