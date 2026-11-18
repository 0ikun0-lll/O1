#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装一个本地可用的 `--data-dir`，用来模拟终评沙箱（纯标准库）。

背景：终评沙箱里的 `--data-dir` 是 `train.csv + test.csv + 随包文件`；
而赛题包只给 `train.csv` / `test_public.csv`，没有 `test.csv`。
直接拿 `test_public.csv` 改名、却不切训练集，会把验证批的标签带进训练，
得到虚高的本地分数。本脚本按官方口径一次性把目录拼好。

用法:
  # ① 验证集模式（默认）：按批次切 train.csv（与 evaluate.py --make-val 同口径）
  python3 make_local_datadir.py --src . --out /tmp/o1_val --mode val --val-batch B05
  #    → /tmp/o1_val/{train.csv,test.csv,shop.csv,...}
  #    → /tmp/o1_val_truth.csv（真值，放在 data-dir **外**，绝不能喂给 predict.py）

  python3 predict.py --data-dir /tmp/o1_val --out /tmp/val_pred.csv
  python3 evaluate.py --pred /tmp/val_pred.csv --truth /tmp/o1_val_truth.csv \
      --items /tmp/o1_val/test.csv --train /tmp/o1_val/train.csv

  # ② 公开集模式：test.csv = test_public.csv，train.csv = 完整 train.csv
  #    （公开集无标签，只能跑通链路，不能本地打分）
  python3 make_local_datadir.py --src . --out /tmp/o1_pub --mode public
  python3 predict.py --data-dir /tmp/o1_pub --out /tmp/public_pred.csv

约束（与终评一致）:
  - `baseline_scores.csv` 只放训练集行；
  - `shop_hist_monthly.csv` / `category_trend.csv` 只到当前测试批次为止；
  - 真值单独存放，不放进 data-dir。
"""
import argparse
import csv
import os
import shutil
import sys

# 随包下发的静态文件（存在才拷）
STATIC = [
    "shop.csv",
    "shop_hist_monthly.csv",
    "category_trend.csv",
    "cost_matrix.csv",
    "baseline_scores.csv",
    "audit_reason_dict.csv",
    "human_review_sample.csv",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = [(c or "").strip() for c in (rd.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rd]
    return cols, rows


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def max_batch(rows):
    vals = [r.get("listing_batch", "") for r in rows]
    vals = [v for v in vals if v]
    return max(vals) if vals else None


def filter_batch(path, dst, until):
    """把 shop_hist_monthly / category_trend 截到 batch <= until。"""
    cols, rows = read_csv(path)
    if "batch" not in cols:
        shutil.copyfile(path, dst)
        return None, None
    kept = [r for r in rows if r.get("batch", "") <= until]
    write_csv(dst, cols, kept)
    return len(rows), len(kept)


def build(src, out, mode, val_batch):
    os.makedirs(out, exist_ok=True)

    if mode == "val":
        cols, rows = read_csv(os.path.join(src, "train.csv"))
        if "listing_batch" not in cols:
            sys.exit("[FAIL] train.csv 缺少 listing_batch 列")
        tr = [r for r in rows if r["listing_batch"] < val_batch]
        va = [r for r in rows if r["listing_batch"] >= val_batch]
        if not tr or not va:
            sys.exit("[FAIL] --val-batch %s 切不出训练/验证两段（train 批次 %s..%s）"
                     % (val_batch, min(r["listing_batch"] for r in rows),
                        max(r["listing_batch"] for r in rows)))
        test_cols = [c for c in cols if c != "tier"]
        va_feat = [{k: r.get(k, "") for k in test_cols} for r in va]
        write_csv(os.path.join(out, "train.csv"), cols, tr)
        write_csv(os.path.join(out, "test.csv"), test_cols, va_feat)
        truth_path = out.rstrip("/\\") + "_truth.csv"
        truth_cols = ["item_id", "tier", "group"]
        truth_rows = [{"item_id": r["item_id"], "tier": r.get("tier", ""),
                       "group": r.get("shop_main_cate") or "ALL"} for r in va]
        write_csv(truth_path, truth_cols, truth_rows)
        train_ids = {r["item_id"] for r in tr}
        until = max_batch(va)
        print("验证集模式：train %d 行（%s..）；test %d 行（%s..）；真值 → %s"
              % (len(tr), min(r["listing_batch"] for r in tr), len(va), until, truth_path))
        print("打分：python3 evaluate.py --pred <pred.csv> --truth %s "
              "--items %s/test.csv --train %s/train.csv" % (truth_path, out, out))
    elif mode == "public":
        shutil.copyfile(os.path.join(src, "train.csv"), os.path.join(out, "train.csv"))
        shutil.copyfile(os.path.join(src, "test_public.csv"), os.path.join(out, "test.csv"))
        train_ids = {r["item_id"] for r in read_csv(os.path.join(out, "train.csv"))[1]}
        until = max_batch(read_csv(os.path.join(out, "test.csv"))[1])
        print("公开集模式：train %d 行；test 3,000 行（%s）；公开集无标签，只能跑通链路"
              % (len(train_ids), until))
    else:
        sys.exit("[FAIL] 未知 --mode %s" % mode)

    # 静态文件
    for name in STATIC:
        p = os.path.join(src, name)
        if not os.path.exists(p):
            continue
        if name in ("shop_hist_monthly.csv", "category_trend.csv") and until:
            before, after = filter_batch(p, os.path.join(out, name), until)
            if before is not None:
                print("  %-24s batch <= %s：%d → %d 行" % (name, until, before, after))
        elif name == "baseline_scores.csv":
            cols, rows = read_csv(p)
            kept = [r for r in rows if r.get("item_id") in train_ids]
            write_csv(os.path.join(out, name), cols, kept)
            print("  %-24s 只保留训练集行：%d 行" % (name, len(kept)))
        else:
            shutil.copyfile(p, os.path.join(out, name))
    print("已生成 data-dir：%s" % out)
    print("提醒：predict.py 只应读取 --data-dir；真值文件不要拷进去。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".", help="解压后的赛题包目录（含 train.csv / test_public.csv）")
    ap.add_argument("--out", required=True, help="输出的 data-dir（建议放 /tmp，不要放代码包内）")
    ap.add_argument("--mode", choices=["val", "public"], default="val")
    ap.add_argument("--val-batch", default="B05",
                    help="验证批起点（含）：B05 → 验证 B05；B04 → 验证 B04–B05")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.src, "train.csv")):
        sys.exit("[FAIL] %s 下找不到 train.csv" % a.src)
    build(a.src, a.out, a.mode, a.val_batch)


if __name__ == "__main__":
    main()
