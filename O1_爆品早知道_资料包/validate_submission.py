#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交格式校验（公开榜 / 隐藏榜通用）。只用 Python 3.9+ 标准库。

用法:
  python3 validate_submission.py --pred submission_public.csv --items test_public.csv
  python3 validate_submission.py --pred submission.csv --items test.csv      # 终评沙箱内

检查项:
  1) 表头包含 item_id, score, decision（不区分大小写、忽略首尾空格）
  2) item_id 覆盖 items 文件全部行，无重复、无多余
  3) score 可解析为 [0,1] 浮点
  4) decision ∈ {放行, 驳回}
  5) 预测文件必须是 UTF-8 编码（含 BOM 可读）；GBK/ANSI 会给出干净报错
退出码: 0 通过；1 有问题（打印逐条原因）。
"""
import argparse
import csv
import math
import sys

REQUIRED = ["item_id", "score", "decision"]


class ReadError(Exception):
    """文件读取/编码错误（给选手的干净报错，不打印回溯）。"""


def read_csv(path):
    """列名去空白并转小写（表头大小写不敏感）；非 UTF-8 抛 ReadError。"""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            raw_cols = list(rd.fieldnames or [])
            colmap = {(c or ""): (c or "").strip().lower() for c in raw_cols}
            cols = [colmap[c] for c in raw_cols]
            rows = [{colmap.get((k or ""), (k or "").strip().lower()): (v or "").strip()
                     for k, v in r.items()} for r in rd]
    except UnicodeDecodeError:
        raise ReadError("文件不是 UTF-8 编码（可能是 Excel 另存的 GBK/ANSI）。"
                        "请用「另存为 → CSV UTF-8」重新导出后重试：%s" % path)
    return cols, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--items", required=True, help="待预测清单（test_public.csv / test.csv）")
    a = ap.parse_args()

    errs = []
    icols, items = read_csv(a.items)
    need_ids = [r["item_id"] for r in items]
    need = set(need_ids)

    try:
        pcols, pred = read_csv(a.pred)
    except Exception as e:  # noqa: BLE001
        print("[FAIL] 无法读取预测文件: %s" % e)
        sys.exit(1)
    for c in REQUIRED:
        if c not in pcols:
            errs.append("缺少列: %s（实际表头: %s）" % (c, ",".join(pcols)))

    seen = set()
    for i, r in enumerate(pred, 1):
        pid = r.get("item_id", "")
        if not pid:
            errs.append("第 %d 行 item_id 为空" % i)
            continue
        if pid in seen:
            errs.append("item_id 重复: %s" % pid)
        seen.add(pid)
        try:
            v = float(r.get("score", ""))
            if not math.isfinite(v):
                errs.append("%s score 非有限数: %r（NaN/inf 不允许）" % (pid, r.get("score")))
            elif not (0.0 <= v <= 1.0):
                errs.append("%s score 越界: %s（要求 0~1）" % (pid, v))
        except (TypeError, ValueError):
            errs.append("%s score 不可解析: %r" % (pid, r.get("score")))
        if r.get("decision") not in ("放行", "驳回"):
            errs.append("%s decision 非法: %r（只允许 放行/驳回）" % (pid, r.get("decision")))

    missing = [i for i in need_ids if i not in seen]
    extra = [i for i in seen if i not in need]
    if missing:
        errs.append("缺少 %d 行（示例: %s）" % (len(missing), ", ".join(missing[:5])))
    if extra:
        errs.append("多出 %d 行（示例: %s）" % (len(extra), ", ".join(sorted(extra)[:5])))

    print("预测行数 %d / 需要 %d；唯一 item_id %d" % (len(pred), len(need_ids), len(seen)))
    if errs:
        for e in errs[:30]:
            print("[FAIL]", e)
        if len(errs) > 30:
            print("... 另有 %d 条" % (len(errs) - 30))
        sys.exit(1)
    print("[OK] 提交格式校验通过")


if __name__ == "__main__":
    main()
