#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O1 本地评测（只用 Python 3.9+ 标准库，与正式口径逐行一致）。

用法
  # 1) 从 train.csv 按批次切出验证集（默认最后一批 B05 作验证）
  python3 evaluate.py --make-val [--train train.csv] [--val-batch B05]
      → train_sub.csv（前段训练集，含标签）
      → val_items.csv（验证集，去标签）
      → val_truth.csv（item_id,tier,group，组织者口径；group=shop_main_cate）

  # 2) 评测（默认严格：覆盖 / 唯一 / score∈[0,1] / decision 合法）
  python3 evaluate.py --pred val_pred.csv --truth val_truth.csv [--train train_sub.csv]
  # 如只想看官方评分器的容错行为，可加 --allow-incomplete
  python3 evaluate.py --pred val_pred.csv --truth val_truth.csv --allow-incomplete

预测文件列：item_id,score,decision（默认严格模式三列齐全，decision ∈ 放行/驳回；
加 --allow-incomplete 时 decision 可缺省，按 score≥0.5 推）。
预测文件必须为 UTF-8 编码；表头不区分大小写、忽略首尾空格（建议小写）。

口径（与终评评分口径完全一致）：
  1) 分层 NDCG（随机归一化）：按 group（shop_main_cate）分组，组内 NDCG@K（默认 K=100，
     组不足 K 取全组），增益 gain(S,A,B,C,D)=(15,7,3,1,0)，<20 品并组后各组等权宏平均。
     **每组先减去随机排序的期望值再做归一化**：
         NDCG_adj = (NDCG - E[NDCG_随机]) / (1 - E[NDCG_随机])
     其中 E[NDCG_随机] 由该组增益分布精确算出。因此"组内全同分"= 0 分，
     排得比随机还差是负分；没有任何增益差异（全 D）的组不参与宏平均。
  2) 冷启动子分：训练期零历史的店铺（shop_id 不在训练集里）单独成池，
     算一个全池 NDCG_adj，占 NDCG 部分的 COLD_WEIGHT（默认 0.2）。
  3) 决策得分（40%）：max(0, (L_地板 - L_本方案) / (L_地板 - L_全对))
     地板 = 全托管放行、其余驳回；地板 = 0 分，全对 = 1 分，低于地板记 0（不产生负决策分）。
     原始决策分 (L_全放行 - L_本方案) / (L_全放行 - L_全对) 仍会打印，仅供对照，不计分。
  4) 客观总分 = 0.6 × [(1-w)·NDCG_adj_全 + w·NDCG_adj_冷启动] + 0.4 × 决策得分

缺失处理（两套评测一致，绝不使用真值兜底）
  - 默认严格：缺行 / 重复 / 多出 item_id / score 非有限数或越界 / decision 非法 →
    直接报错退出，与终评先跑 validate_submission.py 的行为一致；
  - 加 --allow-incomplete 才启用官方评分器的容错：
    真值商品在预测里缺 score → score 记 -1.0，排序落到最后；decision **强制记「放行」**
    并计入代价（即使文件里给该行写了「驳回」也不采用）；
    有 score 但缺 decision / decision 非法 → decision = 放行 if score >= 0.5 else 驳回；
    非法 score（非数值 / 空 / NaN / inf）→ 视为缺失；重复 item_id → 后一行完整覆盖前一行
    （后一行的 score 非法则视为缺失，后一行没写 decision 则按最终 score 重新推断）；
    预测里多出真值以外的 item_id → 忽略（不参与排序与计分）；
  - 预测为空 / item_id 全部不匹配 → 两种模式都直接报错退出，不给 0 分兜底。
"""
import argparse
import csv
import math
import os
import sys

TIERS = ["S", "A", "B", "C", "D"]
GAIN = {"S": 15, "A": 7, "B": 3, "C": 1, "D": 0}
DECISION_COST = {("S", "驳回"): 12, ("A", "驳回"): 8, ("B", "驳回"): 2,
                 ("B", "放行"): 2, ("C", "放行"): 3, ("D", "放行"): 4,
                 ("S", "放行"): 0, ("A", "放行"): 0, ("C", "驳回"): 0, ("D", "驳回"): 0}
MIN_GROUP = 20         # 组内样本 < 20 并入"其他"
K = 100
COLD_WEIGHT = 0.2      # 冷启动子分在 NDCG 部分里的权重
MISSING_SCORE = -1.0   # 缺失行排序落到最后
MISSING_DECISION = "放行"
FLOOR_MODE = "全托管"  # 单字段地板：全托管→放行，其余→驳回


class ReadError(Exception):
    """文件读取/编码错误（给选手的干净报错，不打印回溯）。"""


def read_table(path):
    """返回 (列名, 行字典)；列名去空白并转小写，值去首尾空白。

    表头大小写不敏感（Item_ID / item_id 等价）；文件必须是 UTF-8（含 BOM 可读）。
    非 UTF-8（例如 Excel 直接另存的 GBK CSV）抛出 ReadError，由调用方给出干净提示。
    """
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


def read_csv(path):
    return read_table(path)[1]


def strict_validate(path, truth):
    """与 validate_submission.py 等价的严格检查；返回错误列表。"""
    errs = []
    try:
        cols, rows = read_table(path)
    except (OSError, ReadError) as e:
        return ["无法读取预测文件: %s" % e]
    for c in ("item_id", "score", "decision"):
        if c not in cols:
            errs.append("缺少列: %s（实际表头: %s）" % (c, ",".join(cols)))
    seen = set()
    dup = set()
    for n, r in enumerate(rows, 1):
        i = r.get("item_id", "")
        if not i:
            errs.append("第 %d 行 item_id 为空" % n)
            continue
        if i in seen:
            dup.add(i)
        seen.add(i)
        if i not in truth:
            errs.append("%s 不在真值清单中（多出的行）" % i)
        try:
            v = float(r.get("score", ""))
            if not math.isfinite(v):
                errs.append("%s score 非有限数: %r（NaN/inf 不允许）" % (i, r.get("score")))
            elif not (0.0 <= v <= 1.0):
                errs.append("%s score 越界: %s（要求 0~1）" % (i, v))
        except (TypeError, ValueError):
            errs.append("%s score 不可解析: %r" % (i, r.get("score")))
        if r.get("decision") not in ("放行", "驳回"):
            errs.append("%s decision 非法: %r（只允许 放行/驳回）" % (i, r.get("decision")))
    if dup:
        errs.append("item_id 重复: %s" % ", ".join(sorted(dup)[:5]))
    missing = [i for i in truth if i not in seen]
    if missing:
        errs.append("缺少 %d 行（示例: %s）" % (len(missing), ", ".join(missing[:5])))
    return errs


def load_cost_matrix(path=None):
    """读取随包 cost_matrix.csv；缺文件/缺行时回退到内置默认值。"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_matrix.csv")
    cost = dict(DECISION_COST)
    try:
        for r in read_csv(path):
            tier, dec = r.get("true_tier", ""), r.get("decision", "")
            if tier in TIERS and dec in ("放行", "驳回"):
                cost[(tier, dec)] = int(round(float(r.get("cost", "0"))))
    except (OSError, ValueError):
        pass
    return cost


COST = load_cost_matrix()


def ndcg_at_k(order, truth, k=K):
    gains = [GAIN[truth[i]] for i in order[:k]]
    dcg = sum(g / math.log2(j + 2) for j, g in enumerate(gains))
    ideal = sorted((GAIN[t] for t in truth.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(j + 2) for j, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def expected_ndcg(truth, k=K):
    """随机排序下的 NDCG 期望（等概率排列，逐位置线性期望，无近似）。"""
    vals = list(truth.values())
    n = len(vals)
    ideal = sorted((GAIN[t] for t in vals), reverse=True)[:k]
    idcg = sum(g / math.log2(j + 2) for j, g in enumerate(ideal))
    if idcg <= 0:
        return 0.0
    mean_gain = sum(GAIN[t] for t in vals) / float(n)
    edcg = sum(mean_gain / math.log2(j + 2) for j in range(min(k, n)))
    return edcg / idcg


def to_adj(ndcg, rand):
    """随机归一化：随机水平 = 0，完美排序 = 1。"""
    if rand >= 1.0 - 1e-9:
        return 0.0
    return (ndcg - rand) / (1.0 - rand)


def eval_stratified(truth, score, group, min_group=MIN_GROUP, k=K):
    """truth: {id: tier}; score: {id: float}; group: {id: group}

    返回 (macro_adj, macro_raw, per, skipped)：
      per[g] = (ndcg, rand, adj, n)；增益全为 0 的组无排序信息，不参与宏平均。
    """
    buckets = {}
    for i in truth:
        buckets.setdefault(group.get(i, "ALL"), []).append(i)
    merged = {g: v for g, v in buckets.items() if len(v) >= min_group}
    small = [i for g, v in buckets.items() if len(v) < min_group for i in v]
    if small:
        merged["其他"] = small
    per, skipped = {}, []
    for g, ids in merged.items():
        st = {i: truth[i] for i in ids}
        order = sorted(ids, key=lambda i: (-score.get(i, MISSING_SCORE), i))
        raw = ndcg_at_k(order, st, k)
        rand = expected_ndcg(st, k)
        if rand >= 1.0 - 1e-9:            # 全组同档 / 增益全 0：无排序信号
            skipped.append(g)
            per[g] = (raw, rand, 0.0, len(ids))
            continue
        per[g] = (raw, rand, to_adj(raw, rand), len(ids))
    used = [v[2] for g, v in per.items() if g not in skipped]
    macro_adj = sum(used) / len(used) if used else 0.0
    macro_raw = (sum(v[0] for g, v in per.items() if g not in skipped) / len(used)) if used else 0.0
    return macro_adj, macro_raw, per, skipped


def eval_cold(truth, score, is_cold, k=K):
    """冷启动子分：训练期零历史店铺的商品池，整体 NDCG 随机归一化。"""
    ids = [i for i in truth if is_cold.get(i)]
    if not ids:
        return None
    st = {i: truth[i] for i in ids}
    order = sorted(ids, key=lambda i: (-score.get(i, MISSING_SCORE), i))
    raw = ndcg_at_k(order, st, k)
    rand = expected_ndcg(st, k)
    return len(ids), raw, rand, to_adj(raw, rand)


def loss_anchors(truth):
    """返回 (L_全放行, L_全驳回, L_全对)。全对 = 逐品取代价较小者。"""
    n = float(len(truth))
    allow = sum(COST[(t, "放行")] for t in truth.values()) / n
    reject = sum(COST[(t, "驳回")] for t in truth.values()) / n
    best = sum(min(COST[(t, "放行")], COST[(t, "驳回")])
               for t in truth.values()) / n
    return allow, reject, best


def floor_decisions(items_rows):
    """单字段地板：全托管→放行，其余→驳回；缺列时用 n_attr_rows 代理。"""
    out = {}
    for r in items_rows:
        i = r.get("item_id", "")
        mode = (r.get("fulfillment_mode") or "").strip()
        if mode:
            out[i] = "放行" if mode == FLOOR_MODE else "驳回"
        else:
            n = (r.get("n_attr_rows") or "").strip()
            if n:
                out[i] = "放行" if n != "-1" else "驳回"
    return out


def eval_decision(truth, decision, score, floor_dec=None):
    allow_loss, reject_loss, best_loss = loss_anchors(truth)
    total = 0.0
    for i, t in truth.items():
        d = decision.get(i)
        if d not in ("放行", "驳回"):
            d = "放行" if score.get(i, MISSING_SCORE) >= 0.5 else "驳回"
        total += COST[(t, d)]
    avg = total / float(len(truth))
    span = allow_loss - best_loss
    dec_raw = (allow_loss - avg) / span if span > 1e-9 else 0.0
    floor_loss = None
    floor_raw = None
    below_floor = None
    if floor_dec and all(i in floor_dec for i in truth):
        floor_loss = sum(COST[(t, floor_dec[i])] for i, t in truth.items()) / float(len(truth))
        floor_raw = (allow_loss - floor_loss) / span if span > 1e-9 else 0.0
        floor_span = floor_loss - best_loss
        dec = max(0.0, (floor_loss - avg) / floor_span) if floor_span > 1e-9 else 0.0
        below_floor = avg > floor_loss + 1e-12
    else:
        dec = dec_raw
    return (avg, dec, dec_raw, (allow_loss, reject_loss, best_loss),
            floor_loss, floor_raw, below_floor)


def report(truth, score, decision, group, min_group=MIN_GROUP, missing=None,
           is_cold=None, cold_weight=COLD_WEIGHT, k=K, floor_dec=None):
    if missing is None:                       # 兜底；正常由 main 在预填前统计后传入
        missing = [i for i in truth if i not in score]
    order_all = sorted(truth, key=lambda i: (-score.get(i, MISSING_SCORE), i))
    ndcg_all = ndcg_at_k(order_all, truth, k)
    macro_adj, macro_raw, per, skipped = eval_stratified(truth, score, group, min_group, k)
    cold = eval_cold(truth, score, is_cold or {}, k)
    (avg_loss, dec_score, dec_raw, anchors,
     floor_loss, floor_raw, _below_floor) = eval_decision(truth, decision, score, floor_dec)

    if cold and cold_weight > 0:
        ndcg_part = (1 - cold_weight) * macro_adj + cold_weight * cold[3]
        detail = "%.4f × (1-%.2f) + %.4f × %.2f" % (macro_adj, cold_weight, cold[3], cold_weight)
    else:
        ndcg_part = macro_adj
        detail = "%.4f（无冷启动子分）" % macro_adj
    total = 0.6 * ndcg_part + 0.4 * dec_score

    print("样本数 %d（缺 score 的 %d 件：score 记 -1 排最后、decision 强制记放行）"
          % (len(truth), len(missing)))
    print("分层 NDCG@%d 原始    : %.4f   （原始值含随机底分，仅供参考，不计分）"
          % (k, macro_raw))
    print("分层 NDCG 随机归一化  : %+.4f   ← 随机排序 = 0，完美排序 = 1" % macro_adj)
    for g, (raw, rand, adj, c) in sorted(per.items(), key=lambda x: -x[1][3]):
        print("    %-28s n=%-4d NDCG=%.4f  E[随机]=%.4f  归一化=%+.4f"
              % (g, c, raw, rand, adj))
    if skipped:
        print("    （无排序信号的组不参与宏平均：%s）" % "、".join(sorted(skipped)))
    if cold:
        print("冷启动子分（%d 件，占 NDCG 部分 %.0f%%）: %+.4f  （NDCG=%.4f / E[随机]=%.4f）"
              % (cold[0], cold_weight * 100, cold[3], cold[1], cold[2]))
    print("全池 NDCG@%d         : %.4f   （参考指标，不计分）" % (k, ndcg_all))
    print("平均误判损失          : %.4f" % avg_loss)
    print("  锚点（按本批真值档位分布算）：全放行 L=%.4f（原始 0 分）｜全对 L=%.4f（1 分）｜全驳回 L=%.4f"
          % (anchors[0], anchors[2], anchors[1]))
    if floor_loss is not None:
        print("  单字段地板：L=%.4f（原始决策分 %+.4f；计分口径 0）" % (floor_loss, floor_raw))
    else:
        print("  [WARN] 缺少 fulfillment_mode / n_attr_rows，决策分按原始口径")
    print("  决策分（原始）= (%.4f - %.4f) / (%.4f - %.4f) = %+.4f"
          % (anchors[0], avg_loss, anchors[0], anchors[2], dec_raw))
    print("  决策增量（计分）= %+.4f   ← 地板=0 / 全对=1；低于地板记 0；占客观分 40%%"
          % dec_score)
    print("NDCG 部分             : %s = %+.4f   ← 占客观分 60%%" % (detail, ndcg_part))
    print("客观总分              : %+.4f" % total)
    if missing:
        print("[WARN] %d 件真值商品在预测里缺失，已按 -1 排序、按放行计入代价（不使用真值兜底）"
              % len(missing))
    return total


def make_val(train_path, val_batch):
    rows = read_csv(train_path)
    tr = [r for r in rows if r["listing_batch"] < val_batch]
    va = [r for r in rows if r["listing_batch"] >= val_batch]
    with open(train_path, newline="", encoding="utf-8-sig") as f:
        cols = [(c or "").strip() for c in csv.DictReader(f).fieldnames]
    with open("train_sub.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(tr)
    with open("val_items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, [c for c in cols if c != "tier"], extrasaction="ignore")
        w.writeheader()
        w.writerows(va)
    with open("val_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "tier", "group"])
        for r in va:
            w.writerow([r["item_id"], r["tier"], r.get("shop_main_cate", "ALL")])
    print("train_sub.csv %d 行；val_items/val_truth %d 行（listing_batch >= %s）"
          % (len(tr), len(va), val_batch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-val", action="store_true")
    ap.add_argument("--train", default="train.csv",
                    help="训练集（用于判定冷启动店铺）；缺省 train.csv")
    ap.add_argument("--val-batch", default="B05",
                    help="验证批次起点（含）：listing_batch >= 该值的批次都进入验证集；"
                         "默认 B05（train 共 B01–B05），传 B04 会验证 B04–B05 两批")
    ap.add_argument("--pred")
    ap.add_argument("--truth")
    ap.add_argument("--items", default=None, help="验证集特征文件（缺省取 val_items.csv，用于取分组键与 shop_id）")
    ap.add_argument("--min-group", type=int, default=MIN_GROUP)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--cold-weight", type=float, default=COLD_WEIGHT,
                    help="冷启动子分在 NDCG 部分里的权重，默认 0.2；0 = 关闭")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="启用官方评分器的容错（缺行/非法值按缺失处理）；默认严格校验")
    a = ap.parse_args()
    if a.make_val:
        make_val(a.train, a.val_batch)
        return
    if not a.pred or not a.truth:
        ap.error("需要 --pred 与 --truth，或使用 --make-val")
    truth_rows = read_csv(a.truth)
    cols = list(truth_rows[0].keys()) if truth_rows else []
    truth = {r["item_id"]: r["tier"] for r in truth_rows}
    if not a.allow_incomplete:
        errs = strict_validate(a.pred, truth)
        if errs:
            print("严格校验失败（与终评 validate_submission.py 一致；如需官方容错加 --allow-incomplete）：")
            for e in errs[:30]:
                print("[FAIL]", e)
            if len(errs) > 30:
                print("... 另有 %d 条" % (len(errs) - 30))
            sys.exit(1)
    group = {r["item_id"]: (r.get("group") or "ALL") for r in truth_rows}
    items_path = a.items or ("val_items.csv" if os.path.exists("val_items.csv") else None)
    shop_of = {}
    floor_dec = {}
    if items_path and os.path.exists(items_path):
        items_rows = read_csv(items_path)
        for r in items_rows:
            group[r["item_id"]] = r.get("shop_main_cate") or "ALL"
            shop_of[r["item_id"]] = r.get("shop_id", "")
        floor_dec = floor_decisions(items_rows)
    if not shop_of:
        if not a.allow_incomplete:
            sys.exit("[FAIL] 找不到验证集特征文件（--items / val_items.csv）：缺少它无法计算单字段地板与冷启动子分，"
                     "评分会偏离官方口径。请用 --items 指定（例如 --items val_items.csv），"
                     "或加 --allow-incomplete 仅查看容错行为。")
        print("[WARN] 找不到验证集特征文件（--items / val_items.csv），冷启动子分按 0 权重处理")
    # 冷启动：训练集里一条记录都没有的店铺
    # 验证本地切分时请显式传 --train train_sub.csv（否则 B05 店铺会被误当成"有历史"）
    train_path = a.train
    warm = set()
    if os.path.exists(train_path):
        warm = {r.get("shop_id", "") for r in read_csv(train_path)}
    else:
        print("[WARN] 找不到 %s，无法判定冷启动店铺" % train_path)
    is_cold = {i: bool(shop_of.get(i)) and shop_of.get(i) not in warm for i in truth}

    score, decision = {}, {}
    try:
        pred_rows = read_csv(a.pred)
    except (OSError, ReadError) as e:
        sys.exit("[FAIL] 无法读取预测文件: %s" % e)
    for r in pred_rows:
        i = r.get("item_id", "")
        if i not in truth:
            continue
        try:
            v = float(r.get("score", ""))
            if not math.isfinite(v):
                raise ValueError
            score[i] = v
        except (TypeError, ValueError):
            score.pop(i, None)        # 后一行覆盖前一行；非法 score 视为缺失
        d = r.get("decision", "")
        if d in ("放行", "驳回"):
            decision[i] = d
        else:
            decision.pop(i, None)     # 后一行没有合法 decision → 按最后一行重新推断
    missing = [i for i in truth if i not in score]   # 必须在预填前统计，否则恒为 0
    if not score:
        sys.exit("预测文件为空或 item_id 全部不匹配真值")
    for i in truth:                   # 缺失行的统一处理（与正式评测一致）
        if i not in score:
            score[i] = MISSING_SCORE
            decision[i] = MISSING_DECISION   # 缺 score 时强制放行，忽略文件里的 decision
        elif i not in decision:
            decision[i] = "放行" if score[i] >= 0.5 else "驳回"
    report(truth, score, decision, group, a.min_group, missing,
           is_cold=is_cold, cold_weight=a.cold_weight, k=a.k,
           floor_dec=floor_dec or None)


if __name__ == "__main__":
    main()
