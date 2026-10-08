"""匹配质量评测脚本：``python eval_match.py``

失物招领的匹配是**排序问题**：系统不需要对每一对都给出答案，只需要保证
「真正的配对」排在最前面，且分数与「无关配对」之间有可用的分界线。

本脚本用 ``seed.py`` 生成的标注数据做量化评测：

* **P@1 / Recall@3**：8 组真实配对中，正确对象排在第 1 名 / 前 3 名的比例；
* **正负样本分布**：真实配对的分数区间 vs. 干扰配对的分数区间；
* **最佳阈值**：扫描阈值，给出 F1 最高的工作点，并说明精确率/召回率取舍；
* **通道贡献**：图像通道与文本通道各自的区分能力（含「同图不同物」压力测试）。

用法：
    python eval_match.py
    python eval_match.py --details      # 打印每一对的明细
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from lostfound import db, matcher  # noqa: E402
from lostfound.image_features import item_image_similarity  # noqa: E402
from lostfound.text_features import build_corpus, text_similarity  # noqa: E402

# 人工标注：seed.py 中 8 组「同一件物品」的配对（按标题关键字匹配）
PAIR_MARKERS = [
    ("黑色双肩包", "图书馆捡到黑色背包"),
    ("粉色自动折叠雨伞", "捡到一把粉色雨伞"),
    ("白色无线蓝牙耳机", "A 区 203 教室拾到白色耳机盒"),
    ("校园一卡通丢失", "三食堂捡到蓝色校园卡"),
    ("墨绿色保温杯", "体育馆捡到绿色水杯"),
    ("银色笔记本电脑", "实验楼 501 遗留银色笔记本电脑"),
    ("宿舍钥匙串", "操场边捡到一串钥匙"),
    ("紫色运动手环", "游泳馆拾到紫色手环"),
    ("《线性代数》教材与笔记本", "B 区 305 教室发现教材和笔记本"),
]


def _find(items: list[dict], keyword: str) -> dict | None:
    for item in items:
        if keyword in item["title"]:
            return item
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="失物招领匹配质量评测")
    parser.add_argument("--details", action="store_true", help="打印每一对配对的明细")
    parser.add_argument("--verbose", action="store_true", help="压力测试中额外打印图像子通道分解")
    parser.add_argument("--db", default=os.path.join(BASE_DIR, "data", "lostfound.db"))
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(f"✗ 找不到数据库 {args.db}，请先执行： python seed.py --reset")
        return 2

    conn = db.connect(args.db)
    rows = conn.execute(db.ITEM_SELECT + " ORDER BY i.id").fetchall()
    items = [db.row_to_item(r) for r in rows]
    images = db.fetch_images(conn, [it["id"] for it in items])
    for item in items:
        item["images"] = images.get(int(item["id"]), [])
    conn.close()

    lost_items = [it for it in items if it["type"] == "lost"]
    found_items = [it for it in items if it["type"] == "found"]
    corpus = build_corpus(items)

    # ---------------- 正样本 / 负样本打分 ---------------- #
    positives: list[tuple[dict, dict, dict]] = []
    for lost_kw, found_kw in PAIR_MARKERS:
        lost = _find(lost_items, lost_kw)
        found = _find(found_items, found_kw)
        if lost and found:
            positives.append((lost, found, matcher.score_pair(lost, found, corpus=corpus)))
    positive_ids = {(int(a["id"]), int(b["id"])) for a, b, _ in positives}

    negatives = []
    for a in lost_items:
        for b in found_items:
            if (int(a["id"]), int(b["id"])) in positive_ids:
                continue
            negatives.append((a, b, matcher.score_pair(a, b, corpus=corpus)))

    print("=" * 74)
    print("  失物招领匹配质量评测")
    print("=" * 74)
    print(f"  数据规模：失物 {len(lost_items)} 条 · 拾物 {len(found_items)} 条 · "
          f"共 {len(lost_items) * len(found_items)} 组配对")
    print(f"  标注正样本 {len(positives)} 组 · 负样本 {len(negatives)} 组")

    # ---------------- P@1 / Recall@3 ---------------- #
    top1 = top3 = 0
    ranks = []
    for lost, found, _ in positives:
        scored = sorted(
            ((b, matcher.score_pair(lost, b, corpus=corpus)["score"]) for b in found_items),
            key=lambda pair: -pair[1],
        )
        rank = next((i + 1 for i, (item, _s) in enumerate(scored) if int(item["id"]) == int(found["id"])), None)
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank is not None and rank <= 3:
            top3 += 1

    total = max(1, len(positives))
    print("-" * 74)
    print(f"  P@1        : {top1}/{len(positives)} = {top1 / total * 100:.1f}%   （正确对象排名第一）")
    print(f"  Recall@3   : {top3}/{len(positives)} = {top3 / total * 100:.1f}%   （正确对象进入前三）")
    print(f"  平均排名    : {sum(r for r in ranks if r) / total:.2f}")
    if args.details:
        for (lost, found, _), rank in zip(positives, ranks):
            print(f"     · [{lost['title'][:22]:<24}] → 排名 {rank}")

    # ---------------- 分数分布 ---------------- #
    pos_scores = sorted(sc["score"] for _a, _b, sc in positives)
    neg_scores = sorted(sc["score"] for _a, _b, sc in negatives)

    def quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    print("-" * 74)
    print("  分数分布")
    print(f"    正样本  min {pos_scores[0]:.3f} · 中位 {quantile(pos_scores, 0.5):.3f} · max {pos_scores[-1]:.3f}")
    print(f"    负样本  min {neg_scores[0]:.3f} · 中位 {quantile(neg_scores, 0.5):.3f} · max {neg_scores[-1]:.3f}")
    overlap = sum(1 for s in neg_scores if s >= pos_scores[0])
    print(f"    负样本中高于「最低正样本分({pos_scores[0]:.3f})」的数量：{overlap}/{len(neg_scores)}")

    # 列出得分最高的负样本，便于确认它们是否真的"不该匹配"
    top_negatives = sorted(negatives, key=lambda item: -item[2]["score"])[:3]
    if top_negatives:
        print("    得分最高的负样本（需人工确认是否确实无关）：")
        for left, right, scored in top_negatives:
            print(
                f"      {scored['score']:.3f}  [{left['title'][:20]}] ↔ [{right['title'][:20]}]  "
                f"图像 {scored['image_score']:.3f} · 文本 {scored['text_score']:.3f} · 地点 {scored['location_score']:.3f}"
            )

    # ---------------- 阈值扫描 ---------------- #
    print("-" * 74)
    print("  阈值扫描（正样本视为应命中）")
    print(f"    {'阈值':<8}{'精确率':<10}{'召回率':<10}{'F1':<8}{'命中数':<10}{'误报数'}")
    best = None
    for step in range(10, 86, 5):
        threshold = step / 100
        tp = sum(1 for s in pos_scores if s >= threshold)
        fp = sum(1 for s in neg_scores if s >= threshold)
        fn = len(pos_scores) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        marker = ""
        if best is None or f1 > best[1]:
            best = (threshold, f1, precision, recall)
            marker = "  ← 最佳 F1"
        print(f"    {threshold:<8.2f}{precision:<10.3f}{recall:<10.3f}{f1:<8.3f}{tp:<10}{fp}{marker}")

    if best:
        print(f"    建议工作点：阈值 {best[0]:.2f} → 精确率 {best[2]:.3f} / 召回率 {best[3]:.3f} / F1 {best[1]:.3f}")
    print(f"    系统当前展示阈值：疑似 {matcher.CANDIDATE_THRESHOLD} · 高可信 {matcher.STRONG_THRESHOLD}")

    # ---------------- 通道贡献 ---------------- #
    print("-" * 74)
    print("  通道区分能力（正样本 vs 负样本 均值）")
    for label, key in (("图像感知", "image_score"), ("文本标签", "text_score"), ("地点语义", "location_score")):
        p = [sc[key] for _a, _b, sc in positives]
        n = [sc[key] for _a, _b, sc in negatives]
        p_mean = sum(p) / max(1, len(p))
        n_mean = sum(n) / max(1, len(n))
        p_with = sum(1 for x in p if x > 0.05)
        print(f"    {label:<10} 正样本均值 {p_mean:.3f} · 负样本均值 {n_mean:.3f} · "
              f"区分度 {p_mean - n_mean:+.3f} · 正样本有信号 {p_with}/{len(p)}")

    # ---------------- 同图压力测试：同色不同物 ---------------- #
    print("-" * 74)
    print("  图像通道压力测试（同色系 / 同品类 但并非同一物品）")
    stress = [
        ("黑色双肩包", "图书馆捡到黑色背包", "同物品配对（基准）"),
        ("黑色双肩包", "B 区 305 教室发现教材和笔记本", "不同物品（文本无关）"),
        ("蓝色双肩包", "图书馆捡到黑色背包", "同为背包但颜色不同"),
        ("粉色自动折叠雨伞", "捡到一把粉色雨伞", "同物品配对（基准）"),
        ("红色雨伞", "捡到一把粉色雨伞", "同品类不同颜色"),
        ("黄色雨伞", "捡到一把粉色雨伞", "同品类不同颜色"),
        ("白色无线蓝牙耳机", "图书馆一楼捡到白色有线耳机", "同色同品类，无线 vs 有线"),
        ("墨绿色保温杯", "教学楼 B 区教室捡到纯黑色保温杯", "同品类不同颜色"),
    ]
    for left_kw, right_kw, note in stress:
        left = _find(items, left_kw)
        right = _find(items, right_kw)
        if not left or not right:
            continue
        img, detail = item_image_similarity(left.get("images") or [], right.get("images") or [])
        txt, tdetail = text_similarity(left, right, corpus)
        print(f"    {note:<28} 图像 {img:.3f} (汉明 {detail.get('hamming')}) · "
              f"文本 {txt:.3f} (TF-IDF {tdetail.get('tfidf'):.3f} 标签 {tdetail.get('tag'):.3f})")
        if args.verbose:
            labels = detail.get("dominant_labels") or ["", ""]
            print(
                f"        └ 形状(pHash) {detail.get('phash', 0):.3f} · 颜色 {detail.get('color', 0):.3f} "
                f"(前景 {detail.get('crop_color', 0):.3f}) · 整图形状 {detail.get('full_phash', 0):.3f} "
                f"(汉明 {detail.get('full_hamming', 0)}) · 前景主色 [{labels[0]} | {labels[1]}]"
            )

    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
