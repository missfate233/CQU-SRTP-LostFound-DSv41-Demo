"""混合匹配引擎：图像感知得分 × 文本标签得分 的融合、排序、解释与持久化。

融合策略
--------
``score = w_img · image_similarity + w_txt · text_similarity + w_loc · location_similarity``

* 双侧都有图片时默认 ``w_img=0.45, w_txt=0.50, w_loc=0.05``
  —— 图像与文本大致等权，符合「拍照 + 文字描述」的真实使用习惯；
* 任一侧缺图时自动把图像权重按比例转移给文本，保证缺图记录仍能被召回；
* 地点语义只作为小幅加成，防止「同一栋楼里丢的两个黑色耳机」被误判为同一物品。

除总分外，引擎还必须给出**可解释证据**（reasons）：失物招领是高敏感场景，
只给一个黑盒百分比无法让用户信服；因此每条匹配都输出人类可读的理由，
例如「感知哈希距离 6/64（结构高度一致）」「标签命中：黑色耳机、白色」。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Optional

from . import db
from .image_features import item_image_similarity
from .image_features import PHASH_BITS
from .text_features import build_corpus, location_similarity, text_similarity

DEFAULT_WEIGHTS = {"image": 0.45, "text": 0.50, "location": 0.05}

# 判定阈值。标定依据：``eval_match.py`` 在 9 组标注正样本 × 135 组负样本上的扫描
#   * 0.55 —— 正样本最低分 0.557，负样本最高分 0.575（"白色无线耳机 vs 白色有线耳机"
#     这类确实很像是同一件东西），此点精确率 0.900 / 召回率 1.000；
#   * 0.42 —— 正样本全部命中，同时把明显无关的组合压到后面；
#   * 0.30 —— 展示下限，保证「疑似」候选不会被漏掉，交由人工核对。
# 注意：匹配排序才是核心指标（P@1 = 100%），阈值只决定"标记为多可信"。
STRONG_THRESHOLD = 0.55
CANDIDATE_THRESHOLD = 0.30
WEAK_THRESHOLD = 0.18


def normalize_weights(weights: Optional[dict] = None) -> dict:
    """校验并归一化权重；若一侧无图，会把图像权重让渡给文本。"""
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        for key in ("image", "text", "location"):
            if weights.get(key) is not None:
                try:
                    merged[key] = max(0.0, float(weights[key]))
                except (TypeError, ValueError):
                    continue
    total = sum(merged.values())
    if total <= 0:
        merged = dict(DEFAULT_WEIGHTS)
        total = sum(merged.values())
    return {k: v / total for k, v in merged.items()}


def rebalance_for_missing_image(weights: dict, has_image: bool) -> dict:
    """缺少图像证据时，把图像权重按比例转移给文本与地点。"""
    if has_image:
        return weights
    w = dict(weights)
    freed = w.pop("image", 0.0)
    rest = w.get("text", 0.0) + w.get("location", 0.0)
    if rest <= 0:
        return {"image": 0.0, "text": 1.0, "location": 0.0}
    w["text"] = w.get("text", 0.0) + freed * (w.get("text", 0.0) / rest)
    w["location"] = w.get("location", 0.0) + freed * (w.get("location", 0.0) / rest)
    w["image"] = 0.0
    return w


# --------------------------------------------------------------------------- #
# 证据文案
# --------------------------------------------------------------------------- #


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def build_reasons(image_score: float, image_detail: dict, text_score: float, text_detail: dict, loc_score: float) -> list[str]:
    """把各通道得分翻译成人类可读的匹配理由。"""
    reasons: list[str] = []

    if image_score > 0 and image_detail:
        shape = float(image_detail.get("shape", 0.0))
        color = float(image_detail.get("color", 0.0))
        if shape >= 0.85:
            reasons.append(f"物品轮廓高度吻合（轮廓 Dice {_pct(shape)}）")
        elif shape >= 0.7:
            reasons.append(f"物品轮廓较接近（轮廓 Dice {_pct(shape)}）")
        elif shape > 0 and shape < 0.5:
            reasons.append(f"物品轮廓差异较大（轮廓 Dice {_pct(shape)}）")
        if color >= 0.85:
            reasons.append(f"物品颜色分布几乎相同（直方图相交 {_pct(color)}）")
        elif color >= 0.7:
            reasons.append(f"物品颜色分布接近（直方图相交 {_pct(color)}）")
        elif color > 0 and color < 0.5:
            reasons.append(f"物品颜色分布差异较大（直方图相交 {_pct(color)}）")
        labels = image_detail.get("dominant_labels") or []
        if len(labels) == 2 and labels[0] and labels[1]:
            if labels[0] == labels[1]:
                reasons.append(f"两侧图像主色均为「{labels[0]}」")
            else:
                reasons.append(f"主色对比：{labels[0]} / {labels[1]}")
        dist = int(image_detail.get("hamming", PHASH_BITS))
        if float(image_detail.get("phash", 0.0)) >= 0.9:
            reasons.append(f"整图感知哈希几乎一致（距离 {dist}/64），疑似同一张照片")

    if text_score > 0 and text_detail:
        hits = text_detail.get("tag_hits") or []
        if hits:
            reasons.append("标签命中：" + "、".join(hits[:5]))
        tfidf = float(text_detail.get("tfidf", 0.0))
        if tfidf >= 0.45:
            reasons.append(f"文字描述语义高度相似（TF-IDF 余弦 {tfidf:.2f}）")
        elif tfidf >= 0.2:
            reasons.append(f"文字描述存在共同关键词（TF-IDF 余弦 {tfidf:.2f}）")
        if float(text_detail.get("color_text", 0.0)) > 0 and not hits:
            reasons.append("文字描述中的颜色一致")

    if loc_score >= 0.85:
        reasons.append("遗失/拾取地点吻合")
    elif loc_score >= 0.5:
        reasons.append("遗失/拾取地点相近")

    if not reasons:
        reasons.append("综合特征弱相关，建议人工核对")
    return reasons


def match_level(score: float) -> tuple[str, str]:
    """返回 (等级标识, 中文标签)。"""
    if score >= STRONG_THRESHOLD:
        return "strong", "高可信匹配"
    if score >= 0.42:
        return "likely", "较可能匹配"
    if score >= CANDIDATE_THRESHOLD:
        return "candidate", "疑似匹配"
    return "weak", "弱相关"


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #


def score_pair(
    item_a: dict,
    item_b: dict,
    *,
    corpus=None,
    weights: Optional[dict] = None,
) -> dict:
    """对一条「失物 x 拾物」组合打分（不依赖数据库，便于单测与调参）。"""
    base_weights = normalize_weights(weights)
    images_a = item_a.get("images") or []
    images_b = item_b.get("images") or []
    has_image = bool(images_a and images_b)

    img_score, img_detail = item_image_similarity(images_a, images_b)
    txt_score, txt_detail = text_similarity(item_a, item_b, corpus)
    loc_score = location_similarity(str(item_a.get("location") or ""), str(item_b.get("location") or ""))

    w = rebalance_for_missing_image(base_weights, has_image)
    score = (
        w["image"] * img_score
        + w["text"] * txt_score
        + w["location"] * loc_score
    )
    score = float(max(0.0, min(1.0, score)))
    level, level_label = match_level(score)

    detail = {
        "weights": {k: round(v, 4) for k, v in w.items()},
        "image": img_detail,
        "text": txt_detail,
        "location": round(loc_score, 4),
        "level": level,
        "level_label": level_label,
        "breakdown": {
            "image": round(img_score, 4),
            "text": round(txt_score, 4),
            "location": round(loc_score, 4),
        },
    }
    return {
        "score": round(score, 4),
        "image_score": round(float(img_score), 4),
        "text_score": round(float(txt_score), 4),
        "location_score": round(float(loc_score), 4),
        "reasons": build_reasons(img_score, img_detail, txt_score, txt_detail, loc_score),
        "detail": detail,
        "level": level,
        "level_label": level_label,
    }


# --------------------------------------------------------------------------- #
# 数据库级批量匹配
# --------------------------------------------------------------------------- #


def _load_all_items(conn: sqlite3.Connection, *, exclude_ids: Iterable[int] = ()) -> list[dict]:
    rows = conn.execute(db.ITEM_SELECT + " ORDER BY i.id").fetchall()
    items = [db.row_to_item(r) for r in rows]
    images = db.fetch_images(conn, [it["id"] for it in items])
    skip = {int(i) for i in exclude_ids}
    out = []
    for it in items:
        if int(it["id"]) in skip:
            continue
        it["images"] = images.get(int(it["id"]), [])
        out.append(it)
    return out


def find_matches_for(
    conn: sqlite3.Connection,
    item_id: int,
    *,
    top_k: int = 8,
    threshold: float = WEAK_THRESHOLD,
    weights: Optional[dict] = None,
    persist: bool = True,
) -> list[dict]:
    """为指定物品找出最匹配的对手方记录（失物→拾物，拾物→失物）。"""
    target = db.get_item(conn, int(item_id))
    if not target:
        return []
    target_type = target.get("type")
    counterpart = "found" if target_type == "lost" else "lost"

    candidates = [it for it in _load_all_items(conn) if it.get("type") == counterpart]
    if not candidates:
        return []

    # IDF 语料：把目标与全部候选放在同一语料里，IDF 才反映真实稀有度
    corpus = build_corpus([target, *candidates])

    results: list[dict] = []
    for cand in candidates:
        scored = score_pair(target, cand, corpus=corpus, weights=weights)
        if scored["score"] < threshold:
            continue
        results.append({"item": cand, **scored})

    results.sort(key=lambda r: (-r["score"], -r["image_score"], r["item"]["id"]))
    results = results[: max(1, int(top_k))]

    if persist:
        save_matches(conn, target, results)
    return results


def save_matches(conn: sqlite3.Connection, target: dict, results: list[dict]) -> None:
    """把匹配结果写入 matches 表（失物 id 恒小于拾物 id 由写入规则保证）。"""
    now = db.now_iso()
    rows = []
    for res in results:
        other = res["item"]
        lost_id, found_id = (target["id"], other["id"]) if target["type"] == "lost" else (other["id"], target["id"])
        rows.append(
            (
                int(lost_id),
                int(found_id),
                res["score"],
                res["image_score"],
                res["text_score"],
                json.dumps(res["reasons"], ensure_ascii=False),
                json.dumps(res["detail"], ensure_ascii=False),
                now,
            )
        )
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO matches(lost_id, found_id, score, image_score, text_score, reasons, detail, computed_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(lost_id, found_id) DO UPDATE SET
            score = excluded.score,
            image_score = excluded.image_score,
            text_score = excluded.text_score,
            reasons = excluded.reasons,
            detail = excluded.detail,
            computed_at = excluded.computed_at
        """,
        rows,
    )
    top = max((r[2] for r in rows), default=0.0)
    conn.execute("UPDATE items SET match_top = MAX(match_top, ?) WHERE id = ?", (top, int(target["id"])))
    conn.commit()


def recompute_all(conn: sqlite3.Connection, *, weights: Optional[dict] = None, clear: bool = True) -> dict:
    """全量重算：所有 失物 × 拾物 组合。

    返回统计信息（组合数、命中数、耗时），供管理端「重新计算」按钮展示。
    """
    import time

    started = time.perf_counter()
    if clear:
        conn.execute("DELETE FROM matches")
        conn.execute("UPDATE items SET match_top = 0")

    items = _load_all_items(conn)
    lost = [it for it in items if it.get("type") == "lost"]
    found = [it for it in items if it.get("type") == "found"]
    corpus = build_corpus(items)

    hits = 0
    top_by_item: dict[int, float] = {}
    batch: list[tuple] = []
    now = db.now_iso()

    for a in lost:
        for b in found:
            scored = score_pair(a, b, corpus=corpus, weights=weights)
            if scored["score"] < WEAK_THRESHOLD:
                continue
            hits += 1
            batch.append(
                (
                    int(a["id"]),
                    int(b["id"]),
                    scored["score"],
                    scored["image_score"],
                    scored["text_score"],
                    json.dumps(scored["reasons"], ensure_ascii=False),
                    json.dumps(scored["detail"], ensure_ascii=False),
                    now,
                )
            )
            for side in (a["id"], b["id"]):
                top_by_item[int(side)] = max(top_by_item.get(int(side), 0.0), scored["score"])

    if batch:
        conn.executemany(
            """
            INSERT INTO matches(lost_id, found_id, score, image_score, text_score, reasons, detail, computed_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(lost_id, found_id) DO UPDATE SET
                score = excluded.score,
                image_score = excluded.image_score,
                text_score = excluded.text_score,
                reasons = excluded.reasons,
                detail = excluded.detail,
                computed_at = excluded.computed_at
            """,
            batch,
        )
    conn.executemany(
        "UPDATE items SET match_top = ? WHERE id = ?",
        [(round(v, 4), k) for k, v in top_by_item.items()],
    )
    conn.commit()
    elapsed = time.perf_counter() - started
    return {
        "combinations": len(lost) * len(found),
        "matches": hits,
        "strong": sum(1 for r in batch if r[2] >= STRONG_THRESHOLD),
        "seconds": round(elapsed, 3),
        "pairs_per_second": int(len(lost) * len(found) / elapsed) if elapsed > 0 else 0,
    }


def load_cached_matches(conn: sqlite3.Connection, item_id: int, *, top_k: int = 8) -> list[dict]:
    """读取 matches 表中的缓存结果（不重算），用于列表/详情快速展示。"""
    rows = conn.execute(
        """
        SELECT m.score, m.image_score, m.text_score, m.reasons, m.detail, m.computed_at,
               CASE WHEN m.lost_id = ? THEN m.found_id ELSE m.lost_id END AS other_id
        FROM matches m
        WHERE m.lost_id = ? OR m.found_id = ?
        ORDER BY m.score DESC
        LIMIT ?
        """,
        (int(item_id), int(item_id), int(item_id), max(1, int(top_k))),
    ).fetchall()
    if not rows:
        return []
    other_ids = [int(r["other_id"]) for r in rows]
    marks = ",".join("?" * len(other_ids))
    item_rows = conn.execute(db.ITEM_SELECT + f" WHERE i.id IN ({marks})", other_ids).fetchall()
    images = db.fetch_images(conn, other_ids)
    by_id = {}
    for row in item_rows:
        item = db.row_to_item(row)
        item["images"] = images.get(int(item["id"]), [])
        by_id[int(item["id"])] = item

    out = []
    for row in rows:
        other = by_id.get(int(row["other_id"]))
        if not other:
            continue
        out.append(
            {
                "item": other,
                "score": float(row["score"]),
                "image_score": float(row["image_score"]),
                "text_score": float(row["text_score"]),
                "reasons": _json_list(row["reasons"]),
                "detail": _json_dict(row["detail"]),
                "computed_at": row["computed_at"],
            }
        )
    return out


def build_corpus_for(target: dict, others: Iterable[dict]) -> "object":
    """把「目标物品 + 候选集」放进同一语料拟合 IDF，保证 IDF 反映真实稀有度。

    单独给目标物品拟合 IDF 会退化（语料只有 1 篇时所有词权重相同），
    匹配质量明显下降，因此评测/试算接口也必须走全语料。
    """
    return build_corpus([target, *others])


def _json_list(raw: Any) -> list:
    try:
        value = json.loads(raw) if raw else []
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _json_dict(raw: Any) -> dict:
    try:
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


_ITEM_COLUMNS_CACHE: Optional[set[str]] = None


def db_item_columns() -> set[str]:
    global _ITEM_COLUMNS_CACHE
    if _ITEM_COLUMNS_CACHE is None:
        _ITEM_COLUMNS_CACHE = {
            "id", "type", "title", "category_id", "description", "tags", "color_name", "location",
            "happened_at", "contact_name", "contact", "status", "user_id", "view_count",
            "match_top", "created_at", "updated_at", "category_name", "category_code",
        }
    return _ITEM_COLUMNS_CACHE
