"""文本标签匹配：中文友好的分词、TF-IDF 余弦、标签重合、地点语义比对。

中文没有空格分词，且失物招领的文本极短（标题 10 余字、标签 3-6 个），
引入 jieba 会带来额外依赖与词典冷启动开销。因此这里采用**字符级 + 词级混合**方案：

* 中文：单字（unigram）与相邻双字（bigram）同时入袋
  - unigram 保证召回（"黑色耳机" 与 "黑色蓝牙耳机" 共享 黑/色/耳/机）
  - bigram 提供局部语序信息（"耳机" 比孤立的 耳、机 更有区分度）
* 英文/数字：整词入袋（"AirPods"、"2023"、"B203"）
* 停用词与单字噪声过滤，避免 "的"、"了"、"在" 等虚词造成虚假相似

在此基础上提供四路互补的文本相似度：

1. ``tfidf_cosine``  —— 全语料 IDF 加权的语义主通道（主权重）
2. ``tag_overlap``   —— 用户显式填写的标签集合重合（F1 / Dice）
3. ``token_overlap`` —— 细粒度 token 重合（对错别字、语序差异更宽容）
4. ``char_dice``     —— 字符二元组 Dice，用于标签两两之间的模糊等价判断

路线 1-3 加权融合为 ``text_similarity``；``location_similarity`` 单独计算，
作为融合层的「地点加成」。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------- #
# 词典与正则
# --------------------------------------------------------------------------- #

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "吗", "呢", "吧",
    "把", "被", "与", "及", "或", "等", "并", "而", "于", "由", "对", "从", "为", "以", "之", "其",
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "is", "are", "was", "were", "it",
    "this", "that", "with", "for", "by", "as", "be", "my", "me", "i", "we", "you", "he", "she", "they",
    "lost", "found", "item", "please", "thanks", "thank",
}

# 失物招领常见的地点/归属提示词，用于地点语义加成
LOCATION_HINTS = (
    "教学楼", "图书馆", "食堂", "宿舍", "操场", "体育馆", "实验楼", "实验", "机房", "自习室", "教室",
    "校车", "车站", "快递", "超市", "便利店", "门口", "广场", "报告厅", "活动中心", "游泳馆", "球场",
    "走廊", "电梯", "洗手间", "卫生间", "咖啡", "书店", "医院", "银行", "公交", "地铁", "天桥", "草坪",
    "a区", "b区", "c区", "d区", "虎溪", "沙坪坝", "老校区", "新校区", "cqu",
)

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[a-zA-Z][a-zA-Z0-9+#.\-]*|\d+")
_CJK_RUN_RE = re.compile(rf"[{_CJK}]+")
_HASH_DIM = 1 << 17  # 131072 维哈希空间，够用且内存友好

# 常见同义词归一化：把不同说法折叠到同一 token，提升文本召回
SYNONYMS = {
    "蓝牙耳机": "耳机", "无线耳机": "耳机", "earphone": "耳机", "earbuds": "耳机", "airpods": "耳机",
    "airpod": "耳机", "headphone": "耳机", "headset": "耳机",
    "手机": "手机", "iphone": "手机", "安卓手机": "手机", "smartphone": "手机", "phone": "手机",
    "笔记本": "电脑", "笔记本电脑": "电脑", "laptop": "电脑", "电脑": "电脑", "ipad": "平板",
    "平板": "平板", "tablet": "平板",
    "双肩包": "书包", "背包": "书包", "bookbag": "书包", "backpack": "书包", "书包": "书包",
    "一卡通": "校园卡", "学生卡": "校园卡", "饭卡": "校园卡", "校园卡": "校园卡",
    "身份证": "身份证", "idcard": "身份证", "id": "身份证",
    "保温杯": "水杯", "杯子": "水杯", "水瓶": "水杯", "水杯": "水杯", "cup": "水杯", "bottle": "水杯",
    "充电宝": "充电宝", "移动电源": "充电宝", "powerbank": "充电宝",
    "u盘": "u盘", "优盘": "u盘", "usb": "u盘", "flashdisk": "u盘",
    "雨伞": "雨伞", "伞": "雨伞", "umbrella": "雨伞",
    "眼镜": "眼镜", "近视镜": "眼镜", "glasses": "眼镜",
    "钥匙": "钥匙", "车钥匙": "钥匙", "key": "钥匙", "keys": "钥匙",
    "钱包": "钱包", "卡包": "钱包", "wallet": "钱包",
    "手表": "手表", "watch": "手表", "手环": "手环", "bracelet": "手环",
    "围巾": "围巾", "scarf": "围巾", "帽子": "帽子", "hat": "帽子", "cap": "帽子",
    "外套": "外套", "夹克": "外套", "jacket": "外套", "coat": "外套", "羽绒服": "外套",
    "银行卡": "银行卡", "bankcard": "银行卡",
    "钥匙扣": "钥匙扣", "keychain": "钥匙扣",
    "黑": "黑色", "黑色": "黑色", "black": "黑色",
    "白": "白色", "白色": "白色", "white": "白色",
    "灰": "灰色", "灰色": "灰色", "gray": "灰色", "grey": "灰色",
    "红": "红色", "红色": "红色", "red": "红色",
    "蓝": "蓝色", "蓝色": "蓝色", "blue": "蓝色",
    "绿": "绿色", "绿色": "绿色", "green": "绿色",
    "黄": "黄色", "黄色": "黄色", "yellow": "黄色",
    "粉": "粉色", "粉色": "粉色", "pink": "粉色",
    "紫": "紫色", "紫色": "紫色", "purple": "紫色",
    "棕": "棕色", "棕色": "棕色", "咖啡色": "棕色", "brown": "棕色",
    "橙": "橙色", "橙色": "橙色", "orange": "橙色",
    "青": "青色", "青色": "青色", "cyan": "青色",
    "米色": "米色", "beige": "米色",
}

# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """全角转半角、去多余空白、统一小写（仅英文）。"""
    if not text:
        return ""
    chars = []
    for ch in str(text):
        code = ord(ch)
        if code == 0x3000:
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        chars.append(ch)
    out = "".join(chars)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def tokenize(text: str, *, use_synonyms: bool = True) -> list[str]:
    """把任意中英混排文本切成 token 列表（含中文 bigram）。"""
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalized):
        if _CJK_RUN_RE.fullmatch(raw):
            run = raw
            for ch in run:
                if ch not in STOPWORDS:
                    tokens.append(ch)
            for i in range(len(run) - 1):
                bigram = run[i : i + 2]
                if bigram not in STOPWORDS:
                    tokens.append(bigram)
        else:
            word = raw.lower().strip(".-")
            if not word or word in STOPWORDS:
                continue
            # 过滤噪声：单个拉丁字母（"a"/"x"）无信息量；单个数字保留（型号、房间号）
            if len(word) == 1 and word.isalpha() and word != "u":
                continue
            tokens.append(word)
    if use_synonyms:
        extra = [SYNONYMS[t] for t in tokens if t in SYNONYMS]
        tokens.extend(extra)
    return tokens


def normalize_tag(tag: str) -> str:
    """标签归一化：去空白、小写、同义词折叠。"""
    value = normalize_text(tag).strip().strip("#,，、;；")
    if not value:
        return ""
    lowered = value.lower()
    return SYNONYMS.get(lowered, SYNONYMS.get(value, value))


def normalize_tags(tags: Iterable[str] | str | None, *, limit: int = 24) -> list[str]:
    """清洗标签列表：支持字符串(逗号/空格分隔)或列表，去重保序。"""
    if tags is None:
        return []
    if isinstance(tags, str):
        raw_items = re.split(r"[,，、;；|\s]+", tags)
    else:
        raw_items = [str(t) for t in tags]
    out: list[str] = []
    for item in raw_items:
        value = normalize_tag(item)
        if value and value not in out and len(value) <= 24:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def item_text(item: dict, *, include_location: bool = True) -> str:
    """把一条物品记录拼成用于匹配的文本（标题权重最高，故重复一次）。"""
    parts = [
        str(item.get("title", "")),
        str(item.get("title", "")),  # 标题双权重
        str(item.get("description", "")),
        " ".join(item.get("tags") or []),
        str(item.get("color_name", "") or ""),
        str(item.get("category_name", "") or ""),
    ]
    if include_location:
        parts.append(str(item.get("location", "") or ""))
    return normalize_text(" ".join(p for p in parts if p))


# --------------------------------------------------------------------------- #
# TF-IDF（哈希技巧 + Cosine）
# --------------------------------------------------------------------------- #


def hash_token(token: str) -> int:
    """稳定哈希（不用内置 hash()，避免 PYTHONHASHSEED 随机化）。"""
    h = 2166136261
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h % _HASH_DIM


class TfidfCorpus:
    """小型语料上的 TF-IDF 向量化器。

    物品数量在千级以内，每次匹配直接对全量物品重新拟合 IDF 即可（毫秒级），
    无需持久化模型，避免「模型与数据不一致」的常见坑。
    """

    def __init__(self, documents: Optional[dict[int, str]] = None):
        self.df: Counter = Counter()
        self.total_docs = 0
        self.vectors: dict[int, dict[int, float]] = {}
        self.norms: dict[int, float] = {}
        if documents:
            self.fit(documents)

    def fit(self, documents: dict[int, str]) -> "TfidfCorpus":
        self.df = Counter()
        tokenized: dict[int, Counter] = {}
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            counts = Counter(tokens)
            tokenized[doc_id] = counts
            for token in counts:
                self.df[hash_token(token)] += 1
        self.total_docs = max(1, len(documents))
        self.vectors, self.norms = {}, {}
        for doc_id, counts in tokenized.items():
            self.vectors[doc_id] = self._vectorize(counts)
            self.norms[doc_id] = _norm(self.vectors[doc_id])
        return self

    def idf(self, hashed: int) -> float:
        # 平滑 IDF，+1 保证语料中未出现的词也有正权重
        return math.log((1 + self.total_docs) / (1 + self.df.get(hashed, 0))) + 1.0

    def _vectorize(self, counts: Counter) -> dict[int, float]:
        if not counts:
            return {}
        max_tf = max(counts.values()) or 1
        total = sum(counts.values()) or 1
        vector: dict[int, float] = {}
        for token, count in counts.items():
            tf = 0.5 + 0.5 * (count / max_tf)          # 增强型 TF 抑制长文本偏置
            weight = tf * self.idf(hash_token(token))
            vector[hash_token(token)] = vector.get(hash_token(token), 0.0) + weight
        # 长度归一化（余弦范数在外部统一处理，这里仅按长度做轻微缩放）
        scale = 1.0 / (1.0 + math.log(1 + total / max(1, len(counts))))
        return {k: v * scale for k, v in vector.items()}

    def vector_of(self, text: str) -> tuple[dict[int, float], float]:
        vector = self._vectorize(Counter(tokenize(text)))
        return vector, _norm(vector)


def _norm(vector: dict[int, float]) -> float:
    return math.sqrt(sum(v * v for v in vector.values())) or 1.0


def cosine(vec_a: dict[int, float], norm_a: float, vec_b: dict[int, float], norm_b: float) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(weight * vec_b.get(key, 0.0) for key, weight in vec_a.items())
    if dot <= 0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (norm_a * norm_b))))


def tfidf_similarity(corpus: TfidfCorpus, id_a: int, id_b: int) -> float:
    va, vb = corpus.vectors.get(id_a), corpus.vectors.get(id_b)
    if not va or not vb:
        return 0.0
    return cosine(va, corpus.norms.get(id_a, 1.0), vb, corpus.norms.get(id_b, 1.0))


# --------------------------------------------------------------------------- #
# 标签与 token 重合
# --------------------------------------------------------------------------- #


def dice_similarity(a: str, b: str) -> float:
    """字符二元组 Dice 系数，用于判断两个标签是否「几乎一样」。"""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else (0.5 if a in b or b in a else 0.0)
    grams_a = {a[i : i + 2] for i in range(len(a) - 1)}
    grams_b = {b[i : i + 2] for i in range(len(b) - 1)}
    if not grams_a or not grams_b:
        return 0.0
    inter = len(grams_a & grams_b)
    return float(2 * inter / (len(grams_a) + len(grams_b)))


def tag_overlap(tags_a: Iterable[str], tags_b: Iterable[str], *, fuzzy: float = 0.6) -> tuple[float, list[str]]:
    """标签集合的 Dice 重合度，支持模糊等价（"黑色耳机" ≈ "耳机"）。

    返回 ``(score, 命中标签对)``。
    """
    set_a = [t for t in (normalize_tag(t) for t in tags_a or []) if t]
    set_b = [t for t in (normalize_tag(t) for t in tags_b or []) if t]
    if not set_a or not set_b:
        return 0.0, []
    hits: list[str] = []
    used_b: set[int] = set()
    for ta in set_a:
        best_idx, best_score = -1, 0.0
        for idx, tb in enumerate(set_b):
            if idx in used_b:
                continue
            score = dice_similarity(ta, tb)
            if ta in tb or tb in ta:
                score = max(score, 0.8)
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx >= 0 and best_score >= fuzzy:
            used_b.add(best_idx)
            hits.append(set_b[best_idx] if set_b[best_idx] == ta else f"{ta}≈{set_b[best_idx]}")
    precision = len(hits) / len(set_a)
    recall = len(hits) / len(set_b)
    score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(score), hits


def token_overlap(text_a: str, text_b: str) -> float:
    """token 层面的加权重合度（bigram 权重高于 unigram）。"""
    def bag(text: str) -> Counter:
        counts: Counter = Counter()
        for token in tokenize(text):
            counts[token] += 2.0 if len(token) >= 2 else 1.0
        return counts

    bag_a, bag_b = bag(text_a), bag(text_b)
    if not bag_a or not bag_b:
        return 0.0
    keys = set(bag_a) | set(bag_b)
    inter = sum(min(bag_a.get(k, 0.0), bag_b.get(k, 0.0)) for k in keys)
    union = sum(max(bag_a.get(k, 0.0), bag_b.get(k, 0.0)) for k in keys)
    return float(inter / union) if union else 0.0


# --------------------------------------------------------------------------- #
# 地点语义
# --------------------------------------------------------------------------- #


def location_similarity(loc_a: str, loc_b: str) -> float:
    """地点字符串相似度：包含关系 > 提示词命中 > token 重合。"""
    a = normalize_text(loc_a).lower()
    b = normalize_text(loc_b).lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    hints_a = {h for h in LOCATION_HINTS if h in a}
    hints_b = {h for h in LOCATION_HINTS if h in b}
    if hints_a and hints_b:
        if hints_a & hints_b:
            return 0.9
        return 0.0
    return min(0.6, token_overlap(a, b) * 1.4)


# --------------------------------------------------------------------------- #
# 对外主入口
# --------------------------------------------------------------------------- #

TEXT_WEIGHTS = {"tfidf": 0.50, "tag": 0.32, "token": 0.18}


def build_corpus(items: Iterable[dict]) -> TfidfCorpus:
    documents = {int(it["id"]): item_text(it) for it in items}
    return TfidfCorpus(documents)


def text_similarity(
    item_a: dict,
    item_b: dict,
    corpus: Optional[TfidfCorpus] = None,
) -> tuple[float, dict]:
    """综合文本标签相似度，返回 ``(score, detail)``。"""
    text_a = item_text(item_a)
    text_b = item_text(item_b)

    if corpus is not None and int(item_a["id"]) in corpus.vectors and int(item_b["id"]) in corpus.vectors:
        tfidf = tfidf_similarity(corpus, int(item_a["id"]), int(item_b["id"]))
    else:
        single = TfidfCorpus({int(item_a["id"]): text_a, int(item_b["id"]): text_b})
        tfidf = tfidf_similarity(single, int(item_a["id"]), int(item_b["id"]))

    tag_score, tag_hits = tag_overlap(item_a.get("tags") or [], item_b.get("tags") or [])
    tok_score = token_overlap(text_a, text_b)
    color_bonus = 0.0
    ca, cb = (item_a.get("color_name") or "").strip(), (item_b.get("color_name") or "").strip()
    if ca and cb:
        color_bonus = 1.0 if normalize_tag(ca) == normalize_tag(cb) else dice_similarity(normalize_tag(ca), normalize_tag(cb)) * 0.5

    raw = (
        TEXT_WEIGHTS["tfidf"] * tfidf
        + TEXT_WEIGHTS["tag"] * tag_score
        + TEXT_WEIGHTS["token"] * tok_score
    )
    # 颜色文字描述一致时给予小幅加成（最多 +8%），不改变主排序趋势
    score = min(1.0, raw * (1.0 + 0.08 * color_bonus))

    return float(score), {
        "tfidf": round(tfidf, 4),
        "tag": round(tag_score, 4),
        "token": round(tok_score, 4),
        "tag_hits": tag_hits[:8],
        "color_text": round(color_bonus, 4),
    }
