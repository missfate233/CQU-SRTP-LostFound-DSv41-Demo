"""SQLite 数据访问层：建表、连接管理、通用查询与序列化。

设计要点
--------
* 只使用标准库 ``sqlite3``，无 ORM，便于评审与部署；
* 开启 WAL + 外键约束，配合 Flask 的 ``g`` 做「每请求一连接」；
* ``items`` 同时存储「失物(lost)」与「拾物(found)」记录，通过 ``type`` 区分，
  便于统一做跨类型匹配；
* ``matches`` 表缓存撮合结果，重复查询时可直接命中缓存（见 ``matcher.py``）。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

ITEM_TYPES = ("lost", "found")
ITEM_TYPE_LABEL = {"lost": "寻物启事", "found": "失物招领"}
STATUS_LABEL = {
    "open": "进行中",
    "matched": "已匹配",
    "closed": "已完成",
    "archived": "已下架",
}
STATUS_CHOICES = tuple(STATUS_LABEL.keys())

DEFAULT_CATEGORIES = [
    ("card", "证件卡片", "一卡通 / 身份证 / 学生证 / 银行卡 / 钥匙卡"),
    ("digital", "数码电子", "手机 / 笔记本 / 平板 / 耳机 / 充电宝 / U盘"),
    ("bag", "箱包书包", "书包 / 手提包 / 行李箱 / 钱包 / 卡包"),
    ("clothing", "衣物鞋帽", "外套 / 围巾 / 帽子 / 手套 / 运动鞋"),
    ("book", "书籍资料", "教材 / 笔记本 / 实验报告 / 打印讲义"),
    ("sports", "运动器材", "球拍 / 水杯 / 护具 / 头盔 / 球类"),
    ("glasses", "眼镜饰品", "近视镜 / 墨镜 / 手表 / 首饰 / 发卡"),
    ("key", "钥匙门禁", "宿舍钥匙 / 车钥匙 / 门禁卡 / 钥匙扣"),
    ("umbrella", "雨具水杯", "雨伞 / 保温杯 / 水杯 / 雨衣"),
    ("other", "其他物品", "以上分类未覆盖的物品"),
]

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    display_name  TEXT    NOT NULL DEFAULT '',
    contact       TEXT    NOT NULL DEFAULT '',
    role          TEXT    NOT NULL DEFAULT 'user',   -- user | admin
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    code       TEXT    NOT NULL UNIQUE,
    name       TEXT    NOT NULL,
    hint       TEXT    NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    type           TEXT    NOT NULL CHECK (type IN ('lost', 'found')),
    title          TEXT    NOT NULL,
    category_id    INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    description    TEXT    NOT NULL DEFAULT '',
    tags           TEXT    NOT NULL DEFAULT '[]',   -- JSON 数组：标签(文本标签匹配核心)
    color_name     TEXT    NOT NULL DEFAULT '',     -- 归一化主色描述：黑色 等
    location       TEXT    NOT NULL DEFAULT '',
    happened_at    TEXT    NOT NULL DEFAULT '',     -- ISO 字符串，可精确到分钟
    contact_name   TEXT    NOT NULL DEFAULT '',
    contact        TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL DEFAULT 'open',
    user_id        INTEGER REFERENCES users(id) ON DELETE SET NULL,
    view_count     INTEGER NOT NULL DEFAULT 0,
    match_top      REAL    NOT NULL DEFAULT 0,      -- 冗余：最佳匹配得分(列表页展示)
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_type     ON items(type, status, id DESC);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category_id);

CREATE TABLE IF NOT EXISTS item_images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    filename        TEXT    NOT NULL,
    thumb           TEXT    NOT NULL DEFAULT '',
    width           INTEGER NOT NULL DEFAULT 0,
    height          INTEGER NOT NULL DEFAULT 0,
    phash_hex       TEXT    NOT NULL DEFAULT '',   -- 64bit 感知哈希(16 位十六进制)
    phash_bits      BLOB,                          -- 同样的 64bit，按位存储便于计算
    color_hist      TEXT    NOT NULL DEFAULT '[]', -- 8x4x4 HSV 直方图(JSON，整图)
    dominant        TEXT    NOT NULL DEFAULT '{}', -- 主色信息(JSON，整图)
    -- 前景（物体）特征：图像匹配主通道，显著提升「同色不同物」的区分度
    crop_phash_hex  TEXT    NOT NULL DEFAULT '',
    crop_phash_bits BLOB,
    crop_hist       TEXT    NOT NULL DEFAULT '[]',
    crop_dominant   TEXT    NOT NULL DEFAULT '{}',
    shape_sig       BLOB,                          -- 尺度归一化轮廓指纹(256bit)
    shape_sig_hex   TEXT    NOT NULL DEFAULT '',
    blob_fill       REAL    NOT NULL DEFAULT 0,    -- 前景像素/包围盒面积(区分方块与圆)
    blob_ratio      REAL    NOT NULL DEFAULT 0,    -- 前景像素占比
    blob_box        TEXT    NOT NULL DEFAULT '[]', -- 物体包围盒 [x0,y0,x1,y1]
    sort_order      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_item ON item_images(item_id, sort_order);

-- 增量迁移：老库缺少前景特征列时自动补齐（幂等）

CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lost_id       INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    found_id      INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    score         REAL    NOT NULL DEFAULT 0,   -- 融合得分(受权重影响，展示用)
    image_score   REAL    NOT NULL DEFAULT 0,   -- 图像感知得分(权重无关)
    text_score    REAL    NOT NULL DEFAULT 0,   -- 文本标签得分(权重无关)
    reasons       TEXT    NOT NULL DEFAULT '[]',
    detail        TEXT    NOT NULL DEFAULT '{}',
    computed_at   TEXT    NOT NULL,
    UNIQUE (lost_id, found_id)
);
CREATE INDEX IF NOT EXISTS idx_matches_lost  ON matches(lost_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_matches_found ON matches(found_id, score DESC);
"""

# --------------------------------------------------------------------------- #
# 时间工具
# --------------------------------------------------------------------------- #


def now_iso() -> str:
    """返回本地时区的 ISO8601 字符串（秒级，便于展示与排序）。"""
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def humanize_time(value: str) -> str:
    """把时间戳转成「3 小时前」这类相对描述。"""
    dt = parse_iso(value)
    if not dt:
        return value or ""
    delta = datetime.now() - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return dt.strftime("%Y-%m-%d %H:%M")
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    if seconds < 86400:
        return f"{seconds // 3600} 小时前"
    if seconds < 86400 * 30:
        return f"{seconds // 86400} 天前"
    return dt.strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# 连接与初始化
# --------------------------------------------------------------------------- #


def connect(db_path: str) -> sqlite3.Connection:
    """创建一个带 Row 工厂的连接（调用方负责关闭）。"""
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


# 增量迁移表：{表名: [(列名, 列定义), ...]}
# 老数据库升级时自动 ALTER TABLE 补齐，避免用户手动删库重建。
MIGRATIONS = {
    "item_images": [
        ("crop_phash_hex", "TEXT NOT NULL DEFAULT ''"),
        ("crop_phash_bits", "BLOB"),
        ("crop_hist", "TEXT NOT NULL DEFAULT '[]'"),
        ("crop_dominant", "TEXT NOT NULL DEFAULT '{}'"),
        ("shape_sig", "BLOB"),
        ("shape_sig_hex", "TEXT NOT NULL DEFAULT ''"),
        ("blob_fill", "REAL NOT NULL DEFAULT 0"),
        ("blob_ratio", "REAL NOT NULL DEFAULT 0"),
        ("blob_box", "TEXT NOT NULL DEFAULT '[]'"),
    ],
    "items": [
        ("match_top", "REAL NOT NULL DEFAULT 0"),
        ("view_count", "INTEGER NOT NULL DEFAULT 0"),
    ],
}


def _apply_migrations(conn: sqlite3.Connection) -> list[str]:
    """对已存在的表补齐缺失列，返回实际执行的变更列表。"""
    applied: list[str] = []
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if not existing:
            continue  # 表尚不存在，CREATE TABLE 已建全
        for name, ddl in columns:
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            applied.append(f"{table}.{name}")
    return applied


def init_db(db_path: str) -> None:
    """建表 + 写入默认分类 + 增量迁移（幂等）。"""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        rows = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()
        if not rows or rows["n"] == 0:
            conn.executemany(
                "INSERT INTO categories(code, name, hint, sort_order) VALUES (?,?,?,?)",
                [(code, name, hint, idx) for idx, (code, name, hint) in enumerate(DEFAULT_CATEGORIES)],
            )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 序列化辅助
# --------------------------------------------------------------------------- #


def _loads(raw: Any, fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def row_to_image(row: sqlite3.Row | dict) -> dict:
    """把图片行转成可安全序列化的字典。

    **BLOB 列必须转成十六进制字符串**：``sqlite3`` 返回 ``bytes``，而 ``bytes``
    无法被 ``json.dumps`` 处理，直接塞进 ``jsonify`` 会抛
    ``TypeError: Object of type bytes is not JSON serializable``。
    下游（``image_features.coerce_bits``）本身同时接受 bytes 与十六进制字符串，
    因此统一在数据层出口做这个转换，是最省心的边界处理方式。
    """
    data = dict(row)
    hist = _loads(data.get("color_hist"), [])
    data["color_hist"] = hist if isinstance(hist, list) else []
    data["dominant"] = _loads(data.get("dominant"), {})
    crop_hist = _loads(data.get("crop_hist"), [])
    data["crop_hist"] = crop_hist if isinstance(crop_hist, list) else []
    data["crop_dominant"] = _loads(data.get("crop_dominant"), {})
    data["blob_box"] = _loads(data.get("blob_box"), [])

    raw_bits = data.pop("phash_bits", None)
    raw_crop_bits = data.pop("crop_phash_bits", None)
    raw_shape = data.pop("shape_sig", None)
    data["phash_bits_hex"] = bytes(raw_bits).hex() if raw_bits else data.get("phash_hex", "")
    data["crop_phash_bits_hex"] = bytes(raw_crop_bits).hex() if raw_crop_bits else data.get("crop_phash_hex", "")
    if raw_shape:
        data["shape_sig_hex"] = bytes(raw_shape).hex()
    return data


def row_to_item(row: sqlite3.Row | dict, images: Optional[Iterable] = None) -> dict:
    data = dict(row)
    tags = _loads(data.get("tags"), [])
    data["tags"] = [str(t).strip() for t in tags if str(t).strip()] if isinstance(tags, list) else []
    data["category_name"] = data.pop("category_name", None) or "未分类"
    data["category_code"] = data.pop("category_code", None) or ""
    data["type_label"] = ITEM_TYPE_LABEL.get(data.get("type", ""), data.get("type", ""))
    data["status_label"] = STATUS_LABEL.get(data.get("status", ""), data.get("status", ""))
    data["created_human"] = humanize_time(data.get("created_at", ""))
    data["happened_human"] = data.get("happened_at") or data.get("created_at", "")
    if images is not None:
        data["images"] = [row_to_image(img) for img in images]
    return data


ITEM_SELECT = """
SELECT i.*, c.name AS category_name, c.code AS category_code
FROM items i
LEFT JOIN categories c ON c.id = i.category_id
"""


def insert_image(
    conn: sqlite3.Connection,
    item_id: int,
    *,
    saved: dict,
    feats: dict,
    sort_order: int = 0,
) -> int:
    """写入一条图片记录（含整图与前景两套感知特征）。

    全项目只保留这一处 INSERT，避免字段列表在 app / seed / 迁移脚本之间漂移。
    """
    cur = conn.execute(
        """
        INSERT INTO item_images(item_id, filename, thumb, width, height,
                                phash_hex, phash_bits, color_hist, dominant,
                                crop_phash_hex, crop_phash_bits, crop_hist, crop_dominant,
                                shape_sig, shape_sig_hex, blob_fill, blob_ratio, blob_box,
                                sort_order, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(item_id),
            saved.get("filename", ""),
            saved.get("thumb", ""),
            int(feats.get("width") or 0),
            int(feats.get("height") or 0),
            feats.get("phash_hex", ""),
            bytes(feats["phash_bits"].tobytes()) if feats.get("phash_bits") is not None else None,
            json.dumps(feats.get("color_hist") or []),
            json.dumps(feats.get("dominant") or {}, ensure_ascii=False),
            feats.get("crop_phash_hex", ""),
            bytes(feats["crop_phash_bits"].tobytes()) if feats.get("crop_phash_bits") is not None else None,
            json.dumps(feats.get("crop_hist") or []),
            json.dumps(feats.get("crop_dominant") or {}, ensure_ascii=False),
            _signature_bytes(feats.get("shape_sig")),
            feats.get("shape_sig_hex", ""),
            float(feats.get("blob_fill") or 0.0),
            float(feats.get("blob_ratio") or 0.0),
            json.dumps(feats.get("blob_box") or []),
            int(sort_order),
            now_iso(),
        ),
    )
    return int(cur.lastrowid)


def _signature_bytes(signature: Any) -> Optional[bytes]:
    """把 0/1 轮廓指纹打包成 32 字节存储。"""
    if signature is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(signature).reshape(-1)
        if arr.size == 0:
            return None
        return np.packbits((arr != 0).astype(np.uint8)).tobytes()
    except Exception:  # noqa: BLE001 - 特征缺失不应阻断入库
        return None


def next_sort_order(conn: sqlite3.Connection, item_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(sort_order), -1) AS m FROM item_images WHERE item_id = ?", (int(item_id),)
    ).fetchone()
    return int(row["m"]) + 1 if row else 0


def fetch_images(conn: sqlite3.Connection, item_ids: Iterable[int]) -> dict[int, list[dict]]:
    """一次取出多个物品的图片，避免 N+1 查询。"""
    ids = [int(i) for i in item_ids]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT * FROM item_images WHERE item_id IN ({marks}) ORDER BY item_id, sort_order, id",
        ids,
    ).fetchall()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["item_id"]), []).append(row_to_image(row))
    return grouped


def get_item(conn: sqlite3.Connection, item_id: int) -> Optional[dict]:
    row = conn.execute(ITEM_SELECT + " WHERE i.id = ?", (item_id,)).fetchone()
    if not row:
        return None
    images = fetch_images(conn, [item_id]).get(int(item_id), [])
    return row_to_item(row, images)


def query_items(
    conn: sqlite3.Connection,
    *,
    item_type: str | None = None,
    category: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
    user_id: int | None = None,
    order: str = "newest",
    limit: int = 24,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """按条件分页查询物品，返回 (items, total)。"""
    where: list[str] = []
    params: list[Any] = []

    if item_type in ITEM_TYPES:
        where.append("i.type = ?")
        params.append(item_type)
    if category:
        where.append("c.code = ?")
        params.append(category)
    if status:
        where.append("i.status = ?")
        params.append(status)
    if user_id:
        where.append("i.user_id = ?")
        params.append(int(user_id))
    if keyword:
        like = f"%{keyword.strip()}%"
        where.append(
            "(i.title LIKE ? OR i.description LIKE ? OR i.tags LIKE ? OR i.location LIKE ? OR i.color_name LIKE ?)"
        )
        params.extend([like] * 5)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total = int(conn.execute(
        "SELECT COUNT(*) AS n FROM items i LEFT JOIN categories c ON c.id = i.category_id" + clause,
        params,
    ).fetchone()["n"])

    order_sql = {
        "newest": "i.created_at DESC, i.id DESC",
        "oldest": "i.created_at ASC, i.id ASC",
        "hot": "i.view_count DESC, i.id DESC",
        "match": "i.match_top DESC, i.id DESC",
    }.get(order, "i.created_at DESC, i.id DESC")

    rows = conn.execute(
        ITEM_SELECT + clause + f" ORDER BY {order_sql} LIMIT ? OFFSET ?",
        [*params, max(1, min(int(limit), 100)), max(0, int(offset))],
    ).fetchall()

    items = [row_to_item(r) for r in rows]
    images = fetch_images(conn, [it["id"] for it in items])
    for it in items:
        it["images"] = images.get(int(it["id"]), [])
    return items, total


def list_categories(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.*, (SELECT COUNT(*) FROM items i WHERE i.category_id = c.id) AS item_count
        FROM categories c ORDER BY c.sort_order, c.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM items WHERE type='lost')                       AS lost_count,
          (SELECT COUNT(*) FROM items WHERE type='found')                      AS found_count,
          (SELECT COUNT(*) FROM items WHERE status IN ('matched','closed'))     AS solved_count,
          (SELECT COUNT(*) FROM matches WHERE score >= 0.55)                    AS strong_match_count,
          (SELECT COUNT(*) FROM item_images)                                    AS image_count
        """
    ).fetchone()
    data = dict(row) if row else {}
    total = int(data.get("lost_count", 0)) + int(data.get("found_count", 0))
    data["total_count"] = total
    data["solve_rate"] = round(int(data.get("solved_count", 0)) / total * 100, 1) if total else 0.0
    return data


def find_user(conn: sqlite3.Connection, username: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def get_user(conn: sqlite3.Connection, user_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def upsert_category_counts(conn: sqlite3.Connection) -> None:
    """占位：分类计数由查询实时聚合，无需物化。"""
    return None
