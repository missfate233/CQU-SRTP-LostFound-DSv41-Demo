"""Flask 应用工厂 + REST API + 页面路由。

结构
----
* ``create_app()``  —— 应用工厂（便于测试与多实例部署）
* 页面路由         —— Jinja 模板渲染（首屏 SSR，利于分享与收录）
* ``/api/*``       —— 纯 JSON 接口，前端异步交互与管理端调用

鉴权：Flask 原生 session cookie（``werkzeug.security`` 做 PBKDF2 口令哈希）。
失物招领属于低敏感度场景，未登录用户可浏览、检索、查看匹配结果；
发布信息需要登录，避免刷单与虚假信息污染匹配结果。
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from . import __version__, db, image_features as imf, matcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(BASE_DIR, "data", "lostfound.db")
DEFAULT_UPLOADS = os.path.join(BASE_DIR, "static", "uploads")
MAX_IMAGES_PER_ITEM = 4
MAX_CONTENT_LENGTH = 24 * 1024 * 1024  # 单次请求上限（多图合计）


# --------------------------------------------------------------------------- #
# 连接管理
# --------------------------------------------------------------------------- #


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = db.connect(current_app.config["DATABASE"])
    return g.db


def close_db(_exc: Optional[BaseException] = None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


# --------------------------------------------------------------------------- #
# 鉴权
# --------------------------------------------------------------------------- #


def current_user() -> Optional[dict]:
    if "user" not in g:
        user_id = session.get("user_id")
        g.user = db.get_user(get_db(), int(user_id)) if user_id else None
    return g.user


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            if request.path.startswith("/api/"):
                return jsonify(ok=False, error="unauthorized", message="请先登录后再操作"), 401
            flash("请先登录", "warning")
            return redirect(url_for("login_page", next=request.path))
        return view(*args, **kwargs)

    return wrapper


def ensure_demo_user(conn: sqlite3.Connection) -> None:
    """保证存在演示账号，方便评审快速体验（口令均为 demo1234）。"""
    for username, display, contact, role in (
        ("demo", "演示用户", "demo@cqu.edu.cn", "user"),
        ("admin", "系统管理员", "admin@cqu.edu.cn", "admin"),
    ):
        if not db.find_user(conn, username):
            conn.execute(
                "INSERT INTO users(username, password_hash, display_name, contact, role, created_at) VALUES (?,?,?,?,?,?)",
                (username, generate_password_hash("demo1234"), display, contact, role, db.now_iso()),
            )
    conn.commit()


# --------------------------------------------------------------------------- #
# 请求/响应辅助
# --------------------------------------------------------------------------- #


def payload() -> dict:
    """同时兼容 JSON 与表单提交。"""
    if request.is_json:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}
    if request.form:
        return {k: v for k, v in request.form.items()}
    return {}


def json_safe(value: Any) -> Any:
    """递归地把不可 JSON 序列化的类型转换成安全形式。

    主要针对 ``sqlite3`` 返回的 ``bytes``（BLOB）。数据层出口已经做了转换，
    这里是**边界兜底**：任何未来新增的二进制字段都不会再让整个接口 500。
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def ok(**data):
    return jsonify(ok=True, **json_safe(data))


def fail(message: str, status: int = 400, code: str = "bad_request"):
    return jsonify(ok=False, error=code, message=message), status


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 物品写入
# --------------------------------------------------------------------------- #


def probe_image(feats: dict) -> dict:
    """把 ``extract_features`` 的输出裁剪成「仅感知特征」的图片字典。

    临时上传的照片不落盘：只为算分，避免演示/试算把磁盘塞满。
    两个试算接口（``/api/match/pair`` 与 ``/api/items/<id>/match``）共用本函数，
    防止特征字段列表在两处漂移。
    """
    return {
        "id": None,
        "phash_hex": feats.get("phash_hex", ""),
        "phash_bits": feats["phash_bits"].tobytes() if feats.get("phash_bits") is not None else None,
        "color_hist": feats.get("color_hist") or [],
        "dominant": feats.get("dominant") or {},
        "crop_phash_hex": feats.get("crop_phash_hex", ""),
        "crop_phash_bits": feats["crop_phash_bits"].tobytes() if feats.get("crop_phash_bits") is not None else None,
        "crop_hist": feats.get("crop_hist") or [],
        "crop_dominant": feats.get("crop_dominant") or {},
        "shape_sig_hex": feats.get("shape_sig_hex", ""),
        "blob_fill": feats.get("blob_fill", 0.0),
    }


def build_probe_images(raw_images: Any) -> list[dict]:
    """把 base64 dataURL 列表转成「仅含感知特征」的图片字典，供内存内比对使用。"""
    import base64
    import binascii

    out: list[dict] = []
    if not isinstance(raw_images, list):
        return out
    for entry in raw_images[:2]:
        if not isinstance(entry, str) or not entry.strip():
            continue
        text = entry.strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[-1]
        try:
            raw = base64.b64decode(text, validate=False)
        except (binascii.Error, ValueError):
            continue
        valid, _ = imf.validate_upload(raw)
        if not valid:
            continue
        try:
            out.append(probe_image(imf.extract_features(imf.load_image(raw, max_side=1024))))
        except Exception:  # noqa: BLE001
            continue
    return out


def build_probe_images_from_files(files: list) -> list[dict]:
    """把上传的文件对象转成感知特征字典（同样不落盘）。"""
    out: list[dict] = []
    for storage in files[:2]:
        if not storage or not storage.filename:
            continue
        raw = storage.read()
        valid, _ = imf.validate_upload(raw)
        if not valid:
            continue
        try:
            out.append(probe_image(imf.extract_features(imf.load_image(raw, max_side=1024))))
        except Exception:  # noqa: BLE001
            continue
    return out


def save_uploaded_images(conn: sqlite3.Connection, item_id: int, files: list) -> list[dict]:
    """保存上传图片：校验 → 提特征 → 落盘 → 入库。返回新建的图片记录。"""
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    created: list[dict] = []
    order = db.next_sort_order(conn, item_id)

    for storage in files[:MAX_IMAGES_PER_ITEM]:
        if not storage or not storage.filename:
            continue
        raw = storage.read()
        valid, reason = imf.validate_upload(raw)
        if not valid:
            current_app.logger.warning("跳过非法图片 %s: %s", storage.filename, reason)
            continue
        try:
            img = imf.load_image(raw, max_side=1600)
            feats = imf.extract_features(img)
            saved = imf.save_image_variants(img, upload_dir, storage.filename)
        except Exception as exc:  # noqa: BLE001
            current_app.logger.warning("图片处理失败 %s: %s", storage.filename, exc)
            continue

        image_id = db.insert_image(conn, item_id, saved=saved, feats=feats, sort_order=order)
        order += 1
        created.append({"id": image_id, **{k: v for k, v in feats.items() if k != "crop_phash_bits"}, **saved})
    return created


def store_images_from_payload(conn: sqlite3.Connection, item_id: int, raw_images: list) -> int:
    """支持 API 直接提交 base64 dataURL 图片（便于脚本/移动端调用）。"""
    import base64
    import binascii

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    count = 0
    order = db.next_sort_order(conn, item_id)
    for entry in raw_images[:MAX_IMAGES_PER_ITEM]:
        if not isinstance(entry, str) or not entry.strip():
            continue
        text = entry.strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[-1]
        try:
            raw = base64.b64decode(text, validate=False)
        except (binascii.Error, ValueError):
            continue
        valid, _ = imf.validate_upload(raw)
        if not valid:
            continue
        try:
            img = imf.load_image(raw, max_side=1600)
            feats = imf.extract_features(img)
            saved = imf.save_image_variants(img, upload_dir, "upload.jpg")
        except Exception:  # noqa: BLE001
            continue
        db.insert_image(conn, item_id, saved=saved, feats=feats, sort_order=order)
        order += 1
        count += 1
    return count


def create_item(conn: sqlite3.Connection, data: dict, files: list, user: Optional[dict]) -> tuple[Optional[int], list[str]]:
    """创建一条失物/拾物记录，返回 (item_id, 错误列表)。"""
    errors: list[str] = []
    item_type = (data.get("type") or "").strip()
    if item_type not in db.ITEM_TYPES:
        errors.append("type 必须是 lost 或 found")
    title = (data.get("title") or "").strip()
    if len(title) < 2:
        errors.append("标题至少 2 个字")
    if len(title) > 80:
        errors.append("标题不超过 80 个字")
    if errors:
        return None, errors

    category_id = None
    category_code = (data.get("category") or data.get("category_code") or "").strip()
    if category_code:
        row = conn.execute("SELECT id FROM categories WHERE code = ?", (category_code,)).fetchone()
        if row:
            category_id = int(row["id"])
    if category_id is None:
        row = conn.execute("SELECT id FROM categories WHERE code = 'other'").fetchone()
        category_id = int(row["id"]) if row else None

    tags = data.get("tags")
    if isinstance(tags, str):
        tags = [t for t in tags.replace("，", ",").split(",")]
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()][:16]

    now = db.now_iso()
    cur = conn.execute(
        """
        INSERT INTO items(type, title, category_id, description, tags, color_name, location, happened_at,
                          contact_name, contact, status, user_id, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            item_type,
            title,
            category_id,
            (data.get("description") or "").strip()[:2000],
            json.dumps(tags, ensure_ascii=False),
            (data.get("color_name") or data.get("color") or "").strip()[:24],
            (data.get("location") or "").strip()[:120],
            (data.get("happened_at") or now)[:32],
            (data.get("contact_name") or (user or {}).get("display_name") or "").strip()[:40],
            (data.get("contact") or (user or {}).get("contact") or "").strip()[:80],
            (data.get("status") if data.get("status") in db.STATUS_CHOICES else "open"),
            int(user["id"]) if user else None,
            now,
            now,
        ),
    )
    item_id = int(cur.lastrowid)

    saved = save_uploaded_images(conn, item_id, files)
    base64_images = data.get("images") or []
    if isinstance(base64_images, list):
        store_images_from_payload(conn, item_id, base64_images)

    # 若用户没有填主色，用第一张图的感知主色自动补全（让文本标签匹配也能用上颜色）
    if not (data.get("color_name") or data.get("color")) and saved:
        dominant = saved[0].get("dominant") or {}
        if dominant.get("label"):
            conn.execute("UPDATE items SET color_name = ? WHERE id = ?", (dominant["label"], item_id))

    conn.commit()
    return item_id, []


# --------------------------------------------------------------------------- #
# 应用工厂
# --------------------------------------------------------------------------- #


def create_app(config: Optional[dict] = None) -> Flask:
    app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"), template_folder=os.path.join(BASE_DIR, "templates"))
    app.config.update(
        DATABASE=os.environ.get("LOSTFOUND_DB", DEFAULT_DB),
        UPLOAD_FOLDER=os.environ.get("LOSTFOUND_UPLOADS", DEFAULT_UPLOADS),
        SECRET_KEY=os.environ.get("LOSTFOUND_SECRET", "cqu-srtp-lostfound-dev-secret-change-me"),
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        JSON_AS_ASCII=False,
        TEMPLATES_AUTO_RELOAD=True,
        SEND_FILE_MAX_AGE_DEFAULT=0,
    )
    if config:
        app.config.update(config)

    os.makedirs(os.path.dirname(app.config["DATABASE"]) or BASE_DIR, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    db.init_db(app.config["DATABASE"])
    with db.connect(app.config["DATABASE"]) as boot_conn:
        ensure_demo_user(boot_conn)

    app.teardown_appcontext(close_db)

    @app.context_processor
    def inject_globals():
        return {
            "current_user": current_user(),
            "item_type_label": db.ITEM_TYPE_LABEL,
            "status_label": db.STATUS_LABEL,
            "app_version": __version__,
            "now": datetime.now(),
        }

    @app.template_filter("dt")
    def _dt(value, fmt="%Y-%m-%d %H:%M"):
        parsed = db.parse_iso(str(value or ""))
        return parsed.strftime(fmt) if parsed else (value or "")

    @app.template_filter("ago")
    def _ago(value):
        return db.humanize_time(str(value or ""))

    @app.template_filter("pct")
    def _pct(value):
        try:
            return f"{round(float(value) * 100)}%"
        except (TypeError, ValueError):
            return "0%"

    @app.template_filter("score_class")
    def _score_class(value):
        level, _ = matcher.match_level(float(value or 0))
        return level

    # ----------------------------------------------------------------- #
    # 页面
    # ----------------------------------------------------------------- #

    @app.route("/")
    def index():
        conn = get_db()
        stats_data = db.stats(conn)
        recent_lost, _ = db.query_items(conn, item_type="lost", limit=6, order="newest")
        recent_found, _ = db.query_items(conn, item_type="found", limit=6, order="newest")
        hot_matches = conn.execute(
            """
            SELECT m.score, m.reasons, m.image_score, m.text_score,
                   li.id AS lost_id, li.title AS lost_title, li.location AS lost_location,
                   fi.id AS found_id, fi.title AS found_title, fi.location AS found_location,
                   (SELECT thumb FROM item_images WHERE item_id = li.id ORDER BY sort_order LIMIT 1) AS lost_thumb,
                   (SELECT thumb FROM item_images WHERE item_id = fi.id ORDER BY sort_order LIMIT 1) AS found_thumb
            FROM matches m
            JOIN items li ON li.id = m.lost_id
            JOIN items fi ON fi.id = m.found_id
            ORDER BY m.score DESC LIMIT 4
            """
        ).fetchall()
        showcases = []
        for row in hot_matches:
            data = dict(row)
            try:
                data["reasons"] = json.loads(row["reasons"]) if row["reasons"] else []
            except (TypeError, ValueError):
                data["reasons"] = []
            showcases.append(data)
        return render_template(
            "index.html",
            stats=stats_data,
            recent_lost=recent_lost,
            recent_found=recent_found,
            showcases=showcases,
            categories=db.list_categories(conn),
        )

    @app.route("/browse")
    def browse_page():
        conn = get_db()
        item_type = request.args.get("type") or "lost"
        if item_type not in db.ITEM_TYPES:
            item_type = "lost"
        keyword = (request.args.get("q") or "").strip()
        category = (request.args.get("category") or "").strip()
        order = request.args.get("order") or "newest"
        page = max(1, to_int(request.args.get("page"), 1))
        per_page = 12
        items, total = db.query_items(
            conn,
            item_type=item_type,
            category=category or None,
            keyword=keyword or None,
            order=order,
            limit=per_page,
            offset=(page - 1) * per_page,
        )
        pages = max(1, (total + per_page - 1) // per_page)
        return render_template(
            "browse.html",
            items=items,
            total=total,
            page=page,
            pages=pages,
            item_type=item_type,
            keyword=keyword,
            category=category,
            order=order,
            categories=db.list_categories(conn),
        )

    @app.route("/item/<int:item_id>")
    def item_page(item_id: int):
        conn = get_db()
        item = db.get_item(conn, item_id)
        if not item:
            abort(404)
        conn.execute("UPDATE items SET view_count = view_count + 1 WHERE id = ?", (item_id,))
        conn.commit()
        matches = matcher.find_matches_for(conn, item_id, top_k=6, persist=True)
        return render_template("item.html", item=item, matches=matches)

    @app.route("/publish")
    @login_required
    def publish_page():
        return render_template(
            "publish.html",
            categories=db.list_categories(get_db()),
            preset_type=request.args.get("type") if request.args.get("type") in db.ITEM_TYPES else "lost",
        )

    @app.route("/login")
    def login_page():
        return render_template("login.html", next=request.args.get("next") or "")

    @app.route("/algorithm")
    def algorithm_page():
        """算法演示：现场对比任意两条文本（可选上传图片）的匹配得分。

        把各通道得分与权重完全暴露给用户，既是教学演示，也是调参入口。
        """
        return render_template(
            "algorithm.html",
            categories=db.list_categories(get_db()),
            weights=matcher.normalize_weights(),
            thresholds={
                "strong": matcher.STRONG_THRESHOLD,
                "candidate": matcher.CANDIDATE_THRESHOLD,
                "weak": matcher.WEAK_THRESHOLD,
            },
        )

    @app.route("/admin")
    @login_required
    def admin_page():
        user = current_user() or {}
        conn = get_db()
        scope = None if user.get("role") == "admin" else int(user.get("id") or 0)
        items, total = db.query_items(conn, user_id=scope, limit=50, order="newest")
        return render_template("admin.html", items=items, total=total, stats=db.stats(conn))

    # ----------------------------------------------------------------- #
    # API: 鉴权
    # ----------------------------------------------------------------- #

    @app.post("/api/auth/register")
    def api_register():
        data = payload()
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not (3 <= len(username) <= 24) or not username.replace("_", "").isalnum():
            return fail("用户名需为 3-24 位字母、数字或下划线")
        if len(password) < 6:
            return fail("密码至少 6 位")
        conn = get_db()
        if db.find_user(conn, username):
            return fail("该用户名已被注册", 409, "conflict")
        conn.execute(
            "INSERT INTO users(username, password_hash, display_name, contact, role, created_at) VALUES (?,?,?,?, 'user', ?)",
            (
                username,
                generate_password_hash(password),
                (data.get("display_name") or username).strip()[:40],
                (data.get("contact") or "").strip()[:80],
                db.now_iso(),
            ),
        )
        conn.commit()
        user = db.find_user(conn, username)
        session["user_id"] = int(user["id"])
        return ok(user={"id": user["id"], "username": user["username"], "display_name": user["display_name"]})

    @app.post("/api/auth/login")
    def api_login():
        data = payload()
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        conn = get_db()
        user = db.find_user(conn, username)
        if not user or not check_password_hash(user["password_hash"], password):
            return fail("用户名或密码不正确", 401, "unauthorized")
        session["user_id"] = int(user["id"])
        return ok(user={"id": user["id"], "username": user["username"], "display_name": user["display_name"], "role": user["role"]})

    @app.post("/api/auth/logout")
    def api_logout():
        session.pop("user_id", None)
        g.pop("user", None)
        return ok()

    @app.get("/api/auth/me")
    def api_me():
        user = current_user()
        if not user:
            return ok(user=None)
        return ok(user={"id": user["id"], "username": user["username"], "display_name": user["display_name"], "role": user["role"], "contact": user["contact"]})

    # ----------------------------------------------------------------- #
    # API: 数据
    # ----------------------------------------------------------------- #

    @app.get("/api/categories")
    def api_categories():
        return ok(categories=db.list_categories(get_db()))

    @app.get("/api/stats")
    def api_stats():
        return ok(stats=db.stats(get_db()))

    @app.get("/api/items")
    def api_items():
        conn = get_db()
        limit = max(1, min(to_int(request.args.get("limit"), 12), 48))
        page = max(1, to_int(request.args.get("page"), 1))
        items, total = db.query_items(
            conn,
            item_type=request.args.get("type") or None,
            category=request.args.get("category") or None,
            status=request.args.get("status") or None,
            keyword=(request.args.get("q") or "").strip() or None,
            order=request.args.get("order") or "newest",
            limit=limit,
            offset=(page - 1) * limit,
        )
        return ok(items=items, total=total, page=page, limit=limit, pages=max(1, (total + limit - 1) // limit))

    @app.get("/api/items/<int:item_id>")
    def api_item(item_id: int):
        item = db.get_item(get_db(), item_id)
        if not item:
            return fail("记录不存在", 404, "not_found")
        return ok(item=item)

    @app.post("/api/items")
    def api_create_item():
        data = payload()
        files = request.files.getlist("images") or request.files.getlist("image")
        conn = get_db()
        item_id, errors = create_item(conn, data, files, current_user())
        if errors:
            return fail("；".join(errors), 422, "validation_error")
        with_images = data.get("match", "1") not in ("0", 0, False, "false")
        matches = matcher.find_matches_for(conn, item_id, top_k=6, persist=True) if with_images else []
        return ok(item=db.get_item(conn, item_id), matches=matches, match_count=len(matches)), 201

    @app.delete("/api/items/<int:item_id>")
    @login_required
    def api_delete_item(item_id: int):
        conn = get_db()
        item = db.get_item(conn, item_id)
        if not item:
            return fail("记录不存在", 404, "not_found")
        user = current_user() or {}
        if user.get("role") != "admin" and int(item.get("user_id") or 0) != int(user.get("id") or 0):
            return fail("只能删除自己发布的记录", 403, "forbidden")
        for image in item.get("images") or []:
            for name in (image.get("filename"), image.get("thumb")):
                if not name:
                    continue
                path = os.path.join(current_app.config["UPLOAD_FOLDER"], name)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
        return ok(deleted=item_id)

    # ----------------------------------------------------------------- #
    # API: 匹配
    # ----------------------------------------------------------------- #

    @app.get("/api/items/<int:item_id>/matches")
    def api_matches(item_id: int):
        conn = get_db()
        if not db.get_item(conn, item_id):
            return fail("记录不存在", 404, "not_found")
        top_k = max(1, min(to_int(request.args.get("top_k"), 6), 20))
        threshold = request.args.get("threshold")
        try:
            threshold = float(threshold) if threshold is not None else matcher.WEAK_THRESHOLD
        except (TypeError, ValueError):
            threshold = matcher.WEAK_THRESHOLD
        weights = {
            "image": request.args.get("w_image"),
            "text": request.args.get("w_text"),
            "location": request.args.get("w_location"),
        }
        results = matcher.find_matches_for(
            conn, item_id, top_k=top_k, threshold=max(0.0, min(threshold, 1.0)), weights=weights, persist=True
        )
        return ok(item_id=item_id, matches=results, count=len(results), weights=matcher.normalize_weights(weights))

    @app.post("/api/match/pair")
    def api_match_pair():
        """任意两条记录的完整比对（含图像通道），不写库。

        与 ``/api/match/text`` 的区别：这里会在服务端对提交的图片做感知特征提取，
        因此「图像感知 + 文本标签」两条通道都能参与打分，适合算法演示台与调参。
        """
        data = payload()
        pair = []
        for key in ("left", "right"):
            side = data.get(key) or {}
            if not isinstance(side, dict):
                return fail(f"{key} 需要是对象")
            tags = side.get("tags")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.replace("，", ",").split(",") if t.strip()]
            entry = {
                "id": 1 if key == "left" else 2,
                "type": side.get("type") or ("lost" if key == "left" else "found"),
                "title": str(side.get("title") or ""),
                "description": str(side.get("description") or ""),
                "tags": [str(t).strip() for t in (tags or []) if str(t).strip()][:16],
                "color_name": str(side.get("color_name") or ""),
                "location": str(side.get("location") or ""),
                "category_name": str(side.get("category_name") or ""),
                "images": [],
            }
            entry["images"] = build_probe_images(side.get("images"))
            pair.append(entry)

        if not pair[0]["title"] and not pair[1]["title"]:
            return fail("请至少填写一侧的标题")

        scored = matcher.score_pair(pair[0], pair[1], weights=data.get("weights"))
        return ok(result=scored, sides=[{"title": p["title"], "images": len(p["images"])} for p in pair])

    @app.post("/api/match/text")
    def api_match_text():
        """纯文本标签匹配：不落库、不要求登录，用于快速试算与算法调参演示。"""
        data = payload()
        left = data.get("left") or {}
        right = data.get("right") or {}
        if not isinstance(left, dict) or not isinstance(right, dict):
            return fail("left / right 需要是对象")
        dummy_a = {
            "id": 1,
            "type": "lost",
            "title": str(left.get("title") or ""),
            "description": str(left.get("description") or ""),
            "tags": left.get("tags") or [],
            "color_name": left.get("color_name") or "",
            "location": left.get("location") or "",
            "category_name": left.get("category_name") or "",
            "images": [],
        }
        dummy_b = {
            "id": 2,
            "type": "found",
            "title": str(right.get("title") or ""),
            "description": str(right.get("description") or ""),
            "tags": right.get("tags") or [],
            "color_name": right.get("color_name") or "",
            "location": right.get("location") or "",
            "category_name": right.get("category_name") or "",
            "images": [],
        }
        scored = matcher.score_pair(dummy_a, dummy_b, weights=data.get("weights"))
        return ok(result=scored)

    @app.post("/api/items/<int:item_id>/match")
    def api_match_with_image(item_id: int):
        """以「临时上传的图片 + 文本」直接检索库内对手方记录（不落库）。

        典型用法：捡到东西的人先拍照试查，再决定是否发布。
        """
        conn = get_db()
        left = db.get_item(conn, item_id)
        if not left:
            return fail("记录不存在", 404, "not_found")
        data = payload()

        probe_images = build_probe_images(data.get("images"))
        probe_images.extend(
            build_probe_images_from_files(request.files.getlist("images") or request.files.getlist("image"))
        )

        target = dict(left)
        if probe_images:
            target["images"] = probe_images
        for field in ("title", "description", "location", "color_name"):
            if data.get(field):
                target[field] = str(data[field])
        if data.get("tags"):
            tags = data["tags"]
            target["tags"] = [t.strip() for t in (tags.split(",") if isinstance(tags, str) else tags) if str(t).strip()]

        counterpart = "found" if target["type"] == "lost" else "lost"
        others, _ = db.query_items(conn, item_type=counterpart, limit=100)
        corpus = matcher.build_corpus_for(target, others)
        results = []
        for other in others:
            scored = matcher.score_pair(target, other, corpus=corpus, weights=data.get("weights"))
            if scored["score"] >= matcher.WEAK_THRESHOLD:
                results.append({"item": other, **scored})
        results.sort(key=lambda r: -r["score"])
        return ok(probe_images=len(probe_images), matches=results[:10], count=len(results))

    @app.post("/api/match/recompute")
    @login_required
    def api_recompute():
        user = current_user() or {}
        if user.get("role") != "admin":
            return fail("仅管理员可执行全量重算", 403, "forbidden")
        data = payload()
        result = matcher.recompute_all(get_db(), weights=data.get("weights"), clear=bool(data.get("clear", True)))
        return ok(result=result)

    @app.get("/api/matches/top")
    def api_top_matches():
        conn = get_db()
        limit = max(1, min(to_int(request.args.get("limit"), 8), 50))
        min_score = request.args.get("min_score")
        try:
            min_score = float(min_score) if min_score is not None else matcher.WEAK_THRESHOLD
        except (TypeError, ValueError):
            min_score = matcher.WEAK_THRESHOLD
        rows = conn.execute(
            """
            SELECT m.lost_id, m.found_id, m.score, m.image_score, m.text_score, m.reasons
            FROM matches m WHERE m.score >= ? ORDER BY m.score DESC LIMIT ?
            """,
            (min_score, limit),
        ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["reasons"] = matcher._json_list(row["reasons"])  # noqa: SLF001 - 同包内部工具
            lost = db.get_item(conn, int(row["lost_id"]))
            found = db.get_item(conn, int(row["found_id"]))
            data["lost"] = {"id": lost["id"], "title": lost["title"], "thumb": (lost.get("images") or [{}])[0].get("thumb")} if lost else None
            data["found"] = {"id": found["id"], "title": found["title"], "thumb": (found.get("images") or [{}])[0].get("thumb")} if found else None
            out.append(data)
        return ok(matches=out)

    # ----------------------------------------------------------------- #
    # 错误处理
    # ----------------------------------------------------------------- #

    @app.errorhandler(404)
    def _404(_err):
        if request.path.startswith("/api/"):
            return fail("接口不存在", 404, "not_found")
        return render_template("error.html", code=404, message="页面不存在或记录已被删除"), 404

    @app.errorhandler(413)
    def _413(_err):
        return fail(f"上传内容过大（上限 {MAX_CONTENT_LENGTH // 1024 // 1024}MB）", 413, "payload_too_large")

    @app.errorhandler(500)
    def _500(err):  # pragma: no cover
        current_app.logger.exception("未捕获异常: %s", err)
        if request.path.startswith("/api/"):
            return fail("服务器内部错误", 500, "internal_error")
        return render_template("error.html", code=500, message="服务器内部错误"), 500

    return app


def run() -> None:  # pragma: no cover - 开发入口
    app = create_app()
    host = os.environ.get("LOSTFOUND_HOST", "127.0.0.1")
    port = int(os.environ.get("LOSTFOUND_PORT", "5000"))
    debug = os.environ.get("LOSTFOUND_DEBUG", "0") == "1"
    print(f"* 失物招领系统启动中 → http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
