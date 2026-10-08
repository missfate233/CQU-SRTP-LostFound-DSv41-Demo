"""端到端测试：``python -m unittest discover -s tests -v``

覆盖范围
--------
1. **图像感知通道**：pHash 对「缩放 / 亮度变化 / JPEG 压缩」的鲁棒性，以及对
   「不同颜色 / 不同形状」的区分度；
2. **文本标签通道**：中文分词、同义词归一化、TF-IDF 余弦、标签模糊重合；
3. **融合引擎**：权重归一化、缺图时权重让渡、证据文案生成、阈值分级；
4. **REST API**：发布（含真实图片上传）→ 匹配 → 检索 → 删除 的完整闭环，
   以及鉴权、参数校验、错误码；
5. **序列化安全**：所有返回体必须可被 JSON 序列化（防止 bytes/BLOB 漏出）。

测试使用临时目录与临时数据库，不污染 data/ 与 static/uploads/。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import unittest
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lostfound import db, image_features as imf, matcher, text_features as tf  # noqa: E402


# --------------------------------------------------------------------------- #
# 测试用图片生成
# --------------------------------------------------------------------------- #


def make_photo(
    color: tuple[int, int, int],
    *,
    background: tuple[int, int, int] = (238, 238, 240),
    size: tuple[int, int] = (400, 300),
    scale: float = 0.6,
    brightness: float = 1.0,
    shape: str = "square",
    jpeg_quality: int = 92,
) -> bytes:
    """合成一张「背景 + 主体」的测试照片，返回 JPEG 字节。"""
    canvas = Image.new("RGB", size, background)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box_size = int(min(size) * scale)
    left = (size[0] - box_size) // 2
    top = (size[1] - box_size) // 2
    box = [left, top, left + box_size, top + box_size]
    if shape == "circle":
        draw.ellipse(box, fill=(*color, 255))
    elif shape == "triangle":
        draw.polygon(
            [(left + box_size // 2, top), (left, top + box_size), (left + box_size, top + box_size)],
            fill=(*color, 255),
        )
    else:
        draw.rectangle(box, fill=(*color, 255))
    canvas.paste(layer, (0, 0), layer)

    if brightness != 1.0:
        overlay = Image.new("RGB", size, (255, 255, 255) if brightness > 1 else (0, 0, 0))
        canvas = Image.blend(canvas, overlay, abs(brightness - 1.0) * 0.4)

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def png_bytes(color=(30, 120, 220), size=(60, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class TempEnvMixin:
    """为每个测试准备独立的数据库与上传目录。

    临时目录放在**项目内**的 ``_testtmp/``，并且用 ``os.makedirs`` 创建而**不是**
    ``tempfile.mkdtemp``：受管环境（Windows 沙箱）会对 mkdtemp 生成的目录施加
    限制性 ACL，随后任何子目录创建/文件写入都会得到 ``WinError 5 拒绝访问``。
    用普通 makedirs 建目录则完全正常。
    """

    TMP_ROOT = os.path.join(BASE_DIR, "_testtmp")

    def setUp(self) -> None:  # noqa: D102
        os.makedirs(self.TMP_ROOT, exist_ok=True)
        self.tmpdir = os.path.join(self.TMP_ROOT, f"run_{uuid.uuid4().hex[:8]}")
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.upload_dir = os.path.join(self.tmpdir, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self._old_env = {k: os.environ.get(k) for k in ("LOSTFOUND_DB", "LOSTFOUND_UPLOADS")}
        os.environ["LOSTFOUND_DB"] = self.db_path
        os.environ["LOSTFOUND_UPLOADS"] = self.upload_dir

    def tearDown(self) -> None:  # noqa: D102
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def connect(self):
        db.init_db(self.db_path)
        return db.connect(self.db_path)


# --------------------------------------------------------------------------- #
# 1. 图像感知通道
# --------------------------------------------------------------------------- #


class TestImageChannel(unittest.TestCase):
    def test_image_similarity_stable_under_scale_and_brightness(self):
        """同一物体：缩放、亮度变化、JPEG 质量下降后相似度仍应很高。

        这是本项目最关键的一条回归测试：早期版本用「裁切后的 pHash」做形状通道，
        同一方块换尺度后汉明距离高达 24~32（与「换成圆形」的 21 相当），
        图像通道几乎失去区分度。改用尺度归一化轮廓 + 前景颜色直方图后，
        同物体各变体稳定在 0.85 以上。
        """
        base = imf.extract_features(imf.load_image(make_photo((200, 60, 60), scale=0.6)))
        for label, blob in (
            ("缩放 0.9", make_photo((200, 60, 60), scale=0.9)),
            ("缩放 0.8", make_photo((200, 60, 60), scale=0.8)),
            ("缩放 0.45", make_photo((200, 60, 60), scale=0.45)),
            ("缩放 0.4", make_photo((200, 60, 60), scale=0.4)),
            ("亮度 +12%", make_photo((200, 60, 60), scale=0.6, brightness=1.12)),
            ("亮度 -12%", make_photo((200, 60, 60), scale=0.6, brightness=0.88)),
            ("低质量 JPEG", make_photo((200, 60, 60), scale=0.6, jpeg_quality=45)),
        ):
            other = imf.extract_features(imf.load_image(blob))
            score, detail = imf.image_pair_similarity(base, other)
            self.assertGreater(score, 0.8, f"{label} 的图像相似度过低：{score:.3f} ({detail})")

    def test_phash_is_kept_only_as_a_weak_hint(self):
        """pHash 仍然保留，但只作为「疑似同一张照片」的弱证据。"""
        base = imf.extract_features(imf.load_image(make_photo((200, 60, 60), scale=0.6)))
        same_photo = imf.extract_features(imf.load_image(make_photo((200, 60, 60), scale=0.6)))
        rescaled = imf.extract_features(imf.load_image(make_photo((200, 60, 60), scale=0.42)))
        _, detail_same = imf.image_pair_similarity(base, same_photo)
        _, detail_scaled = imf.image_pair_similarity(base, rescaled)
        self.assertEqual(detail_same["hamming"], 0, "完全相同的照片哈希距离应为 0")
        self.assertGreater(
            detail_scaled["hamming"], 10,
            "换尺度后整图哈希距离本就很大（这正是不能把它当主通道的原因）",
        )

    def test_shape_descriptor_distinguishes_shapes(self):
        """轮廓通道必须把「方块 vs 圆形」区分开（尺度归一化后仍可区分）。"""
        square = imf.extract_features(imf.load_image(make_photo((60, 60, 60), shape="square", scale=0.6)))
        circle = imf.extract_features(imf.load_image(make_photo((60, 60, 60), shape="circle", scale=0.6)))
        triangle = imf.extract_features(imf.load_image(make_photo((60, 60, 60), shape="triangle", scale=0.6)))
        square_bigger = imf.extract_features(imf.load_image(make_photo((60, 60, 60), shape="square", scale=0.85)))

        same_shape = imf.shape_score(square, square_bigger)
        diff_shape = imf.shape_score(square, triangle)
        circle_vs_square = imf.shape_score(square, circle)

        self.assertGreater(same_shape, 0.95, f"同形状（不同尺度）应几乎一致：{same_shape}")
        self.assertLess(diff_shape, same_shape - 0.15, f"三角与方块应明显区分：{diff_shape}")
        self.assertLess(circle_vs_square, same_shape, "圆形与方形应低于同形状")

    def test_image_similarity_distinguishes_different_objects(self):
        """颜色 + 形状都不同的物体，图像相似度必须显著更低。"""
        red_square = imf.extract_features(imf.load_image(make_photo((200, 60, 60), shape="square")))
        same = imf.extract_features(imf.load_image(make_photo((200, 60, 60), shape="square", scale=0.75)))
        blue_triangle = imf.extract_features(imf.load_image(make_photo((40, 70, 210), shape="triangle")))
        red_triangle = imf.extract_features(imf.load_image(make_photo((200, 60, 60), shape="triangle")))

        score_same, _ = imf.image_pair_similarity(red_square, same)
        score_diff_shape, _ = imf.image_pair_similarity(red_square, red_triangle)
        score_diff_all, _ = imf.image_pair_similarity(red_square, blue_triangle)

        self.assertGreater(score_same, score_diff_shape, "同形状应高于异形状（同色）")
        self.assertGreater(score_same, score_diff_all, "同物体应高于颜色形状都不同的物体")
        self.assertGreater(score_same - score_diff_all, 0.15, "区分度不足，无法用于排序")

    def test_foreground_color_histogram_reacts_to_color(self):
        """前景颜色直方图必须能区分「同形状不同颜色」，且对亮度变化不敏感。

        注意必须用 ``object_color_histogram``（前景掩膜内统计）而不是整图
        ``color_histogram``：整图直方图里背景占多数，物体颜色会被稀释，
        这在真实照片上尤其明显。
        """
        black = imf.object_color_histogram(imf.load_image(make_photo((20, 20, 22))))
        black_rescaled = imf.object_color_histogram(imf.load_image(make_photo((20, 20, 22), scale=0.75)))
        black_brighter = imf.object_color_histogram(imf.load_image(make_photo((20, 20, 22), brightness=1.12)))
        pink = imf.object_color_histogram(imf.load_image(make_photo((230, 120, 170))))

        self.assertGreater(imf.histogram_similarity(black, black_rescaled), 0.85, "同色不同尺度应高度相似")
        self.assertGreater(imf.histogram_similarity(black, black_brighter), 0.85, "同色不同亮度应高度相似")
        self.assertLess(imf.histogram_similarity(black, pink), 0.5, "黑与粉应明显不同")

    def test_foreground_segmentation_finds_object(self):
        """前景分割应圈出主体，且掩膜与裁切图尺寸一致。"""
        img = imf.load_image(make_photo((30, 140, 90), scale=0.5))
        cropped, mask, ratio, box = imf.object_blob(img)
        self.assertEqual(mask.shape, (cropped.height, cropped.width), "掩膜与裁切图尺寸必须一致")
        self.assertGreater(ratio, 0.05, f"前景占比过低，分割可能失败：{ratio}")
        self.assertLess(ratio, 0.9, f"前景占比过高，可能把背景算进来了：{ratio}")
        self.assertTrue(0 <= box[0] < box[2] <= img.width and 0 <= box[1] < box[3] <= img.height, "包围盒越界")

    def test_blob_box_is_in_input_coordinates(self):
        """回归：包围盒必须是**传入图像**的像素坐标。

        早期版本在降采样图上检测却直接返回降采样坐标，导致 640x480 图片的掩膜
        整体偏移到背景上（"黑色书包"的前景主色被算成"浅米色"）。
        """
        for size in ((640, 480), (320, 240), (200, 200)):
            img = imf.load_image(make_photo((30, 32, 38), size=size, scale=0.5))
            cropped, mask, _ratio, box = imf.object_blob(img)
            self.assertTrue(
                0 <= box[0] < box[2] <= img.width and 0 <= box[1] < box[3] <= img.height,
                f"{size} 的包围盒 {box} 超出图像范围 {img.size}",
            )
            # 掩膜覆盖的像素必须真的是物体（深色），而不是背景
            working, full = imf._mask_on_working(img, mask, box)
            self.assertIsNotNone(full, f"{size} 的掩膜不应为空")
            pixels = np.asarray(working, dtype=np.float32)[full]
            self.assertLess(float(pixels.mean()), 120, f"{size} 的掩膜平均亮度 {pixels.mean():.0f} 偏高，疑似选到背景")

    def test_foreground_dominant_color_ignores_background(self):
        """前景主色必须是物体颜色，而不是背景颜色。"""
        obj_color = (35, 40, 48)                    # 近黑
        img = imf.load_image(make_photo(obj_color, background=(240, 236, 228), scale=0.55))
        cropped, mask, _ratio, box = imf.object_blob(img)
        working, full = imf._mask_on_working(img, mask, box)
        masked = imf.dominant_color(working, mask=full)
        unmasked = imf.dominant_color(img)          # 整图（背景占多数）
        self.assertEqual(masked["name"], "黑色", f"前景主色应为黑色，实际 {masked}")
        self.assertNotEqual(unmasked["name"], "黑色", "整图主色应被背景主导（用于对照）")

    def test_hamming_and_similarity_bounds(self):
        bits_a = imf.phash_bits(imf.load_image(make_photo((10, 10, 10))))
        bits_b = imf.phash_bits(imf.load_image(make_photo((250, 250, 250), shape="triangle")))
        self.assertEqual(imf.hamming(bits_a, bits_a), 0)
        self.assertEqual(imf.phash_similarity(bits_a, bits_a), 1.0)
        self.assertLessEqual(imf.hamming(bits_a, bits_b), 64)
        self.assertGreaterEqual(imf.phash_similarity(bits_a, bits_b), 0.0)

    def test_validate_upload_rejects_garbage(self):
        good, _ = imf.validate_upload(make_photo((1, 2, 3)))
        bad, reason = imf.validate_upload(b"this is definitely not an image")
        empty, _ = imf.validate_upload(b"")
        self.assertTrue(good)
        self.assertFalse(bad)
        self.assertIn("无法解析", reason)
        self.assertFalse(empty)


# --------------------------------------------------------------------------- #
# 2. 文本标签通道
# --------------------------------------------------------------------------- #


class TestTextChannel(unittest.TestCase):
    def test_tokenize_chinese_and_english(self):
        tokens = tf.tokenize("黑色蓝牙耳机 AirPods Pro 掉落")
        self.assertIn("黑", tokens)
        self.assertIn("蓝牙", tokens)
        self.assertIn("airpods", tokens)
        self.assertNotIn("的", tokens, "停用词必须被过滤")

    def test_synonym_normalization(self):
        self.assertEqual(tf.normalize_tag("蓝牙耳机"), "耳机")
        self.assertEqual(tf.normalize_tag("一卡通"), "校园卡")
        self.assertEqual(tf.normalize_tag(" 黑色 "), "黑色")

    def test_same_item_scores_high_different_item_low(self):
        corpus = tf.build_corpus(
            [
                {"id": 1, "title": "黑色双肩包", "tags": ["黑色", "双肩包", "图书馆"], "location": "图书馆四楼"},
                {"id": 2, "title": "图书馆捡到黑色背包", "tags": ["黑色", "背包", "图书馆"], "location": "图书馆"},
                {"id": 3, "title": "粉色雨伞", "tags": ["粉色", "雨伞"], "location": "二食堂"},
            ]
        )
        same, _ = tf.text_similarity(
            {"id": 1, "title": "黑色双肩包", "tags": ["黑色", "双肩包", "图书馆"], "location": "图书馆四楼"},
            {"id": 2, "title": "图书馆捡到黑色背包", "tags": ["黑色", "背包", "图书馆"], "location": "图书馆"},
            corpus,
        )
        diff, _ = tf.text_similarity(
            {"id": 1, "title": "黑色双肩包", "tags": ["黑色", "双肩包", "图书馆"], "location": "图书馆四楼"},
            {"id": 3, "title": "粉色雨伞", "tags": ["粉色", "雨伞"], "location": "二食堂"},
            corpus,
        )
        self.assertGreater(same, 0.4, f"同一物品的文本相似度偏低：{same}")
        self.assertLess(diff, same / 2, f"不同物品的文本相似度偏高：{diff}")

    def test_tag_overlap_fuzzy(self):
        score, hits = tf.tag_overlap(["黑色耳机", "白色"], ["耳机", "白色"])
        self.assertGreater(score, 0.6)
        self.assertTrue(hits, "应命中模糊等价标签")

    def test_location_similarity(self):
        self.assertEqual(tf.location_similarity("图书馆四楼", "图书馆四楼"), 1.0)
        self.assertGreater(tf.location_similarity("图书馆四楼自习区", "图书馆"), 0.8)
        self.assertGreater(tf.location_similarity("二食堂门口", "二食堂北门"), 0.8)
        self.assertLess(tf.location_similarity("图书馆", "体育馆"), 0.4)

    def test_normalize_tags_dedup_and_split(self):
        tags = tf.normalize_tags("黑色, 黑色、双肩包 书包；蓝牙耳机")
        self.assertEqual(len(tags), len(set(tags)), "标签必须去重")
        self.assertIn("耳机", tags)


# --------------------------------------------------------------------------- #
# 3. 融合引擎
# --------------------------------------------------------------------------- #


class TestMatcher(unittest.TestCase):
    def _item(self, **kwargs):
        base = {
            "id": 1, "type": "lost", "title": "", "description": "", "tags": [],
            "color_name": "", "location": "", "category_name": "", "images": [],
        }
        base.update(kwargs)
        return base

    def test_weight_normalization(self):
        weights = matcher.normalize_weights({"image": 3, "text": 1, "location": 0})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertAlmostEqual(weights["image"], 0.75, places=6)
        # 非法输入应回落到默认权重而不是崩溃
        fallback = matcher.normalize_weights({"image": "abc", "text": None})
        self.assertAlmostEqual(sum(fallback.values()), 1.0, places=6)

    def test_missing_image_redistributes_weights(self):
        weights = matcher.normalize_weights()
        shifted = matcher.rebalance_for_missing_image(weights, has_image=False)
        self.assertEqual(shifted["image"], 0.0)
        self.assertAlmostEqual(shifted["text"] + shifted["location"], 1.0, places=6)
        untouched = matcher.rebalance_for_missing_image(weights, has_image=True)
        self.assertEqual(untouched, weights)

    def test_score_pair_without_images_uses_text(self):
        left = self._item(id=1, title="黑色双肩包", tags=["黑色", "双肩包", "图书馆"], location="图书馆四楼")
        right = self._item(id=2, type="found", title="图书馆捡到黑色背包", tags=["黑色", "背包", "图书馆"], location="图书馆")
        scored = matcher.score_pair(left, right)
        self.assertGreater(scored["score"], 0.3)
        self.assertEqual(scored["image_score"], 0.0)
        self.assertGreater(scored["text_score"], 0.3)
        self.assertTrue(scored["reasons"])
        self.assertIn(scored["level"], ("strong", "likely", "candidate", "weak"))

    def test_match_levels_are_monotonic(self):
        order = [matcher.match_level(s)[0] for s in (0.9, 0.6, 0.45, 0.32, 0.1)]
        self.assertEqual(order, ["strong", "strong", "likely", "candidate", "weak"])

    def test_reasons_are_human_readable(self):
        reasons = matcher.build_reasons(
            0.8,
            {"shape": 0.9, "color": 0.95, "hamming": 2, "phash": 0.95, "dominant_labels": ["黑色", "黑色"]},
            0.6,
            {"tag_hits": ["黑色", "书包"], "tfidf": 0.5, "color_text": 1.0},
            0.9,
        )
        joined = " ".join(reasons)
        self.assertIn("轮廓", joined)
        self.assertIn("标签命中", joined)
        self.assertIn("地点吻合", joined)
        self.assertTrue(all(isinstance(r, str) and r for r in reasons))

    def test_evidence_never_empty(self):
        reasons = matcher.build_reasons(0.0, {}, 0.0, {}, 0.0)
        self.assertTrue(reasons, "即使全无相关性也必须给出可读说明")


# --------------------------------------------------------------------------- #
# 4. REST API 端到端
# --------------------------------------------------------------------------- #


class TestApi(TempEnvMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        from lostfound.app import create_app

        self.app = create_app(
            {
                "DATABASE": self.db_path,
                "UPLOAD_FOLDER": self.upload_dir,
                "TESTING": True,
                "SECRET_KEY": "test-secret",
            }
        )
        self.client = self.app.test_client()

    # ---------------- 基础只读接口 ---------------- #

    def test_pages_render(self):
        for path in ("/", "/browse?type=lost", "/browse?type=found", "/login", "/algorithm"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, f"{path} 应返回 200")
            self.assertIn("text/html", response.headers["Content-Type"])

    def test_404_page_and_api(self):
        self.assertEqual(self.client.get("/no-such-page").status_code, 404)
        response = self.client.get("/api/items/999999")
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["ok"])

    def test_stats_and_categories(self):
        stats = self.client.get("/api/stats").get_json()
        self.assertTrue(stats["ok"])
        self.assertIn("total_count", stats["stats"])
        categories = self.client.get("/api/categories").get_json()
        self.assertTrue(len(categories["categories"]) >= 10)

    def test_public_pages_require_no_login(self):
        """浏览与匹配查询必须对未登录用户开放。"""
        self.assertEqual(self.client.get("/api/items").status_code, 200)
        self.assertEqual(self.client.get("/api/items/1/matches").status_code, 404)  # 记录不存在

    # ---------------- 鉴权 ---------------- #

    def test_login_logout_flow(self):
        bad = self.client.post("/api/auth/login", json={"username": "demo", "password": "wrong"})
        self.assertEqual(bad.status_code, 401)

        good = self.client.post("/api/auth/login", json={"username": "demo", "password": "demo1234"})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(good.get_json()["user"]["username"], "demo")

        me = self.client.get("/api/auth/me").get_json()
        self.assertIsNotNone(me["user"])

        self.client.post("/api/auth/logout")
        self.assertIsNone(self.client.get("/api/auth/me").get_json()["user"])

    def test_register_validation_and_duplicate(self):
        short = self.client.post("/api/auth/register", json={"username": "ab", "password": "123456"})
        self.assertEqual(short.status_code, 400)
        weak = self.client.post("/api/auth/register", json={"username": "newuser", "password": "123"})
        self.assertEqual(weak.status_code, 400)

        created = self.client.post(
            "/api/auth/register",
            json={"username": "newuser", "password": "secret123", "display_name": "新用户"},
        )
        self.assertEqual(created.status_code, 200)
        duplicate = self.client.post(
            "/api/auth/register", json={"username": "newuser", "password": "secret123"}
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_publish_page_requires_login(self):
        response = self.client.get("/publish")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    # ---------------- 发布 + 匹配闭环 ---------------- #

    def test_publish_with_image_then_match(self):
        """发布两条记录：带图的寻物启事与带图的失物招领，应互相匹配上。"""
        photo_a = make_photo((28, 30, 36), background=(235, 232, 226), scale=0.62, brightness=0.95)
        photo_b = make_photo((28, 30, 36), background=(214, 226, 240), scale=0.78, brightness=1.08)

        created_lost = self.client.post(
            "/api/items",
            data={
                "type": "lost",
                "title": "黑色双肩包（内装教材）",
                "category": "bag",
                "description": "图书馆四楼自习区落下的黑色双肩包，里面有教材和充电器。",
                "tags": "黑色,双肩包,书包,图书馆,教材",
                "color_name": "黑色",
                "location": "图书馆四楼自习区",
                "contact": "13800000000",
                "images": (io.BytesIO(photo_a), "lost.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(created_lost.status_code, 201, created_lost.get_data(as_text=True))
        body = created_lost.get_json()
        self.assertTrue(body["ok"])
        lost_id = body["item"]["id"]
        self.assertEqual(len(body["item"]["images"]), 1)
        self.assertTrue(body["item"]["images"][0]["crop_phash_hex"], "必须存下前景感知哈希")

        created_found = self.client.post(
            "/api/items",
            data={
                "type": "found",
                "title": "图书馆捡到黑色背包",
                "category": "bag",
                "description": "在图书馆四楼发现黑色背包，内有教材和电源适配器。",
                "tags": "黑色,背包,书包,图书馆,教材,电源适配器",
                "color_name": "黑色",
                "location": "图书馆",
                "contact": "13900000000",
                "images": (io.BytesIO(photo_b), "found.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(created_found.status_code, 201)
        found_id = created_found.get_json()["item"]["id"]

        # 发布拾物时应立刻匹配到刚发布的寻物启事
        matches = created_found.get_json()["matches"]
        self.assertTrue(matches, "应至少匹配到一条候选")
        best = matches[0]
        self.assertEqual(best["item"]["id"], lost_id)
        self.assertGreater(best["score"], matcher.CANDIDATE_THRESHOLD)
        self.assertGreater(best["image_score"], 0.0, "图像通道应参与打分")
        self.assertTrue(best["reasons"])

        # 反向查询同样成立
        reverse = self.client.get(f"/api/items/{lost_id}/matches").get_json()
        self.assertTrue(reverse["ok"])
        self.assertEqual(reverse["matches"][0]["item"]["id"], found_id)
        self.assertIn("weights", reverse)

        # 图片文件确实落盘
        for image in body["item"]["images"]:
            self.assertTrue(os.path.isfile(os.path.join(self.upload_dir, image["filename"])))
            self.assertTrue(os.path.isfile(os.path.join(self.upload_dir, image["thumb"])))

    def test_publish_validation_errors(self):
        missing = self.client.post("/api/items", data={"type": "lost", "title": ""})
        self.assertEqual(missing.status_code, 422)
        self.assertFalse(missing.get_json()["ok"])
        bad_type = self.client.post("/api/items", data={"type": "unknown", "title": "测试标题"})
        self.assertEqual(bad_type.status_code, 422)

    def test_publish_text_only_then_match(self):
        """无图记录也必须能匹配（文本通道顶替图像权重）。"""
        self.client.post(
            "/api/items",
            data={
                "type": "lost",
                "title": "线性代数教材与蓝色笔记本",
                "category": "book",
                "tags": "教材,线性代数,笔记本,B区",
                "location": "B 区教学楼 305",
            },
        )
        found = self.client.post(
            "/api/items",
            data={
                "type": "found",
                "title": "B 区 305 教室发现教材和笔记本",
                "category": "book",
                "tags": "教材,线性代数,笔记本,B区",
                "location": "B 区教学楼 305",
            },
        )
        self.assertEqual(found.status_code, 201)
        matches = found.get_json()["matches"]
        self.assertTrue(matches, "纯文本记录也必须能匹配")
        self.assertGreater(matches[0]["text_score"], 0.4)
        self.assertEqual(matches[0]["image_score"], 0.0)
        self.assertEqual(matches[0]["detail"]["weights"]["image"], 0.0, "缺图时图像权重应为 0")

    def test_list_filtering_and_search(self):
        self.client.post("/api/items", data={"type": "lost", "title": "粉色折叠雨伞", "category": "umbrella", "tags": "粉色,雨伞"})
        self.client.post("/api/items", data={"type": "found", "title": "捡到一把粉色雨伞", "category": "umbrella", "tags": "粉色,伞"})

        lost_only = self.client.get("/api/items?type=lost").get_json()
        self.assertTrue(all(it["type"] == "lost" for it in lost_only["items"]))
        self.assertEqual(lost_only["total"], 1)

        by_keyword = self.client.get("/api/items?q=雨伞").get_json()
        self.assertGreaterEqual(by_keyword["total"], 2)

        by_category = self.client.get("/api/items?category=umbrella").get_json()
        self.assertEqual(by_category["total"], 2)

        empty = self.client.get("/api/items?q=不存在的关键词xyz").get_json()
        self.assertEqual(empty["total"], 0)

    def test_item_detail_and_view_count(self):
        created = self.client.post("/api/items", data={"type": "lost", "title": "测试计数物品", "tags": "测试"}).get_json()
        item_id = created["item"]["id"]
        self.client.get(f"/item/{item_id}")
        detail = self.client.get(f"/api/items/{item_id}").get_json()
        self.assertGreaterEqual(detail["item"]["view_count"], 0)
        self.assertIn("images", detail["item"])
        self.assertIn("tags", detail["item"])

    # ---------------- 算法试算接口 ---------------- #

    def test_match_text_endpoint(self):
        response = self.client.post(
            "/api/match/text",
            json={
                "left": {"title": "黑色双肩包", "tags": ["黑色", "双肩包"], "location": "图书馆"},
                "right": {"title": "图书馆捡到黑色背包", "tags": ["黑色", "背包"], "location": "图书馆"},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["result"]
        self.assertGreater(result["score"], 0.0)
        self.assertIn("detail", result)
        self.assertIn("reasons", result)

    def test_match_pair_endpoint_with_images(self):
        """成对试算接口必须真正使用图像通道。"""
        left = make_photo((30, 90, 200), background=(240, 240, 240), scale=0.6)
        right_same = make_photo((30, 90, 200), background=(220, 226, 236), scale=0.8, brightness=1.1)
        right_diff = make_photo((220, 60, 60), background=(220, 226, 236), scale=0.6)

        import base64

        def payload(right_blob):
            return {
                "left": {
                    "title": "蓝色物品 A",
                    "tags": ["蓝色"],
                    "images": ["data:image/jpeg;base64," + base64.b64encode(left).decode()],
                },
                "right": {
                    "title": "蓝色物品 B",
                    "tags": ["蓝色"],
                    "images": ["data:image/jpeg;base64," + base64.b64encode(right_blob).decode()],
                },
            }

        same = self.client.post("/api/match/pair", json=payload(right_same)).get_json()["result"]
        diff = self.client.post("/api/match/pair", json=payload(right_diff)).get_json()["result"]
        self.assertGreater(same["image_score"], 0.0, "图像通道必须生效")
        self.assertGreater(same["image_score"], diff["image_score"], "同色物体应比异色物体更相似")
        self.assertGreater(same["score"], diff["score"], "总分也应体现差异")

    def test_match_pair_rejects_bad_input(self):
        bad = self.client.post("/api/match/pair", json={"left": "not-an-object"})
        self.assertEqual(bad.status_code, 400)
        empty = self.client.post("/api/match/pair", json={"left": {}, "right": {}})
        self.assertEqual(empty.status_code, 400)

    def test_weights_override(self):
        body = {
            "left": {"title": "黑色耳机", "tags": ["黑色", "耳机"]},
            "right": {"title": "白色耳机", "tags": ["白色", "耳机"]},
            "weights": {"image": 0, "text": 1, "location": 0},
        }
        result = self.client.post("/api/match/text", json=body).get_json()["result"]
        self.assertEqual(result["detail"]["weights"]["image"], 0.0)
        self.assertAlmostEqual(result["detail"]["weights"]["text"], 1.0, places=6)

    # ---------------- 删除与权限 ---------------- #

    def test_delete_requires_ownership(self):
        created = self.client.post("/api/items", data={"type": "lost", "title": "匿名发布的记录"}).get_json()
        item_id = created["item"]["id"]

        # 未登录不能删除
        self.assertEqual(self.client.delete(f"/api/items/{item_id}").status_code, 401)

        # 普通用户不能删除他人记录
        self.client.post("/api/auth/login", json={"username": "demo", "password": "demo1234"})
        self.client.post("/api/items", data={"type": "lost", "title": "演示用户自己的记录"})
        others = self.client.delete(f"/api/items/{item_id}")
        self.assertEqual(others.status_code, 403)

        # 管理员可以删除
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "admin", "password": "demo1234"})
        self.assertEqual(self.client.delete(f"/api/items/{item_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/items/{item_id}").status_code, 404)

    def test_recompute_requires_admin(self):
        self.assertEqual(self.client.post("/api/match/recompute", json={}).status_code, 401)
        self.client.post("/api/auth/login", json={"username": "demo", "password": "demo1234"})
        self.assertEqual(self.client.post("/api/match/recompute", json={}).status_code, 403)
        self.client.post("/api/auth/logout")
        self.client.post("/api/auth/login", json={"username": "admin", "password": "demo1234"})
        response = self.client.post("/api/match/recompute", json={"clear": True})
        self.assertEqual(response.status_code, 200)
        self.assertIn("combinations", response.get_json()["result"])

    # ---------------- 序列化与健壮性 ---------------- #

    def test_all_json_responses_are_serializable(self):
        """所有 GET 接口的返回体都必须能被 json.dumps 处理（防止 bytes 漏出）。"""
        self.client.post(
            "/api/items",
            data={
                "type": "lost",
                "title": "带图记录",
                "tags": "黑色,测试",
                "location": "图书馆",
                "images": (io.BytesIO(make_photo((20, 20, 24))), "a.jpg"),
            },
            content_type="multipart/form-data",
        )
        endpoints = [
            "/api/stats",
            "/api/categories",
            "/api/items",
            "/api/items?type=lost",
            "/api/items/1",
            "/api/items/1/matches",
            "/api/matches/top?limit=5",
            "/api/auth/me",
        ]
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            self.assertIn(response.status_code, (200, 404), f"{endpoint} 状态码异常")
            if response.status_code == 200:
                try:
                    json.dumps(response.get_json())
                except TypeError as exc:  # pragma: no cover
                    self.fail(f"{endpoint} 返回体无法序列化：{exc}")

    def test_rejects_non_image_upload(self):
        response = self.client.post(
            "/api/items",
            data={
                "type": "lost",
                "title": "上传了坏文件",
                "images": (io.BytesIO(b"not an image at all"), "evil.jpg"),
            },
            content_type="multipart/form-data",
        )
        # 非法图片应被跳过，记录仍创建成功但没有图片
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["item"]["images"], [])

    def test_oversized_payload_is_handled(self):
        """超过上限的请求应返回 413 而不是 500。"""
        big = io.BytesIO(b"x" * (26 * 1024 * 1024))
        response = self.client.post(
            "/api/items",
            data={"type": "lost", "title": "超大请求", "images": (big, "big.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 413)

    def test_search_with_special_characters(self):
        """SQL 注入式输入不应报错，且不会返回全部数据。"""
        response = self.client.get("/api/items?q=%27%20OR%201%3D1%20--")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["total"], 0)


# --------------------------------------------------------------------------- #
# 5. 数据库层
# --------------------------------------------------------------------------- #


class TestDatabase(TempEnvMixin, unittest.TestCase):
    def test_init_is_idempotent_and_migrates(self):
        db.init_db(self.db_path)
        conn = self.connect()
        categories = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        db.init_db(self.db_path)  # 再执行一次不应报错或重复插入
        again = conn.execute("SELECT COUNT(*) AS n FROM categories").fetchone()["n"]
        conn.close()
        self.assertEqual(categories, again)
        self.assertGreaterEqual(categories, 10)

    def test_migration_adds_missing_columns(self):
        """模拟老库：删掉前景特征列后 init_db 应自动补齐。"""
        import sqlite3

        conn = db.connect(self.db_path)
        conn.executescript(db.SCHEMA)
        conn.commit()
        conn.close()

        # 用不含新列的表结构重建 item_images，模拟旧版本数据库
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE item_images")
        conn.execute(
            """
            CREATE TABLE item_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                thumb TEXT NOT NULL DEFAULT '',
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                phash_hex TEXT NOT NULL DEFAULT '',
                phash_bits BLOB,
                color_hist TEXT NOT NULL DEFAULT '[]',
                dominant TEXT NOT NULL DEFAULT '{}',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

        db.init_db(self.db_path)  # 应自动 ALTER TABLE 补齐
        conn = db.connect(self.db_path)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(item_images)").fetchall()}
        conn.close()
        for expected in ("crop_phash_hex", "crop_phash_bits", "crop_hist", "crop_dominant", "blob_ratio", "blob_box"):
            self.assertIn(expected, columns, f"迁移未补齐列 {expected}")

    def test_cascade_delete_removes_images_and_matches(self):
        conn = self.connect()
        with conn:
            item_id = conn.execute(
                "INSERT INTO items(type, title, tags, created_at, updated_at) VALUES ('lost','测试','[]','2024-01-01 00:00:00','2024-01-01 00:00:00')"
            ).lastrowid
            other_id = conn.execute(
                "INSERT INTO items(type, title, tags, created_at, updated_at) VALUES ('found','测试2','[]','2024-01-01 00:00:00','2024-01-01 00:00:00')"
            ).lastrowid
            conn.execute(
                "INSERT INTO item_images(item_id, filename, created_at) VALUES (?,?,?)",
                (item_id, "x.jpg", "2024-01-01 00:00:00"),
            )
            conn.execute(
                "INSERT INTO matches(lost_id, found_id, score, computed_at) VALUES (?,?,?,?)",
                (item_id, other_id, 0.5, "2024-01-01 00:00:00"),
            )
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM item_images").fetchone()["n"], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"], 0)
        conn.close()

    def test_query_items_pagination_and_order(self):
        conn = self.connect()
        for index in range(5):
            conn.execute(
                "INSERT INTO items(type, title, tags, view_count, created_at, updated_at) VALUES ('lost',?, '[]', ?, ?, ?)",
                (f"记录{index}", index, f"2024-01-0{index + 1} 10:00:00", f"2024-01-0{index + 1} 10:00:00"),
            )
        conn.commit()

        page1, total = db.query_items(conn, item_type="lost", limit=2, offset=0, order="newest")
        page2, _ = db.query_items(conn, item_type="lost", limit=2, offset=2, order="newest")
        hot, _ = db.query_items(conn, item_type="lost", limit=1, order="hot")
        conn.close()

        self.assertEqual(total, 5)
        self.assertEqual(len(page1), 2)
        self.assertEqual(page1[0]["title"], "记录4", "newest 应按时间倒序")
        self.assertEqual(page2[0]["title"], "记录2")
        self.assertEqual(hot[0]["title"], "记录4", "hot 应按浏览量倒序")

    def test_stats_shape(self):
        conn = self.connect()
        stats = db.stats(conn)
        conn.close()
        for key in ("lost_count", "found_count", "total_count", "image_count", "strong_match_count", "solve_rate"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
