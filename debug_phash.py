"""pHash 尺度/亮度不变性实测：``python debug_phash.py``

用于回答「同一物体在不同拍摄距离下，感知哈希距离到底有多大」，
并据此判断相似度映射函数的标定是否合理。
"""

from __future__ import annotations

import io
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image, ImageDraw  # noqa: E402

from lostfound import image_features as imf  # noqa: E402


def photo(color, *, size=(400, 300), scale=0.6, brightness=1.0, shape="square", quality=92, noise=False):
    canvas = Image.new("RGB", size, (240, 240, 242))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box_size = int(min(size) * scale)
    left = (size[0] - box_size) // 2
    top = (size[1] - box_size) // 2
    box = [left, top, left + box_size, top + box_size]
    if shape == "circle":
        draw.ellipse(box, fill=(*color, 255))
    else:
        draw.rectangle(box, fill=(*color, 255))
    canvas.paste(layer, (0, 0), layer)
    if brightness != 1.0:
        overlay = Image.new("RGB", size, (255, 255, 255) if brightness > 1 else (0, 0, 0))
        canvas = Image.blend(canvas, overlay, abs(brightness - 1.0) * 0.4)
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def report(title, base_blob, variants):
    base = imf.extract_features(imf.load_image(base_blob))
    print("=" * 88)
    print(title)
    print(f"{'变体':<22}{'轮廓Dice':<11}{'颜色':<10}{'整图汉明':<10}{'整图pHash':<11}{'综合':<9}{'判定'}")
    print("-" * 88)
    for label, blob in variants:
        other = imf.extract_features(imf.load_image(blob))
        score, detail = imf.image_pair_similarity(base, other)
        verdict = "同物" if score >= 0.6 else ("模糊" if score >= 0.45 else "不同")
        print(
            f"{label:<22}{detail['shape']:<11.3f}{detail['color']:<10.3f}"
            f"{detail['hamming']:<10}{detail['phash']:<11.3f}{score:<9.3f}{verdict}"
        )


def main() -> int:
    base = photo((30, 32, 38), scale=0.6)

    report(
        "深色方形物体（基准 scale=0.60）",
        base,
        [
            ("同尺度重拍", photo((30, 32, 38), scale=0.60)),
            ("缩放 0.90", photo((30, 32, 38), scale=0.90)),
            ("缩放 0.80", photo((30, 32, 38), scale=0.80)),
            ("缩放 0.75", photo((30, 32, 38), scale=0.75)),
            ("缩放 0.70", photo((30, 32, 38), scale=0.70)),
            ("缩放 0.50", photo((30, 32, 38), scale=0.50)),
            ("缩放 0.45", photo((30, 32, 38), scale=0.45)),
            ("缩放 0.40", photo((30, 32, 38), scale=0.40)),
            ("亮度 +12%", photo((30, 32, 38), scale=0.60, brightness=1.12)),
            ("亮度 -12%", photo((30, 32, 38), scale=0.60, brightness=0.88)),
            ("JPEG q=45", photo((30, 32, 38), scale=0.60, quality=45)),
            ("换成圆形（不同物体）", photo((30, 32, 38), scale=0.60, shape="circle")),
            ("换成红色（不同颜色）", photo((210, 60, 60), scale=0.60)),
        ],
    )

    base2 = photo((228, 130, 175), scale=0.8)
    report(
        "浅色（粉色）方形物体（基准 scale=0.80）",
        base2,
        [
            ("缩放 0.60", photo((228, 130, 175), scale=0.60)),
            ("缩放 0.45", photo((228, 130, 175), scale=0.45)),
            ("亮度 -12%", photo((228, 130, 175), scale=0.80, brightness=0.88)),
        ],
    )

    print("=" * 84)
    print("结论指引：若「同物体换尺度」的汉明距离普遍 > 20，说明相似度映射过严，")
    print("需要放宽 phash_similarity 的拉伸系数；若「不同物体」的距离也 < 20，说明过松。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
