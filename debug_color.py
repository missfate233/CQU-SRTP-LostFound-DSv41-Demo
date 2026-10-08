"""颜色通道追踪：``python debug_color.py``

针对「红色方块 scale=0.6 与 scale=0.9 前景直方图相交率仅 0.05」这一异常，
打印掩膜内的真实像素分布，定位颜色通道失效的原因。
"""

from __future__ import annotations

import io
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lostfound import image_features as imf  # noqa: E402


def photo(color, *, size=(400, 300), scale=0.6, brightness=1.0, background=(240, 240, 242), quality=92):
    canvas = Image.new("RGB", size, background)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box_size = int(min(size) * scale)
    left = (size[0] - box_size) // 2
    top = (size[1] - box_size) // 2
    draw.rectangle([left, top, left + box_size, top + box_size], fill=(*color, 255))
    canvas.paste(layer, (0, 0), layer)
    if brightness != 1.0:
        overlay = Image.new("RGB", size, (255, 255, 255) if brightness > 1 else (0, 0, 0))
        canvas = Image.blend(canvas, overlay, abs(brightness - 1.0) * 0.4)
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def inspect(label, blob, object_color):
    img = imf.load_image(blob)
    cropped, mask, ratio, box = imf.object_blob(img)
    working, full = imf._mask_on_working(img, mask, box)
    arr = np.asarray(working, dtype=np.float32)
    total = arr.shape[0] * arr.shape[1]
    inside = arr[full] if full is not None else np.zeros((0, 3))
    object_px = np.array(object_color, dtype=np.float32)
    if inside.size:
        dist_to_object = np.sqrt(((inside - object_px) ** 2).sum(axis=1))
        is_object = dist_to_object < 60
        purity = float(is_object.mean())
        mean_rgb = inside.mean(axis=0).round(0).tolist()
    else:
        purity, mean_rgb = 0.0, []
    hist = imf.object_color_histogram(img)
    dom = imf.dominant_color(working, mask=full)
    print(
        f"{label:<22} 掩膜占比 {ratio:<7.3f} 掩膜像素 {int(mask.sum()):<7} "
        f"掩膜内均值 {str(mean_rgb):<20} 物体纯度 {purity:<6.2f} 直方图非零 {int((hist > 0).sum()):<4} 主色 {dom.get('label')}"
    )
    return hist


def main() -> int:
    red = (200, 60, 60)
    print("=" * 118)
    hist_base = inspect("红色 基准 scale=0.6", photo(red, scale=0.6), red)
    hist_09 = inspect("红色 scale=0.9", photo(red, scale=0.9), red)
    hist_045 = inspect("红色 scale=0.45", photo(red, scale=0.45), red)
    hist_bright = inspect("红色 亮度+12%", photo(red, scale=0.6, brightness=1.12), red)
    hist_blue = inspect("蓝色 scale=0.6", photo((40, 70, 210), scale=0.6), (40, 70, 210))
    print("=" * 118)
    print(f"基准 vs scale=0.9 前景相交 {imf.histogram_similarity(hist_base, hist_09):.3f}")
    print(f"基准 vs scale=0.45 前景相交 {imf.histogram_similarity(hist_base, hist_045):.3f}")
    print(f"基准 vs 亮度+12% 前景相交 {imf.histogram_similarity(hist_base, hist_bright):.3f}")
    print(f"基准 vs 蓝色     前景相交 {imf.histogram_similarity(hist_base, hist_blue):.3f}")
    print("=" * 118)
    print("期望：同色各变体 > 0.8，异色 < 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
