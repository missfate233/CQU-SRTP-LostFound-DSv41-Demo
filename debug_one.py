"""单图分割追踪：``python debug_one.py <文件名或关键字>``

打印一张图的背景估计、距离分布、阈值、掩膜比例与各阶段结果，
用于定位「前景占比为 0」这类分割失败。
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from lostfound import db  # noqa: E402
from lostfound import image_features as imf  # noqa: E402

UPLOADS = os.path.join(BASE_DIR, "static", "uploads")


def main() -> int:
    keyword = sys.argv[1] if len(sys.argv) > 1 else "粉色"
    conn = db.connect(os.path.join(BASE_DIR, "data", "lostfound.db"))
    rows = conn.execute(
        """
        SELECT i.title, i.type, img.filename FROM item_images img
        JOIN items i ON i.id = img.item_id WHERE i.title LIKE ? ORDER BY i.id
        """,
        (f"%{keyword}%",),
    ).fetchall()
    conn.close()
    if not rows:
        print("没有匹配的记录")
        return 1

    for row in rows:
        path = os.path.join(UPLOADS, row["filename"])
        print("=" * 78)
        print(f"{row['title']}  ({row['type']})")
        print(f"文件：{path}")
        img = Image.open(path).convert("RGB")
        print(f"原图尺寸：{img.size}")
        scale = min(1.0, imf.BLOB_MAX_SIDE / max(img.size))
        small = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR) if scale < 1 else img
        arr = np.asarray(small, dtype=np.float32)
        print(f"工作尺寸：{small.size}")

        bg = imf._background_color(arr)
        print(f"背景色估计（四角中位数）：{bg.tolist()}")
        dist = np.sqrt(((arr - bg) ** 2).sum(axis=2)) / 255.0
        print(
            "距离分布：min %.4f | p25 %.4f | p50 %.4f | p75 %.4f | p90 %.4f | p99 %.4f | max %.4f"
            % tuple(np.percentile(dist, [0, 25, 50, 75, 90, 99, 100]))
        )
        median = float(np.median(dist))
        mad = float(np.median(np.abs(dist - median)))
        threshold = imf._robust_threshold(dist)
        print(f"中位数 {median:.4f} · MAD {mad:.4f} · 阈值 median+3·1.4826·MAD = {threshold:.4f}")
        raw_mask = dist > threshold
        print(f"阈值前前景比例：{raw_mask.mean():.4f}（{int(raw_mask.sum())} 像素）")

        filtered = np.asarray(
            Image.fromarray((raw_mask * 255).astype(np.uint8), "L").filter(ImageFilter.MedianFilter(5)),
            dtype=np.uint8,
        ) > 127
        print(f"中值滤波后前景比例：{filtered.mean():.4f}（{int(filtered.sum())} 像素）")
        print(f"BLOB_MASK_THRESHOLD = {imf.BLOB_MASK_THRESHOLD}")

        # 画面中心区域的真实颜色，用于判断物体是否被正确圈出
        center = arr[arr.shape[0] // 2, arr.shape[1] // 2]
        print(f"中心像素颜色：{center.tolist()} · 与背景距离 {float(np.sqrt(((center - bg) ** 2).sum()) / 255):.4f}")

        # 物体真实包围盒（用「与背景色差异」的直接连通统计给出参考上界）
        loose = dist > 0.10
        if loose.any():
            ys, xs = np.where(loose)
            print(f"宽松阈值(0.10)下的区域：bbox ({int(xs.min())},{int(ys.min())})-({int(xs.max())},{int(ys.max())}) "
                  f"占比 {loose.mean():.4f}")
        cropped, mask, ratio, box = imf.object_blob(img)
        print(f"object_blob 结果：裁切 {cropped.size} · 前景占比 {ratio:.4f} · 包围盒 {box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
