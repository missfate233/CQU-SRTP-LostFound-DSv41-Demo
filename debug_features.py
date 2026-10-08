"""前景特征自检：``python debug_features.py [关键字]``

对（默认全部）演示图片打印前景分割与直方图统计，定位「颜色通道静默失效」这类
问题：只要出现了 ``前景直方图空`` 或 ``前景占比 0.000``，说明掩膜不可靠，
该图的颜色通道实际没有参与打分。
"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lostfound import db  # noqa: E402
from lostfound import image_features as imf  # noqa: E402

UPLOADS = os.path.join(BASE_DIR, "static", "uploads")


def main() -> int:
    keyword = sys.argv[1] if len(sys.argv) > 1 else ""
    conn = db.connect(os.path.join(BASE_DIR, "data", "lostfound.db"))
    rows = conn.execute(
        """
        SELECT i.title, i.type, img.filename, img.crop_hist, img.color_hist, img.blob_ratio
        FROM item_images img JOIN items i ON i.id = img.item_id
        ORDER BY i.id
        """
    ).fetchall()
    conn.close()

    print("=" * 108)
    print(f"{'标题':<26}{'前景占比':<9}{'掩膜像素':<9}{'前景直方图':<11}{'整图直方图':<11}{'前景主色':<8}{'裁切尺寸'}")
    print("=" * 108)
    problems = 0
    for row in rows:
        if keyword and keyword not in row["title"]:
            continue
        path = os.path.join(UPLOADS, row["filename"])
        if not os.path.isfile(path):
            continue
        img = Image.open(path).convert("RGB")
        cropped, mask, ratio, box = imf.object_blob(img)
        crop_hist = imf.object_color_histogram(img)
        full_hist = imf.color_histogram(img)
        # 与 extract_features 保持一致：主色必须在前景掩膜上统计
        working, working_mask = imf._mask_on_working(img, mask, box)
        crop_dom = imf.dominant_color(working, mask=working_mask)
        unmasked_dom = imf.dominant_color(img.crop(box))
        crop_stored = json.loads(row["crop_hist"] or "[]")
        stored_sum = float(np.sum(crop_stored)) if crop_stored else 0.0
        flag = ""
        if crop_hist.sum() <= 0 or stored_sum <= 0:
            flag = "  ← 前景直方图空"
            problems += 1
        elif ratio == 0.0:
            flag = "  ← 分割失败(退化为中心区域)"
            problems += 1
        print(
            f"{row['title'][:24]:<26}{ratio:<9.3f}{int(mask.sum()):<9}"
            f"{(crop_hist.sum() > 0 and '有效' or '空'):<11}"
            f"{(full_hist.sum() > 0 and '有效' or '空'):<11}"
            f"{crop_dom.get('label', ''):<8}(无掩膜 {unmasked_dom.get('label', '')})  {cropped.size}{flag}"
        )
    print("=" * 108)
    print(f"可疑图片数量：{problems}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
