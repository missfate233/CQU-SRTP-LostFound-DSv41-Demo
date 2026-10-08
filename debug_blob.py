"""分割诊断：``python debug_blob.py``

打印每张演示图片的前景分割结果，并导出掩膜可视化到 ``data/debug_masks/``，
用于确认「物体定位」是否可靠（前景占比过低/过高都说明阈值需要调整）。
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lostfound import db  # noqa: E402
from lostfound.image_features import dominant_color, object_blob  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "data", "debug_masks")
UPLOADS = os.path.join(BASE_DIR, "static", "uploads")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = db.connect(os.path.join(BASE_DIR, "data", "lostfound.db"))
    rows = conn.execute(
        """
        SELECT i.title, i.type, img.filename, img.blob_ratio, img.blob_box, img.dominant, img.crop_dominant
        FROM item_images img JOIN items i ON i.id = img.item_id
        ORDER BY i.id
        """
    ).fetchall()
    conn.close()

    print("=" * 100)
    print(f"{'标题':<30}{'前景占比':<10}{'包围盒':<26}{'整图主色':<10}{'前景主色'}")
    print("=" * 100)
    for row in rows:
        import json

        box = json.loads(row["blob_box"] or "[]")
        dom = json.loads(row["dominant"] or "{}").get("label", "")
        cdom = json.loads(row["crop_dominant"] or "{}").get("label", "")
        print(f"{row['title'][:28]:<30}{row['blob_ratio']:<10.3f}{str(box):<26}{dom:<10}{cdom}")

        path = os.path.join(UPLOADS, row["filename"])
        if not os.path.isfile(path):
            continue
        img = Image.open(path).convert("RGB")
        cropped, mask, ratio, blob_box = object_blob(img)
        arr_small = np.array(img, dtype=np.uint8).copy()  # np.asarray 返回只读视图，必须复制
        mask_full = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8), "L").resize((img.width, img.height), Image.NEAREST)
        ) > 127
        arr_small[mask_full] = (arr_small[mask_full] * 0.45 + np.array([255, 60, 60]) * 0.55).astype(np.uint8)
        Image.fromarray(arr_small).save(
            os.path.join(OUT_DIR, f"mask_{os.path.splitext(row['filename'])[0]}.jpg"), quality=80
        )
        cropped.save(os.path.join(OUT_DIR, f"crop_{os.path.splitext(row['filename'])[0]}.jpg"), quality=80)
        print(f"    裁切尺寸 {cropped.size} · 前景占比 {ratio:.3f} · 主色 {dominant_color(cropped).get('label')}")

    print("=" * 100)
    print(f"掩膜可视化已导出到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
