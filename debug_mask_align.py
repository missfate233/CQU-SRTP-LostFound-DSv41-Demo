"""掩膜对齐自检：``python debug_mask_align.py <标题关键字>``

确认 ``_full_mask`` 还原到整图坐标系后覆盖的确实是物体像素：
打印掩膜内/外像素的平均颜色，以及橙色掩膜叠加图（导出到 data/debug_masks/）。
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
from lostfound import image_features as imf  # noqa: E402

UPLOADS = os.path.join(BASE_DIR, "static", "uploads")
OUT = os.path.join(BASE_DIR, "data", "debug_masks")


def main() -> int:
    keyword = sys.argv[1] if len(sys.argv) > 1 else "黑色双肩包"
    os.makedirs(OUT, exist_ok=True)
    conn = db.connect(os.path.join(BASE_DIR, "data", "lostfound.db"))
    rows = conn.execute(
        """
        SELECT i.title, img.filename FROM item_images img
        JOIN items i ON i.id = img.item_id WHERE i.title LIKE ? ORDER BY i.id
        """,
        (f"%{keyword}%",),
    ).fetchall()
    conn.close()

    for row in rows:
        path = os.path.join(UPLOADS, row["filename"])
        img = Image.open(path).convert("RGB")
        cropped, blob_mask, ratio, box = imf.object_blob(img)
        working, full = imf._mask_on_working(img, blob_mask, box)
        arr = np.asarray(working, dtype=np.float32)
        print("=" * 70)
        print(f"{row['title']}")
        print(f"  原图 {img.size} · 工作图 {working.size} · 包围盒 {box} · blob 掩膜 {blob_mask.shape}")
        if full is None:
            print("  掩膜为空")
            continue
        print(f"  工作图掩膜 {full.shape} · 覆盖像素 {int(full.sum())} · 占比 {full.mean():.4f}")
        print(f"  掩膜内平均色 {arr[full].mean(axis=0).round(0).tolist()}")
        print(f"  掩膜外平均色 {arr[~full].mean(axis=0).round(0).tolist()}")
        print(f"  掩膜内主色(应=物体颜色) {imf.dominant_color(working, mask=full).get('label')}")
        print(f"  掩膜外主色(应=背景颜色) {imf.dominant_color(working, mask=~full).get('label')}")
        vis = np.array(working, dtype=np.uint8).copy()
        vis[full] = (vis[full] * 0.4 + np.array([255, 120, 0]) * 0.6).astype(np.uint8)
        name = os.path.splitext(os.path.basename(row["filename"]))[0]
        Image.fromarray(vis).save(os.path.join(OUT, f"align_{name}.jpg"), quality=85)
        print(f"  可视化：{os.path.join(OUT, f'align_{name}.jpg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
