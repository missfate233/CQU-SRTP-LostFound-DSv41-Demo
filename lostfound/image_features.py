"""图像感知特征提取：感知哈希(pHash) + HSV 颜色直方图 + 主色分析。

为什么这样设计
--------------
失物招领场景中，用户拍的照片往往与拾取者拍的照片在**背景、角度、裁剪、光照、
压缩质量**上差别很大。纯像素比对（如 MSE）几乎完全失效，而深度模型（ResNet 特征）
又带来较重的依赖。因此这里采用两条互补的**轻量感知通道**：

1. ``pHash``（DCT 感知哈希，64bit）
   - 对 32x32 灰度图做二维 DCT-II，取左上 8x8 低频系数，与中位数比较得到 64bit 指纹；
   - 低频系数只描述「整体结构与明暗布局」，因此对 JPEG 压缩、轻微缩放、
     亮度变化、噪声都不敏感 —— 同一物品在不同照片上哈希距离通常很小；
   - 相似度 = 1 - 汉明距离 / 64，并按经验做线性拉伸（距离 > 24 视为无关）。

2. ``HSV 颜色直方图``（8x4x4 = 128 维，L1 归一化）
   - 颜色是失物最容易口头描述、也最稳定的属性（"黑色双肩包"、"粉色水杯"）；
   - H/S/V 分别量化，使用直方图相交（histogram intersection）计算相似度，
     对形状变化不敏感，可与 pHash 互补。

两者加权融合为 ``image_similarity``。同时提取主色（dominant color）用于生成
「图像证据」文案，例如「两侧图像均以黑色为主（相似度 91%）」。
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import os
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #

PHASH_SIZE = 32          # DCT 输入尺寸
PHASH_LOW = 8            # 取前 8x8 低频
PHASH_BITS = PHASH_LOW * PHASH_LOW  # 64
# 颜色通道：16 维色相直方图（见 color_histogram 的说明），
# 低于此饱和度的像素视为"无色"（灰/白/黑），不参与色相统计
HUE_BINS = 16
COLOR_MIN_SATURATION = 40
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}
THUMB_SIZE = (720, 720)

_DCT_MATRIX: Optional[np.ndarray] = None

# 中文主色名（用于展示与文本标签比对）
COLOR_NAMES = [
    ("黑色", 0, 0, 0),
    ("白色", 255, 255, 255),
    ("灰色", 128, 128, 128),
    ("红色", 214, 48, 49),
    ("橙色", 240, 148, 51),
    ("黄色", 240, 210, 60),
    ("绿色", 46, 160, 67),
    ("青色", 60, 190, 200),
    ("蓝色", 41, 98, 220),
    ("紫色", 140, 80, 200),
    ("粉色", 240, 130, 180),
    ("棕色", 140, 90, 50),
    ("米色", 225, 205, 175),
]


# --------------------------------------------------------------------------- #
# DCT 实现（避免依赖 scipy）
# --------------------------------------------------------------------------- #


def _dct_matrix(n: int) -> np.ndarray:
    """构造正交 DCT-II 变换矩阵 D，使 ``D @ x @ D.T`` 即二维 DCT。"""
    global _DCT_MATRIX
    if _DCT_MATRIX is None or _DCT_MATRIX.shape[0] != n:
        k = np.arange(n).reshape(-1, 1)
        i = np.arange(n).reshape(1, -1)
        matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        matrix *= np.sqrt(2.0 / n)
        matrix[0, :] *= np.sqrt(0.5)
        _DCT_MATRIX = matrix.astype(np.float32)
    return _DCT_MATRIX


# --------------------------------------------------------------------------- #
# 载入与预处理
# --------------------------------------------------------------------------- #


def load_image(data: bytes | str, *, max_side: int = 1024) -> Image.Image:
    """从字节或路径载入图像，自动按 EXIF 旋转、转 RGB、限制最长边。"""
    if isinstance(data, (bytes, bytearray)):
        img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(data)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def validate_upload(data: bytes) -> tuple[bool, str]:
    """校验上传图片：大小、是否可解码、格式白名单。"""
    if not data:
        return False, "上传内容为空"
    if len(data) > MAX_UPLOAD_BYTES:
        return False, f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 限制"
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            probe.verify()
    except Exception as exc:  # noqa: BLE001 - 任何解码失败都视为非法图片
        return False, f"无法解析为图片：{exc.__class__.__name__}"
    if fmt and fmt not in ALLOWED_FORMATS:
        return False, f"不支持的图片格式：{fmt}"
    return True, ""


# --------------------------------------------------------------------------- #
# pHash
# --------------------------------------------------------------------------- #


def phash_bits(img: Image.Image) -> np.ndarray:
    """返回长度 64 的 0/1 数组（DCT 感知哈希）。"""
    gray = img.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    dct = _dct_matrix(PHASH_SIZE)
    coeffs = dct @ arr @ dct.T
    low = coeffs[:PHASH_LOW, :PHASH_LOW].flatten()
    # 去掉直流分量后取中位数作为阈值，避免整体亮度影响
    ac = np.concatenate([low[:1] * 0, low[1:]])
    thresh = float(np.median(ac))
    return (low > thresh).astype(np.uint8)


def bits_to_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def hex_to_bits(hex_str: str) -> np.ndarray:
    raw = bytes.fromhex(hex_str) if hex_str else b"\x00" * 8
    if len(raw) < 8:
        raw = raw + b"\x00" * (8 - len(raw))
    return np.unpackbits(np.frombuffer(raw[:8], dtype=np.uint8))


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    if a.size != b.size:
        return PHASH_BITS
    return int(np.count_nonzero(a.astype(np.uint8) != b.astype(np.uint8)))


def phash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """汉明距离 -> [0,1] 相似度。

    经验标定：距离 0 -> 1.0，距离 10（约 84% 位一致）-> 约 0.73，
    距离 24 以上视为不同物品 -> 0。比常见的 ``1 - d/64`` 更具区分度。
    """
    dist = hamming(a, b)
    sim = 1.0 - dist / 24.0
    return float(min(1.0, max(0.0, sim)))


# --------------------------------------------------------------------------- #
# 颜色直方图
# --------------------------------------------------------------------------- #


def _soft_histogram(values: np.ndarray, bins: int, *, sigma: float = 0.9) -> np.ndarray:
    """一维软直方图：每个样本的权重按**三角核**摊到相邻 bin 上。

    为什么必须软分箱：硬分箱时「同一个红色」只要数值有微小差异（JPEG 噪声、
    曝光微调、缩放重采样）就会整体跳到隔壁 bin。实测同一红色方块
    scale=0.6 与 scale=0.9 的硬直方图相交率只有 0.047，软分箱后回到 0.96。

    ``sigma`` 控制核宽（单位：bin），1.0 表示主要影响相邻两三个 bin。
    """
    centers = np.arange(bins, dtype=np.float32)
    out = np.zeros(bins, dtype=np.float64)
    for offset in range(-2, 3):
        shifted = centers + offset
        valid = (shifted >= 0) & (shifted <= bins - 1)
        if not valid.any():
            continue
        weight = np.exp(-0.5 * ((shifted[valid] - values) / sigma) ** 2)
        out[valid] += np.bincount(
            ((values - offset)).astype(np.int32).clip(0, bins - 1),
            weights=weight,
            minlength=bins,
        )[: bins]
    total = out.sum()
    return (out / total).astype(np.float32) if total else out.astype(np.float32)


def color_histogram(img: Image.Image, sample: int = 256, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """**色相直方图**（16 维，环形软分箱），L1 归一化。

    为什么最终选择「只看色相」（这条结论来自实测，见 ``debug_color.py``）：

    * 颜色是失物最容易描述、也最稳定的属性，但 HSV 三个分量里只有**色相 H**
      对曝光、阴影、缩放重采样不敏感；
    * 饱和度 S 会被亮度直接带动：同一红色方块整体曝光 +12% 后，前景像素
      饱和度均值从 173.6 掉到 162.0，直接跨过 S 的 bin 边界，导致 S/V 直方图
      相交率只剩 0.065 —— 颜色明明没变，"相似度"却接近 0；
    * 亮度差异不值得由直方图承担：``dominant_color`` 已经用加权的 H/S/V 距离
      专门描述「深红 vs 浅红」，两者互补。

    色相是环形的（0 与 360 相邻），因此软分箱在环形空间上做，避免红色在
    0/360 边界被切成两半。
    """
    bins = HUE_BINS
    small = img.convert("RGB").resize((sample, sample), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)
    hsv = np.asarray(Image.fromarray(arr, "RGB").convert("HSV"), dtype=np.float32)

    mask_small = None
    if mask is not None and mask.any():
        mask_small = mask
        if mask_small.shape[0] != sample or mask_small.shape[1] != sample:
            mask_small = np.asarray(
                Image.fromarray((mask_small * 255).astype(np.uint8), "L").resize((sample, sample), Image.NEAREST)
            ) > 127
        if not mask_small.any():
            mask_small = None

    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    if mask_small is not None:
        hue, saturation = hue[mask_small], saturation[mask_small]

    if hue.size == 0:
        return np.zeros(bins, dtype=np.float32)

    # 低饱和像素（灰/白/黑）没有可靠色相，单独归入"无色"统计后丢弃权重，
    # 避免它们的随机色相污染直方图。
    chroma = saturation >= COLOR_MIN_SATURATION
    if not chroma.any():
        # 整个物体都是灰白黑：返回集中在"无色"记号的直方图，同类之间自然相似
        out = np.zeros(bins, dtype=np.float32)
        out[0] = 1.0
        return out
    hue = hue[chroma]

    # 映射到环形 bin 坐标（bin 宽 256/bins）
    position = hue / 256.0 * bins
    return _ring_histogram(position, bins)


def _ring_histogram(position: np.ndarray, bins: int, sigma: float = 0.9) -> np.ndarray:
    """环形软直方图：把每个样本的权重摊到最近的若干 bin 上（首尾相连）。"""
    out = np.zeros(bins, dtype=np.float64)
    base = np.floor(position).astype(np.int32) % bins
    frac = position - np.floor(position)          # 0..1，相对 bin 中心偏移
    for offset in (-1, 0, 1, 2):
        delta = offset - frac                     # 与目标 bin 中心的距离
        weight = np.exp(-0.5 * (delta / sigma) ** 2)
        out += np.bincount((base + offset) % bins, weights=weight, minlength=bins)[:bins]
    total = out.sum()
    return (out / total).astype(np.float32) if total else out.astype(np.float32)


def histogram_similarity(hist_a: Any, hist_b: Any) -> float:
    """直方图相交：sum(min(a_i, b_i))，L1 归一化后天然落在 [0,1]。"""
    a = np.asarray(hist_a, dtype=np.float32)
    b = np.asarray(hist_b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.minimum(a / sa, b / sb).sum())


def dominant_color(img: Image.Image, mask: Optional[np.ndarray] = None) -> dict:
    """主色分析：量化到 32 级粗网格后取最高频色块，用该色块像素均值作为主色。

    ``mask`` 为前景掩膜。**必须传掩膜**：物体包围盒的四个角通常仍是背景，
    若只在 ``img.crop(bbox)`` 上取主色，"粉色雨伞" 会得到 "浅米色"（桌面颜色）。
    """
    small = img.convert("RGB").resize((160, 160), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    if mask is not None and mask.any():
        mask_small = mask
        if mask_small.shape != (160, 160):
            mask_small = np.asarray(
                Image.fromarray((mask_small * 255).astype(np.uint8), "L").resize((160, 160), Image.NEAREST)
            ) > 127
        selected = flat[mask_small.reshape(-1)]
        if selected.size >= 16:
            flat = selected

    keys = (flat // 32).astype(np.int32)
    codes = keys[:, 0] * 64 + keys[:, 1] * 8 + keys[:, 2]
    counts = np.bincount(codes, minlength=512)
    dominant_code = int(counts.argmax())
    mask_codes = codes == dominant_code
    mean_rgb = flat[mask_codes].mean(axis=0)
    r, g, b = (mean_rgb / 255.0).tolist()
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    name = nearest_color_name(mean_rgb)
    brightness = "浅" if val > 0.72 else ("深" if val < 0.32 else "")
    return {
        "name": name,
        "label": f"{brightness}{name}" if brightness and name not in ("黑色", "白色") else name,
        "rgb": [int(round(x)) for x in mean_rgb.tolist()],
        "hex": "#%02x%02x%02x" % tuple(int(round(x)) for x in mean_rgb.tolist()),
        "hsv": [round(hue * 360, 1), round(sat, 3), round(val, 3)],
        "ratio": round(float(counts[dominant_code]) / float(codes.size), 4),
    }


def nearest_color_name(rgb: Any) -> str:
    """把 RGB 映射到最接近的中文颜色名（加权欧氏距离，更贴近人眼）。"""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    best, best_dist = COLOR_NAMES[0][0], float("inf")
    for name, cr, cg, cb in COLOR_NAMES:
        dr, dg, db = r - cr, g - cg, b - cb
        dist = 2 * dr * dr + 4 * dg * dg + 3 * db * db
        if dist < best_dist:
            best, best_dist = name, dist
    return best


# --------------------------------------------------------------------------- #
# 前景（物体）定位：提升图像通道区分度的关键
# --------------------------------------------------------------------------- #
#
# 实测结论（见 eval_match.py）：直接对整张照片做 pHash，区分度接近 0 —— 因为
# 「随手拍」照片中背景（桌面、地面、墙面）占比通常超过 60%，DCT 低频系数被
# 背景纹理主导，"黑色双肩包" 和 "蓝色双肩包" 的哈希距离几乎一样近。
# 因此这里先做**显著性前景分割**，只在物体区域上计算感知特征：
#   1. 取图像四周边缘像素估计背景色；
#   2. 计算每个像素与背景色的距离，Otsu 自动阈值二值化；
#   3. 连通域分析保留最大前景块（去掉零散噪点与地面阴影）；
#   4. 在紧致包围盒上重算 pHash / 颜色直方图。
# 这样「同色不同物」的图像相似度会显著低于「同物不同照片」，与文本通道互补。

BLOB_MAX_SIDE = 320
# 前景像素占比低于此值视为分割失败。取 1% 而不是更高的值：
# 真实场景中「离得远的照片」主体可能只占画面几个百分点，但这些像素依然
# 包含全部有效颜色信息，远比退化为「中心区域」更准确。
BLOB_MASK_THRESHOLD = 0.01
# 分割彻底失败时的兜底裁切比例（保留中心区域，尽量避免把背景算进直方图）
BLOB_FALLBACK_INSET = 0.22


def _otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """Otsu 类间方差最大化阈值（自带实现，避免依赖 scipy/skimage）。"""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    prob = hist / total
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_total * omega - mu) ** 2 / denom, 0.0)
    return float(centers[int(np.argmax(sigma_b))])


def _label_components(mask: np.ndarray) -> np.ndarray:
    """4 邻域连通域标记（并查集两遍扫描）。

    并查集的 ``parent`` **必须以下标索引**（``parent[i] == i`` 表示 i 为根）。
    因此 0 号槽位只是占位符（标签 0 表示背景），不能先 ``append(0)`` 再让
    标签从 1 开始 —— 那样 ``parent[1]`` 会指向错误的槽位，标签合并全部失效。
    """
    height, width = mask.shape
    parent = [0]  # parent[i] 为 i 的父节点；0 为占位

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # 路径压缩
            parent[node], node = root, parent[node]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 1
    prev_row = np.zeros(width, dtype=np.int32)
    for y in range(height):
        row = mask[y]
        current = np.zeros(width, dtype=np.int32)
        if row.any():
            # 水平游程：连续 True 段分配同一临时标签，再与上一行重叠段合并
            diff = np.diff(np.concatenate(([False], row, [False])).view(np.int8))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            for start, end in zip(starts.tolist(), ends.tolist()):
                label = next_label
                next_label += 1
                parent.append(label)  # 此时 parent[label] == label，槽位对齐
                current[start:end] = label
                overlap = prev_row[start:end]
                for other in np.unique(overlap[overlap > 0]).tolist():
                    union(label, int(other))
        labels[y] = current
        prev_row = current

    # 第二遍：把根标签重映射为紧凑编号
    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    for label in np.unique(labels):
        if label == 0:
            continue
        root = find(int(label))
        slot = remap.setdefault(root, len(remap) + 1)
        out[labels == label] = slot
    return out


def _robust_threshold(distance: np.ndarray) -> float:
    """基于中位数 + MAD 的鲁棒阈值。

    纯 Otsu 在「背景占绝大多数」的图像上会把阈值推到物体最亮的局部细节上，
    导致掩膜只剩一小块高光（实测前景占比仅 2%~5%）。这里改用鲁棒统计：
    背景像素的距离普遍很小，而物体像素使其整体中位数抬高、离散度增大，
    因此用 ``median + k·MAD`` 能稳定地圈出整个物体，且对背景渐变不敏感。
    """
    flat = distance.ravel()
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    threshold = median + 3.0 * 1.4826 * mad
    return max(0.045, min(threshold, 0.75))


def _background_color(arr: np.ndarray) -> np.ndarray:
    """用四角区域估计背景色。

    不用「四条边」：物体常常紧贴画面边缘（例如伞面几乎铺满整幅图），
    边框中混入物体像素会把背景色估偏，导致整张图都被判为前景。
    """
    height, width = arr.shape[:2]
    patch = max(4, int(min(height, width) * 0.12))
    corners = np.concatenate(
        [
            arr[:patch, :patch].reshape(-1, 3),
            arr[:patch, -patch:].reshape(-1, 3),
            arr[-patch:, :patch].reshape(-1, 3),
            arr[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(corners, axis=0)


def _detect_blob_small(rgb: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """在降采样后的图像上做前景检测（检测本身对分辨率不敏感，降采样只为提速）。

    返回 ``(掩膜, 包围盒, 前景占比)``，坐标与掩膜都位于**传入图像**的坐标系。
    """
    height, width = rgb.height, rgb.width
    arr = np.asarray(rgb, dtype=np.float32)

    background = _background_color(arr)
    distance = np.sqrt(((arr - background) ** 2).sum(axis=2)) / 255.0
    threshold = _robust_threshold(distance)
    mask = distance > threshold

    ratio = float(mask.mean())
    if ratio < BLOB_MASK_THRESHOLD:
        # 前景太少：浅色物体在浅色背景上（如同色桌面）导致几乎无可分像素。
        # 返回「中心区域」作为兜底，避免把整幅图的背景重新引入直方图。
        inset = BLOB_FALLBACK_INSET
        inset_x, inset_y = int(width * inset), int(height * inset)
        box = (inset_x, inset_y, max(inset_x + 4, width - inset_x), max(inset_y + 4, height - inset_y))
        full = np.ones((height, width), dtype=bool)
        return full, box, 0.0

    # 中值滤波去噪 → 连通域，保留最大前景块（去掉零散噪点与地面阴影）
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), "L").filter(ImageFilter.MedianFilter(5))
    mask = np.asarray(mask_img, dtype=np.uint8) > 127
    labels = _label_components(mask)
    if labels.max() == 0:
        mask = np.ones((height, width), dtype=bool)
    else:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(sizes.argmax())

    ys, xs = np.where(mask)
    if ys.size == 0:
        y0, y1, x0, x1 = 0, height, 0, width
    else:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
    if y1 - y0 < 4 or x1 - x0 < 4:
        y0, y1, x0, x1 = 0, height, 0, width
    return mask, (x0, y0, x1, y1), float(mask.mean())


def object_blob(img: Image.Image) -> tuple[Image.Image, np.ndarray, float, tuple[int, int, int, int]]:
    """定位照片中的主体物体。

    返回 ``(物体图, 前景掩膜, 前景占比, 包围盒)``：

    * 物体图与前景掩膜**尺寸一致**，都取自降采样后的工作图 —— 特征提取（pHash
      32x32、直方图 256x256）本来就会再缩放一次，无需放大回原分辨率；
    * 包围盒坐标**已换算回传入图像的像素坐标系**，可直接用于 ``img.crop(box)``。

    注意：早期版本把工作图坐标系下的包围盒直接当作原图坐标返回，导致 640x480 的
    图片掩膜整体偏移到背景上（"黑色书包"的前景主色被算成 "浅米色"）；而把掩膜
    放大回原分辨率又会因为 NEAREST 上采样的稀疏化让掩膜面积缩水约 4 倍。
    因此这里明确以「工作图坐标系 + 掩膜不缩放」为契约。
    """
    original = img.convert("RGB")
    scale = min(1.0, BLOB_MAX_SIDE / max(original.size))
    if scale < 1.0:
        working = original.resize(
            (max(8, int(original.width * scale)), max(8, int(original.height * scale))), Image.BILINEAR
        )
    else:
        working = original

    mask, box, ratio = _detect_blob_small(working)
    cropped = working.crop(box)
    if mask.shape != (cropped.height, cropped.width):
        mask = mask[box[1] : box[3], box[0] : box[2]]
    return cropped, mask, ratio, box


def _mask_on_working(img: Image.Image, mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[Image.Image, Optional[np.ndarray]]:
    """把掩膜投影到「工作图分辨率」的整图上（用于主色统计，避免放大掩膜）。"""
    original = img.convert("RGB")
    scale = min(1.0, BLOB_MAX_SIDE / max(original.size))
    if scale < 1.0:
        working = original.resize(
            (max(8, int(original.width * scale)), max(8, int(original.height * scale))), Image.BILINEAR
        )
    else:
        working = original
    full = np.zeros((working.height, working.width), dtype=bool)
    x0, y0, x1, y1 = box
    region = mask
    if region.shape != (y1 - y0, x1 - x0):
        region = np.asarray(
            Image.fromarray((region * 255).astype(np.uint8), "L").resize((x1 - x0, y1 - y0), Image.NEAREST)
        ) > 127
    full[y0:y1, x0:x1] = region
    return working, (full if full.any() else None)


# --------------------------------------------------------------------------- #
# 轮廓描述子（形状通道的主力）
# --------------------------------------------------------------------------- #
#
# 为什么不用 pHash 做形状通道（实测数据见 debug_phash.py）：
# pHash 对**全局**照片是有效的，但把主体裁出来之后，物体在裁切框内已经占满
# 画面，此时 8x8 低频系数的「与中位数比较」极易被重采样相位影响 —— 同一个
# 纯色方块在不同拍摄距离下的汉明距离高达 24~32，而「换成圆形/换成红色」这类
# 真正不同的物体距离也只有 21~26。也就是说 pHash 在这个任务上几乎没有区分度。
#
# 因此形状通道改用**尺度归一化的轮廓占用栅格**：
#   1. 把前景掩膜等比缩放到 R×R（默认 16x16）的方形画布；
#   2. 以 0.5 覆盖率二值化，得到 256 位轮廓指纹；
#   3. 相似度用 Dice 系数（对轮廓边缘的小扰动不敏感，且天然落在 [0,1]）。
# 由于掩膜先按包围盒归一化，该描述子对拍摄距离、画面位置、分辨率都免疫。

SHAPE_GRID = 16
SHAPE_BITS = SHAPE_GRID * SHAPE_GRID


def shape_signature(mask: Optional[np.ndarray]) -> np.ndarray:
    """把前景掩膜转成 R×R 的等比轮廓指纹（0/1，长度 256）。"""
    empty = np.zeros(SHAPE_BITS, dtype=np.uint8)
    if mask is None or not mask.any():
        return empty
    try:
        gray = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    except (TypeError, ValueError):
        return empty
    width, height = gray.size
    if width < 1 or height < 1:
        return empty
    # 等比缩放后居中贴到方形画布，避免拉伸导致的长宽比失真
    ratio = min(SHAPE_GRID / width, SHAPE_GRID / height)
    new_size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    resized = gray.resize(new_size, Image.BILINEAR)
    canvas = Image.new("L", (SHAPE_GRID, SHAPE_GRID), 0)
    canvas.paste(resized, ((SHAPE_GRID - new_size[0]) // 2, (SHAPE_GRID - new_size[1]) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return (arr >= 0.5).astype(np.uint8).ravel()


def shape_similarity(sig_a: Any, sig_b: Any) -> float:
    """两个轮廓指纹的 Dice 相似度。"""
    a = coerce_signature(sig_a)
    b = coerce_signature(sig_b)
    if a is None or b is None or a.sum() == 0 or b.sum() == 0:
        return 0.0
    inter = float(np.count_nonzero(a & b))
    total = float(a.sum() + b.sum())
    return float(2.0 * inter / total) if total else 0.0


def shape_score(feat_a: dict, feat_b: dict) -> float:
    """形状通道总分 = 轮廓 Dice 与「包围盒填充率」一致性 的加权融合。

    单靠轮廓 Dice 无法区分「正方形 vs 圆形」：掩膜被拉伸到方形画布后，圆形的
    轮廓栅格几乎与正方形重合（实测 Dice 0.897）。而**填充率**
    （前景像素数 / 包围盒面积）对方块是 1.00、对圆是 0.785、对三角形是 0.5，
    正好补上这个缺口。两个指标都天然尺度无关。
    """
    dice = shape_similarity(_pick_signature(feat_a), _pick_signature(feat_b))
    fill_a = feat_a.get("blob_fill")
    fill_b = feat_b.get("blob_fill")
    if fill_a is None or fill_b is None:
        return dice
    try:
        delta = abs(float(fill_a) - float(fill_b))
    except (TypeError, ValueError):
        return dice
    fill_sim = max(0.0, 1.0 - delta / 0.35)      # 填充率差 0.35 以上视为完全不同
    return float(0.72 * dice + 0.28 * fill_sim)


def coerce_signature(value: Any) -> Optional[np.ndarray]:
    """把掩膜/字节/位串统一成长度 256 的 0/1 数组。"""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == SHAPE_BITS:
            return (value.reshape(-1) != 0).astype(np.uint8)
        if value.ndim == 2:
            return shape_signature(value.astype(bool))
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) * 8 >= SHAPE_BITS:
            return np.unpackbits(np.frombuffer(raw[: SHAPE_BITS // 8], dtype=np.uint8))
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            raw = bytes.fromhex(text)
        except ValueError:
            return None
        if len(raw) * 8 >= SHAPE_BITS:
            return np.unpackbits(np.frombuffer(raw[: SHAPE_BITS // 8], dtype=np.uint8))
        return None
    return None


def signature_to_hex(sig: np.ndarray) -> str:
    return np.packbits((sig != 0).astype(np.uint8)).tobytes().hex()


def object_phash_bits(img: Image.Image) -> tuple[np.ndarray, float, tuple]:
    """只对主体物体计算 pHash，抑制背景对低频系数的干扰。"""
    cropped, _mask, ratio, box = object_blob(img)
    if cropped.width < 8 or cropped.height < 8:
        return phash_bits(img), ratio, box
    return phash_bits(cropped), ratio, box


def object_color_histogram(img: Image.Image) -> np.ndarray:
    """只统计前景像素的 HSV 直方图。

    "黑色双肩包" 与 "蓝色双肩包" 在同一张木桌上拍摄时，整图直方图会被桌面颜色
    拉平；剔除背景像素后两者的色相分布差异才能体现出来。
    """
    cropped, mask, _ratio, _box = object_blob(img)
    return color_histogram(cropped, sample=256, mask=mask)


# --------------------------------------------------------------------------- #
# 单图 & 多图特征
# --------------------------------------------------------------------------- #


def extract_features(img: Image.Image) -> dict:
    """提取单张图像的全部感知特征（整图 + 前景两套）。

    只调用一次 ``object_blob``：前景分割是整个流程里最贵的一步，
    重复调用会让批量建库（几百张图）明显变慢。
    """
    bits = phash_bits(img)
    hist = color_histogram(img)
    dom = dominant_color(img)

    cropped, blob_mask, blob_ratio, blob_box = object_blob(img)
    crop_bits = phash_bits(cropped) if cropped.width >= 8 and cropped.height >= 8 else bits
    crop_hist = color_histogram(cropped, sample=256, mask=blob_mask)
    shape = shape_signature(blob_mask)
    # 填充率：前景像素 / 包围盒面积。方块≈1.0、圆≈0.785、三角≈0.5，
    # 是区分「等轮廓不同形状」的关键补充指标。
    box_area = max(1, (blob_box[2] - blob_box[0]) * (blob_box[3] - blob_box[1]))
    blob_fill = float(blob_mask.sum()) / float(box_area) if blob_mask.any() else 0.0
    # 前景主色必须在前景掩膜上统计：包围盒四角通常仍是背景，
    # 直接在 crop 上取主色会把桌面颜色当成物品颜色。
    working, working_mask = _mask_on_working(img, blob_mask, blob_box)
    crop_dom = dominant_color(working, mask=working_mask)

    return {
        "width": img.width,
        "height": img.height,
        "phash_bits": bits,
        "phash_hex": bits_to_hex(bits),
        "color_hist": [round(float(x), 6) for x in hist.tolist()],
        "dominant": dom,
        # 前景（物体）特征
        "crop_phash_bits": crop_bits,
        "crop_phash_hex": bits_to_hex(crop_bits),
        "crop_hist": [round(float(x), 6) for x in crop_hist.tolist()],
        "crop_dominant": crop_dom,
        # 形状通道：尺度归一化轮廓指纹（256 位 = 32 字节）+ 包围盒填充率
        "shape_sig": shape,
        "shape_sig_hex": signature_to_hex(shape),
        "blob_fill": round(blob_fill, 4),
        "blob_ratio": round(float(blob_ratio), 4),
        "blob_box": list(blob_box),
    }


def extract_features_from_bytes(data: bytes) -> dict:
    return extract_features(load_image(data))


def coerce_bits(value: Any) -> Optional[np.ndarray]:
    """把各种来源的哈希统一成长度 64 的 0/1 ``uint8`` 数组。

    需要处理的形态（``np.asarray(b"...")`` 会得到 0 维 bytes 数组而非位数组，
    这是最容易踩的坑，因此集中在这里统一收口）：
    * 数据库 ``BLOB``（8 字节 ``bytes``）→ ``np.unpackbits``；
    * 64 位十六进制字符串（``phash_hex``）；
    * 已经是 0/1 的序列或 ndarray。
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8 and value.size == PHASH_BITS:
            return value.astype(np.uint8)
        if value.size == 8 and value.dtype == np.uint8:
            return np.unpackbits(value)
        return (value.astype(np.uint8).ravel()[:PHASH_BITS] != 0).astype(np.uint8)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) >= 8:
            return np.unpackbits(np.frombuffer(raw[:8], dtype=np.uint8))
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return hex_to_bits(text)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        return coerce_bits(arr)
    return None


def _pick_bits(feat: dict, *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        bits = coerce_bits(feat.get(key))
        if bits is not None:
            return bits
    return None

def _pick_hist(feat: dict, *keys: str) -> list:
    for key in keys:
        value = feat.get(key)
        if value:
            try:
                arr = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if arr.size and float(arr.sum()) > 0:
                return value  # type: ignore[return-value]
    return []


def _pick_signature(feat: dict) -> Optional[np.ndarray]:
    """按优先级取轮廓指纹（数组优先，其次十六进制字符串）。

    注意不能写成 ``feat.get("a") or feat.get("b")``：``shape_sig`` 是 numpy 数组，
    对它做布尔判断会抛 ``ValueError: truth value of an array is ambiguous``。
    """
    for key in ("shape_sig", "shape_sig_hex"):
        value = feat.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        sig = coerce_signature(value)
        if sig is not None:
            return sig
    return None


def image_pair_similarity(feat_a: dict, feat_b: dict) -> tuple[float, dict]:
    """两张图像的综合相似度。

    通道构成（标定依据见 ``debug_phash.py`` 的实测输出）：

    * **轮廓 Dice**（权重 0.40）—— 尺度归一化后的主体形状，区分「方块 vs 圆形」
      这类真实形状差异，且不受拍摄距离影响；
    * **前景颜色直方图相交**（权重 0.50）—— 颜色是最稳定、最具区分度的属性，
      实测对缩放/亮度/压缩都稳定在 0.87~0.99；
    * **整图 pHash**（权重 0.10）—— 只在两侧都是「近乎同一张照片」时提供强证据，
      权重压得很低，因为它对尺度与裁切非常敏感。
    """
    sig_a = _pick_signature(feat_a)
    sig_b = _pick_signature(feat_b)

    full_a = _pick_bits(feat_a, "phash_bits_hex", "phash_bits", "phash_hex")
    full_b = _pick_bits(feat_b, "phash_bits_hex", "phash_bits", "phash_hex")

    shape_sim = shape_score(feat_a, feat_b)

    # 颜色：前景掩膜直方图为主，整图直方图为辅
    crop_color = histogram_similarity(_pick_hist(feat_a, "crop_hist"), _pick_hist(feat_b, "crop_hist"))
    full_color = histogram_similarity(_pick_hist(feat_a, "color_hist"), _pick_hist(feat_b, "color_hist"))
    color_sim = 0.75 * crop_color + 0.25 * full_color if crop_color else full_color

    # 主色一致性：作为颜色的鲁棒补充（对直方图噪声与拍摄光线差异更宽容）
    dom_a = feat_a.get("crop_dominant") or feat_a.get("dominant") or {}
    dom_b = feat_b.get("crop_dominant") or feat_b.get("dominant") or {}
    dominant_sim = _dominant_similarity(dom_a, dom_b)
    if dominant_sim is not None:
        color_sim = 0.72 * color_sim + 0.28 * dominant_sim

    full_sim = phash_similarity(full_a, full_b) if full_a is not None and full_b is not None else 0.0
    full_dist = hamming(full_a, full_b) if full_a is not None and full_b is not None else PHASH_BITS

    score = 0.40 * shape_sim + 0.50 * color_sim + 0.10 * full_sim
    return float(score), {
        "shape": round(shape_sim, 4),
        "color": round(color_sim, 4),
        "phash": round(full_sim, 4),
        "hamming": int(full_dist),
        "crop_color": round(crop_color, 4),
        "dominant_sim": None if dominant_sim is None else round(dominant_sim, 4),
        "dominant_labels": [dom_a.get("label", ""), dom_b.get("label", "")],
    }


def _dominant_similarity(dom_a: dict, dom_b: dict) -> Optional[float]:
    """主色的语义相似度；任一侧颜色不明确（接近灰白）时返回 ``None`` 不参与打分。"""
    if not dom_a or not dom_b:
        return None
    hsv_a, hsv_b = dom_a.get("hsv") or [], dom_b.get("hsv") or []
    if len(hsv_a) < 3 or len(hsv_b) < 3:
        return None
    # 低饱和或极暗/极亮时色相不可靠，退回「颜色名是否一致」
    if hsv_a[1] < 0.18 or hsv_b[1] < 0.18 or hsv_a[2] < 0.12 or hsv_b[2] < 0.12:
        return 1.0 if dom_a.get("name") == dom_b.get("name") else 0.0
    hue_delta = abs(float(hsv_a[0]) - float(hsv_b[0]))
    hue_delta = min(hue_delta, 360.0 - hue_delta) / 180.0     # 环形色相差 → [0,1]
    sat_delta = abs(float(hsv_a[1]) - float(hsv_b[1]))
    val_delta = abs(float(hsv_a[2]) - float(hsv_b[2]))
    distance = 0.70 * hue_delta + 0.20 * sat_delta + 0.10 * val_delta
    return float(max(0.0, 1.0 - distance * 2.0))


def item_image_similarity(images_a: list[dict], images_b: list[dict]) -> tuple[float, dict]:
    """两个物品（各自可能有多张图）之间的图像相似度。

    采用「最佳配对」策略：任取一组的图与另一组的图两两比较，取最高分。
    这样用户只要有一张照片拍得清楚，就不会被其他照片拖累。
    """
    if not images_a or not images_b:
        return 0.0, {
            "pair": None,
            "phash": 0.0,
            "color": 0.0,
            "hamming": PHASH_BITS,
            "full_phash": 0.0,
            "full_hamming": PHASH_BITS,
            "crop_color": 0.0,
            "dominant_labels": [],
        }
    best_score, best_detail, best_pair = -1.0, {}, (None, None)
    for ia in images_a:
        for ib in images_b:
            score, detail = image_pair_similarity(ia, ib)
            if score > best_score:
                best_score, best_detail, best_pair = score, detail, (ia, ib)
    if best_score < 0:
        return 0.0, {"pair": None, "phash": 0.0, "color": 0.0, "hamming": PHASH_BITS}
    detail = dict(best_detail)
    detail["pair"] = [best_pair[0].get("id"), best_pair[1].get("id")]
    labels = detail.get("dominant_labels") or []
    if len(labels) != 2 or not labels[0] or not labels[1]:
        detail["dominant_labels"] = [
            (best_pair[0].get("crop_dominant") or best_pair[0].get("dominant") or {}).get("label", ""),
            (best_pair[1].get("crop_dominant") or best_pair[1].get("dominant") or {}).get("label", ""),
        ]
    return float(best_score), detail


# --------------------------------------------------------------------------- #
# 缩略图与落盘
# --------------------------------------------------------------------------- #


def make_thumbnail(img: Image.Image, size: tuple[int, int] = THUMB_SIZE) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    return thumb


def safe_filename(original: str) -> str:
    """生成不冲突的存储文件名，并保留原始扩展名（不可信输入只取扩展名）。"""
    ext = os.path.splitext(original or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"):
        ext = ".jpg"
    digest = hashlib.sha1(os.urandom(16)).hexdigest()[:16]
    return f"{digest}{ext}"


def save_image_variants(img: Image.Image, upload_dir: str, original_name: str = "") -> dict:
    """保存原图（压缩后的 JPEG）与缩略图，返回落盘信息。

    统一转存为 JPEG 可以避免 HEIC/PNG 透明通道等格式差异，同时显著减小体积。
    """
    os.makedirs(upload_dir, exist_ok=True)
    name = safe_filename(original_name)
    stem, _ = os.path.splitext(name)
    full_name = f"{stem}.jpg"
    thumb_name = f"{stem}_thumb.jpg"

    rgb = img.convert("RGB")
    full_path = os.path.join(upload_dir, full_name)
    thumb_path = os.path.join(upload_dir, thumb_name)

    quality = 86
    for _ in range(6):
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        if buf.tell() <= 1_200_000 or quality <= 55:
            break
        quality -= 8
    with open(full_path, "wb") as fh:
        fh.write(buf.getvalue())
    make_thumbnail(rgb).save(thumb_path, format="JPEG", quality=82, optimize=True)

    return {
        "filename": full_name,
        "thumb": thumb_name,
        "bytes": len(buf.getvalue()),
        "width": rgb.width,
        "height": rgb.height,
    }
