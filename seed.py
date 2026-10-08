"""演示数据生成器：``python seed.py --reset``

为什么需要「合成的照片」而不是随便找几张图
------------------------------------------
匹配算法必须可复现、可量化验证。本脚本用 PIL/numpy **程序化绘制** 物品场景图：
同一件物品的「失物照片」与「拾物照片」使用不同的背景、光照、缩放比例与轻微旋转，
这正是真实场景中两个人各自拍照时的差异来源。这样既能证明
「pHash + 颜色直方图」对背景/尺度/亮度变化鲁棒，也避免引入任何版权素材。

数据设计（共 20 条：12 条失物、8 条拾物）
* 8 组「真实配对」：物品一致、描述用词不同（考验同义词与标签模糊匹配）
* 4 条「同类干扰项」：同色系/同品类但不同物品（考验区分度，不应高分成对）
* 若干纯文本记录（无图）：验证缺图时权重自动向文本迁移

用法：
    python seed.py            # 追加/更新演示数据（幂等：按标题去重）
    python seed.py --reset     # 清库重建（会删除 data/ 与已上传图片）
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
for path in (VENDOR_DIR, BASE_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from lostfound import db, matcher  # noqa: E402
from lostfound import image_features as imf  # noqa: E402

SEED = 20240501
RNG = random.Random(SEED)
DEMO_DIR = os.path.join(BASE_DIR, "data", "demo_images")


# --------------------------------------------------------------------------- #
# 合成图像
# --------------------------------------------------------------------------- #


def _gradient_background(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    ramp = np.linspace(0, 1, height).reshape(-1, 1)
    top_arr = np.array(top, dtype=np.float32).reshape(1, 3)
    bottom_arr = np.array(bottom, dtype=np.float32).reshape(1, 3)
    rows = top_arr * (1 - ramp) + bottom_arr * ramp            # (H,3)
    canvas = np.repeat(rows[:, None, :], width, axis=1)        # (H,W,3)
    return Image.fromarray(canvas.astype(np.uint8), "RGB")


def _rounded_rect(size: tuple[int, int], radius: int, color) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=color)
    return layer


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, channel * factor))) for channel in color)


def _draw_object(kind: str, color: tuple[int, int, int], scale: float) -> Image.Image:
    """按品类绘制物品（RGBA，透明底）。"""
    unit = 300
    layer = Image.new("RGBA", (unit, unit), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    dark = _shade(color, 0.72)
    light = _shade(color, 1.22)

    if kind == "backpack":
        body = _rounded_rect((180, 210), 40, (*color, 255))
        layer.alpha_composite(body, (60, 60))
        pocket = _rounded_rect((110, 80), 24, (*dark, 255))
        layer.alpha_composite(pocket, (95, 165))
        draw.arc([110, 20, 190, 120], start=200, end=340, fill=(*dark, 255), width=12)

    elif kind == "bottle":
        body = _rounded_rect((110, 190), 34, (*color, 255))
        layer.alpha_composite(body, (95, 85))
        cap = _rounded_rect((56, 34), 12, (*dark, 255))
        layer.alpha_composite(cap, (122, 52))
        band = _rounded_rect((110, 26), 10, (*light, 255))
        layer.alpha_composite(band, (95, 150))

    elif kind == "umbrella":
        draw.pieslice([45, 55, 255, 215], start=180, end=360, fill=(*color, 255))
        draw.line([150, 130, 150, 245], fill=(*dark, 255), width=9)
        draw.arc([150, 215, 195, 255], start=0, end=150, fill=(*dark, 255), width=8)

    elif kind == "book":
        cover = _rounded_rect((180, 230), 12, (*color, 255))
        layer.alpha_composite(cover, (58, 40))
        spine = _rounded_rect((22, 230), 8, (*dark, 255))
        layer.alpha_composite(spine, (58, 40))
        draw.rounded_rectangle([100, 95, 215, 135], radius=6, outline=(*light, 255), width=7)

    elif kind == "card":
        card = _rounded_rect((190, 120), 16, (*color, 255))
        layer.alpha_composite(card, (55, 95))
        draw.ellipse([80, 120, 125, 165], fill=(*light, 255))
        draw.rounded_rectangle([150, 180, 225, 192], radius=5, fill=(*light, 255))

    elif kind == "laptop":
        screen = _rounded_rect((210, 150), 14, (*color, 255))
        layer.alpha_composite(screen, (45, 55))
        inner = _rounded_rect((180, 120), 8, (*_shade(color, 0.45), 255))
        layer.alpha_composite(inner, (60, 70))
        base = _rounded_rect((240, 26), 10, (*_shade(color, 0.85), 255))
        layer.alpha_composite(base, (30, 205))

    elif kind == "glasses":
        draw.ellipse([45, 105, 145, 195], outline=(*color, 255), width=13)
        draw.ellipse([165, 105, 265, 195], outline=(*color, 255), width=13)
        draw.line([143, 140, 167, 140], fill=(*color, 255), width=11)
        draw.line([45, 130, 10, 110], fill=(*dark, 255), width=9)
        draw.line([265, 130, 295, 110], fill=(*dark, 255), width=9)

    elif kind == "phone":
        body = _rounded_rect((130, 245), 26, (*color, 255))
        layer.alpha_composite(body, (85, 30))
        screen = _rounded_rect((106, 205), 18, (*light, 255))
        layer.alpha_composite(screen, (97, 52))
        draw.ellipse([140, 240, 160, 260], fill=(*dark, 255))

    elif kind == "earbuds":
        case = _rounded_rect((170, 130), 40, (*color, 255))
        layer.alpha_composite(case, (65, 105))
        draw.line([65, 168, 235, 168], fill=(*dark, 255), width=6)
        layer.alpha_composite(_rounded_rect((40, 74), 18, (*light, 255)), (78, 22))
        layer.alpha_composite(_rounded_rect((40, 74), 18, (*light, 255)), (182, 22))

    elif kind == "keys":
        draw.ellipse([40, 110, 110, 180], outline=(*color, 255), width=15)
        draw.line([105, 148, 250, 148], fill=(*color, 255), width=13)
        draw.line([215, 148, 215, 186], fill=(*color, 255), width=11)
        draw.line([245, 148, 245, 178], fill=(*color, 255), width=11)
        tag = _rounded_rect((52, 52), 12, (*light, 255))
        layer.alpha_composite(tag, (150, 190))

    elif kind == "watch":
        draw.rounded_rectangle([120, 30, 180, 270], radius=26, fill=(*dark, 255))
        face = _rounded_rect((130, 130), 34, (*color, 255))
        layer.alpha_composite(face, (85, 85))
        draw.line([150, 150, 150, 118], fill=(*light, 255), width=7)
        draw.line([150, 150, 184, 162], fill=(*light, 255), width=7)

    elif kind == "wallet":
        body = _rounded_rect((200, 140), 22, (*color, 255))
        layer.alpha_composite(body, (50, 85))
        flap = _rounded_rect((200, 54), 18, (*dark, 255))
        layer.alpha_composite(flap, (50, 85))
        draw.ellipse([215, 145, 240, 170], fill=(*light, 255))

    elif kind == "scarf":
        draw.polygon([(60, 60), (240, 60), (215, 150), (85, 150)], fill=(*color, 255))
        draw.polygon([(120, 150), (180, 150), (170, 255), (130, 255)], fill=(*dark, 255))
        for y in range(80, 140, 18):
            draw.line([70, y, 230, y], fill=(*light, 255), width=5)

    else:  # other
        draw.ellipse([60, 60, 240, 240], fill=(*color, 255))

    if scale != 1.0:
        new_size = (max(24, int(unit * scale)), max(24, int(unit * scale)))
        layer = layer.resize(new_size, Image.LANCZOS)
    return layer


BACKGROUNDS = [
    ((232, 236, 240), (196, 204, 214)),   # 冷灰桌面
    ((246, 240, 228), (222, 208, 186)),   # 木质书桌
    ((214, 232, 220), (176, 206, 188)),   # 草地/绿垫
    ((238, 226, 232), (208, 188, 198)),   # 浅紫
    ((226, 232, 246), (190, 200, 224)),   # 浅蓝
]


def render_scene(
    kind: str,
    color: tuple[int, int, int],
    *,
    seed_offset: int,
    background_index: int,
    scale: float = 1.0,
    rotate: float = 0.0,
    brightness: float = 1.0,
    size: tuple[int, int] = (640, 480),
) -> Image.Image:
    """合成一张「手机随手拍」风格的照片。"""
    local = random.Random(SEED + seed_offset)
    top, bottom = BACKGROUNDS[background_index % len(BACKGROUNDS)]
    canvas = _gradient_background(size, top, bottom)

    obj = _draw_object(kind, color, scale)
    if rotate:
        obj = obj.rotate(rotate, resample=Image.BICUBIC, expand=True)

    max_x = max(1, size[0] - obj.width)
    max_y = max(1, size[1] - obj.height)
    pos = (local.randint(0, max_x), local.randint(0, max_y))
    canvas.paste(obj, pos, obj)

    # 轻微的高光/阴影，模拟不同光照条件
    overlay = Image.new("RGB", size, (255, 255, 255) if brightness > 1 else (0, 0, 0))
    canvas = Image.blend(canvas, overlay, abs(brightness - 1.0) * 0.35)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.4))

    # 加一点传感器噪声，防止 pHash 出现「完全一致」的假象
    arr = np.asarray(canvas, dtype=np.float32)
    noise = np.asarray(Image.effect_noise(size, 6), dtype=np.float32)
    arr = np.clip(arr + (noise[..., None] - 128.0) * 0.25, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# 场景数据：一件物品 → 两条记录（失物 / 拾物）
# --------------------------------------------------------------------------- #

COLORS = {
    "black": (38, 40, 44),
    "white": (238, 240, 242),
    "pink": (231, 128, 168),
    "blue": (58, 108, 208),
    "green": (52, 148, 92),
    "brown": (124, 84, 48),
    "silver": (186, 192, 200),
    "yellow": (236, 196, 62),
    "purple": (132, 92, 196),
    "red": (206, 66, 62),
}

SCENARIOS = [
    {
        "key": "backpack_black",
        "kind": "backpack",
        "color": "black",
        "lost": dict(
            title="黑色双肩包（内装教材与充电器）",
            category="bag",
            description="昨天下午在图书馆四楼自习区落下的黑色双肩包，里面有《数据结构》教材、一个 65W 充电器和一把伞。包侧面有个小挂饰。",
            tags=["黑色", "双肩包", "书包", "图书馆", "教材", "充电器"],
            color_name="黑色",
            location="虎溪校区图书馆 4 楼自习区",
            days_ago=3,
            contact_name="李明",
            contact="liming@cqu.edu.cn",
        ),
        "found": dict(
            title="图书馆捡到黑色背包一个",
            category="bag",
            description="在四楼靠窗自习位发现一个黑色背包，包里有教材和电源适配器，已交到图书馆一楼总服务台，请失主凭物品特征认领。",
            tags=["黑色", "背包", "书包", "图书馆", "教材", "电源适配器"],
            color_name="黑色",
            location="虎溪校区图书馆",
            days_ago=2,
            contact_name="图书馆服务台",
            contact="023-6510xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "umbrella_pink",
        "kind": "umbrella",
        "color": "pink",
        "lost": dict(
            title="粉色自动折叠雨伞",
            category="umbrella",
            description="周三晚上在二食堂门口丢了把粉色自动伞，伞柄上挂着一个小熊挂件，伞面有白色波点。",
            tags=["粉色", "雨伞", "自动伞", "食堂", "小熊挂件"],
            color_name="粉色",
            location="二食堂门口",
            days_ago=5,
            contact_name="王雨",
            contact="wy2023@cqu.edu.cn",
        ),
        "found": dict(
            title="捡到一把粉色雨伞",
            category="umbrella",
            description="在二食堂北门台阶上捡到粉色折叠伞，带小熊挂饰，已放在食堂一楼失物招领柜。",
            tags=["粉色", "伞", "折叠伞", "食堂", "挂件"],
            color_name="粉色",
            location="二食堂北门",
            days_ago=4,
            contact_name="食堂值班室",
            contact="023-6511xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "earbuds_white",
        "kind": "earbuds",
        "color": "white",
        "lost": dict(
            title="白色无线蓝牙耳机（带充电盒）",
            category="digital",
            description="周四上午在 A 区教学楼 203 上完高数后，白色无线耳机连同充电盒一起不见了，盒盖内侧有轻微划痕。",
            tags=["白色", "蓝牙耳机", "无线耳机", "充电盒", "教学楼", "A区"],
            color_name="白色",
            location="A 区教学楼 203 教室",
            days_ago=6,
            contact_name="张天",
            contact="138xxxx6621",
        ),
        "found": dict(
            title="A 区 203 教室拾到白色耳机盒",
            category="digital",
            description="课后打扫时在座位抽屉里发现白色无线耳机及耳机盒，盒盖有划痕，暂存于 A 区教学楼管理员室。",
            tags=["白色", "耳机", "无线", "充电盒", "教室", "教学楼"],
            color_name="白色",
            location="A 区教学楼 203",
            days_ago=5,
            contact_name="楼管员 陈师傅",
            contact="023-6512xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "card_blue",
        "kind": "card",
        "color": "blue",
        "lost": dict(
            title="校园一卡通丢失（姓名：刘洋）",
            category="card",
            description="周五中午在三食堂刷完卡后卡就不见了，蓝色校园卡，姓名刘洋，学号 2023xxxxxx。",
            tags=["校园卡", "一卡通", "蓝色", "学生卡", "三食堂"],
            color_name="蓝色",
            location="三食堂",
            days_ago=2,
            contact_name="刘洋",
            contact="liuyang@cqu.edu.cn",
        ),
        "found": dict(
            title="三食堂捡到蓝色校园卡一张",
            category="card",
            description="在三食堂收餐处捡到蓝色一卡通，卡面姓名刘洋，已交至三食堂前台，可凭学号领取。",
            tags=["校园卡", "一卡通", "蓝色", "三食堂", "刘洋"],
            color_name="蓝色",
            location="三食堂前台",
            days_ago=1,
            contact_name="三食堂前台",
            contact="023-6513xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "bottle_green",
        "kind": "bottle",
        "color": "green",
        "lost": dict(
            title="墨绿色保温杯（杯身有贴纸）",
            category="umbrella",
            description="周二下午在体育馆打完球忘了拿，墨绿色 500ml 保温杯，杯身贴了一张动漫贴纸，杯底有点磕痕。",
            tags=["墨绿色", "保温杯", "水杯", "体育馆", "贴纸"],
            color_name="绿色",
            location="体育馆羽毛球场",
            days_ago=4,
            contact_name="赵一鸣",
            contact="zhaoym@cqu.edu.cn",
        ),
        "found": dict(
            title="体育馆捡到绿色水杯",
            category="umbrella",
            description="羽毛球场边捡到深绿色保温水杯，瓶身有贴纸，已放在体育馆器材室。",
            tags=["绿色", "水杯", "保温杯", "体育馆", "羽毛球"],
            color_name="绿色",
            location="体育馆器材室",
            days_ago=3,
            contact_name="器材室",
            contact="023-6514xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "laptop_silver",
        "kind": "laptop",
        "color": "silver",
        "lost": dict(
            title="银色笔记本电脑（含院徽贴纸）",
            category="digital",
            description="周五晚上在实验楼 501 做实验后把银色笔记本电脑落在实验室，A 面贴有学院院徽贴纸，电源线未拔。",
            tags=["银色", "笔记本", "电脑", "实验楼", "院徽贴纸"],
            color_name="银色",
            location="实验楼 501 实验室",
            days_ago=7,
            contact_name="孙浩",
            contact="sunhao@cqu.edu.cn",
        ),
        "found": dict(
            title="实验楼 501 遗留银色笔记本电脑",
            category="digital",
            description="清理实验室时发现银色笔记本电脑一台，机盖有院徽贴纸，已锁入 501 实验准备间保险柜。",
            tags=["银色", "笔记本", "电脑", "实验楼", "贴纸"],
            color_name="银色",
            location="实验楼 501",
            days_ago=6,
            contact_name="实验中心",
            contact="023-6515xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "keys_brown",
        "kind": "keys",
        "color": "brown",
        "lost": dict(
            title="宿舍钥匙串（含门禁卡）",
            category="key",
            description="一串宿舍钥匙，挂着棕色皮质钥匙扣和一张门禁卡，可能掉在宿舍楼到操场的路上。",
            tags=["钥匙", "宿舍", "门禁卡", "钥匙扣", "操场"],
            color_name="棕色",
            location="松园七舍到操场沿线",
            days_ago=3,
            contact_name="周琳",
            contact="zhoulin@cqu.edu.cn",
        ),
        "found": dict(
            title="操场边捡到一串钥匙",
            category="key",
            description="操场东侧看台附近捡到钥匙一串，有棕色皮扣和门禁卡，现放于体育部值班室。",
            tags=["钥匙串", "钥匙扣", "门禁卡", "操场", "棕色"],
            color_name="棕色",
            location="操场东侧看台",
            days_ago=2,
            contact_name="体育部值班室",
            contact="023-6516xxxx",
        ),
        "expected": "strong",
    },
    {
        "key": "watch_purple",
        "kind": "watch",
        "color": "purple",
        "lost": dict(
            title="紫色运动手环丢失",
            category="glasses",
            description="紫黑色运动手环，表带是硅胶材质，上周日在游泳馆更衣室换衣服时摘下后忘记拿。",
            tags=["紫色", "手环", "运动手环", "游泳馆", "硅胶表带"],
            color_name="紫色",
            location="游泳馆更衣室",
            days_ago=8,
            contact_name="吴桐",
            contact="wutong@cqu.edu.cn",
        ),
        "found": dict(
            title="游泳馆拾到紫色手环",
            category="glasses",
            description="更衣室长凳上拾到紫色运动手环一只，硅胶表带，已交给游泳馆前台登记保管。",
            tags=["紫色", "手环", "运动手环", "游泳馆"],
            color_name="紫色",
            location="游泳馆前台",
            days_ago=7,
            contact_name="游泳馆前台",
            contact="023-6517xxxx",
        ),
        "expected": "strong",
    },
]

# 干扰项：品类相近但并非同一物品（用来验证算法不会「见包就配」）
DISTRACTORS = [
    dict(
        title="蓝色双肩包（计算机学院）",
        type="lost",
        category="bag",
        description="蓝色双肩包，侧袋有保温杯，主要在计算机学院楼活动。",
        tags=["蓝色", "双肩包", "书包", "计算机学院"],
        color_name="蓝色",
        location="计算机学院楼",
        days_ago=2,
        contact_name="陈同学",
        contact="chen@cqu.edu.cn",
        kind="backpack",
        color="blue",
    ),
    dict(
        title="黑色保温杯（无贴纸）",
        type="found",
        category="umbrella",
        description="教学楼 B 区教室捡到纯黑色保温杯，杯身无任何贴纸。",
        tags=["黑色", "保温杯", "水杯", "教学楼", "B区"],
        color_name="黑色",
        location="B 区教学楼 108",
        days_ago=1,
        contact_name="B 区值班室",
        contact="023-6518xxxx",
        kind="bottle",
        color="black",
    ),
    dict(
        title="白色有线耳机（3.5mm）",
        type="found",
        category="digital",
        description="图书馆一楼捡到白色有线耳机，3.5mm 圆孔接口，非蓝牙。",
        tags=["白色", "有线耳机", "耳机", "图书馆"],
        color_name="白色",
        location="图书馆一楼大厅",
        days_ago=3,
        contact_name="图书馆服务台",
        contact="023-6510xxxx",
        kind="earbuds",
        color="white",
    ),
    dict(
        title="红色雨伞（长柄）",
        type="lost",
        category="umbrella",
        description="红色长柄雨伞，木质伞柄，在宿舍楼下丢失。",
        tags=["红色", "雨伞", "长柄伞", "宿舍"],
        color_name="红色",
        location="松园三舍楼下",
        days_ago=6,
        contact_name="何同学",
        contact="he@cqu.edu.cn",
        kind="umbrella",
        color="red",
    ),
    dict(
        title="黄色雨伞（折叠，无挂件）",
        type="found",
        category="umbrella",
        description="黄色折叠雨伞，无任何挂件，在二食堂楼梯口捡到。",
        tags=["黄色", "雨伞", "折叠伞", "食堂"],
        color_name="黄色",
        location="二食堂楼梯口",
        days_ago=1,
        contact_name="食堂值班室",
        contact="023-6511xxxx",
        kind="umbrella",
        color="yellow",
    ),
]

# 纯文本记录（无图）：验证缺图时权重自动迁移到文本通道
TEXT_ONLY = [
    dict(
        type="lost",
        title="《线性代数》教材与笔记本（内有考研笔记）",
        category="book",
        description="周四上午在 B 区 305 上完线性代数后，教材和一本蓝色笔记本落在教室，笔记本里有大量手写考研笔记。",
        tags=["教材", "线性代数", "笔记本", "考研笔记", "B区"],
        color_name="蓝色",
        location="B 区教学楼 305",
        days_ago=4,
        contact_name="郑好",
        contact="zhenghao@cqu.edu.cn",
    ),
    dict(
        type="found",
        title="B 区 305 教室发现教材和笔记本",
        category="book",
        description="课后检查教室时发现《线性代数》教材一本及蓝色笔记本一本，笔记内容为考研复习要点，已交 B 区楼管。",
        tags=["教材", "线性代数", "笔记本", "考研", "B区"],
        color_name="蓝色",
        location="B 区教学楼 305",
        days_ago=3,
        contact_name="B 区楼管",
        contact="023-6518xxxx",
    ),
    dict(
        type="lost",
        title="黑色钱包（内有身份证与两张银行卡）",
        category="bag",
        description="黑色短款钱包，内有身份证（姓名 马骁）、两张银行卡和少量现金，可能在校车站或校车上丢失。",
        tags=["钱包", "黑色", "身份证", "银行卡", "校车"],
        color_name="黑色",
        location="校车站至虎溪校区校车",
        days_ago=5,
        contact_name="马骁",
        contact="maxiao@cqu.edu.cn",
    ),
]


# --------------------------------------------------------------------------- #
# 写库
# --------------------------------------------------------------------------- #


def _insert_item(conn, data: dict, item_type: str, user_id: int | None) -> int:
    row = conn.execute("SELECT id FROM categories WHERE code = ?", (data.get("category", "other"),)).fetchone()
    category_id = int(row["id"]) if row else None
    created = (datetime.now() - timedelta(days=int(data.get("days_ago", 1)), hours=RNG.randint(0, 12))).replace(microsecond=0)
    cur = conn.execute(
        """
        INSERT INTO items(type, title, category_id, description, tags, color_name, location, happened_at,
                          contact_name, contact, status, user_id, view_count, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?, 'open', ?, ?, ?, ?)
        """,
        (
            item_type,
            data["title"],
            category_id,
            data.get("description", ""),
            json.dumps(db.normalize_tags_for_seed(data.get("tags")), ensure_ascii=False),
            data.get("color_name", ""),
            data.get("location", ""),
            created.isoformat(sep=" "),
            data.get("contact_name", ""),
            data.get("contact", ""),
            user_id,
            RNG.randint(3, 260),
            created.isoformat(sep=" "),
            created.isoformat(sep=" "),
        ),
    )
    return int(cur.lastrowid)


def _attach_image(conn, item_id: int, image: Image.Image, filename: str, user_id: int | None) -> None:
    upload_dir = os.path.join(BASE_DIR, "static", "uploads")
    feats = imf.extract_features(image)
    saved = imf.save_image_variants(image, upload_dir, filename)
    db.insert_image(conn, item_id, saved=saved, feats=feats, sort_order=0)


def _ensure_tags_helper() -> None:
    """给 db 模块补一个标签清洗函数（避免 seeder 依赖 text_features 的私有实现）。"""
    if hasattr(db, "normalize_tags_for_seed"):
        return
    from lostfound.text_features import normalize_tags

    db.normalize_tags_for_seed = normalize_tags  # type: ignore[attr-defined]


def seed(reset: bool = False, verbose: bool = True) -> dict:
    _ensure_tags_helper()
    db_path = os.environ.get("LOSTFOUND_DB", os.path.join(BASE_DIR, "data", "lostfound.db"))

    if reset:
        for path in (db_path, db_path + "-wal", db_path + "-shm"):
            if os.path.isfile(path):
                os.remove(path)
        upload_dir = os.path.join(BASE_DIR, "static", "uploads")
        if os.path.isdir(upload_dir):
            for name in os.listdir(upload_dir):
                if name.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(upload_dir, name))
                    except OSError:
                        pass
        if verbose:
            print("已清空数据库与上传目录")

    db.init_db(db_path)
    conn = db.connect(db_path)
    os.makedirs(DEMO_DIR, exist_ok=True)

    def log(message: str) -> None:
        if verbose:
            print(message)

    try:
        user = db.find_user(conn, "demo")
        user_id = int(user["id"]) if user else None
        existing = {row["title"] for row in conn.execute("SELECT title FROM items").fetchall()}

        created_scenarios = 0
        for index, scenario in enumerate(SCENARIOS):
            kind, color_key = scenario["kind"], scenario["color"]
            color = COLORS[color_key]
            key = scenario["key"]

            # 失物照片：不同背景 + 明显不同的尺度 + 轻微旋转（模拟另一个人、另一台手机）
            # 尺度取 1.0~1.3 让主体占据画面 50%~80%，与真实「对着物品拍照」的构图一致；
            # 两张照片尺度差异大，正好用于验证 pHash 对缩放的鲁棒性。
            lost_img = render_scene(
                kind, color,
                seed_offset=index * 17 + 1,
                background_index=index,
                scale=1.15,
                rotate=6.0,
                brightness=0.94,
            )
            # 拾物照片：同一物品、换背景、不同尺度与更亮的曝光
            found_img = render_scene(
                kind, color,
                seed_offset=index * 17 + 9,
                background_index=index + 2,
                scale=0.85,
                rotate=-5.0,
                brightness=1.10,
            )

            for item_type, data, image in (("lost", scenario["lost"], lost_img), ("found", scenario["found"], found_img)):
                if data["title"] in existing:
                    continue
                item_id = _insert_item(conn, data, item_type, user_id)
                _attach_image(conn, item_id, image, f"{key}_{item_type}.jpg", user_id)
                existing.add(data["title"])
            created_scenarios += 1
            log(f"  · 配对场景 [{index + 1}/{len(SCENARIOS)}] {scenario['lost']['title']}")

        for index, data in enumerate(DISTRACTORS):
            if data["title"] in existing:
                continue
            image = render_scene(
                data["kind"], COLORS[data["color"]],
                seed_offset=900 + index * 13,
                background_index=index + 1,
                scale=1.0,
                rotate=3.0,
                brightness=1.0,
            )
            item_id = _insert_item(conn, data, data["type"], user_id)
            _attach_image(conn, item_id, image, f"distractor_{index}.jpg", user_id)
            existing.add(data["title"])
            log(f"  · 干扰样本 [{index + 1}/{len(DISTRACTORS)}] {data['title']}")

        for index, data in enumerate(TEXT_ONLY):
            if data["title"] in existing:
                continue
            _insert_item(conn, data, data["type"], user_id)
            existing.add(data["title"])
            log(f"  · 纯文本样本 [{index + 1}/{len(TEXT_ONLY)}] {data['title']}")

        conn.commit()

        log("正在计算全部 失物 × 拾物 组合的匹配得分 ...")
        result = matcher.recompute_all(conn)
        log(
            f"  完成：比较 {result['combinations']} 组，命中 {result['matches']} 组"
            f"（其中高可信 {result['strong']} 组），耗时 {result['seconds']}s"
            f"（{result['pairs_per_second']} 组/秒）"
        )

        stats = db.stats(conn)
        log(
            f"  数据概览：失物 {stats['lost_count']} 条 · 拾物 {stats['found_count']} 条 · "
            f"图片 {stats['image_count']} 张 · 强匹配 {stats['strong_match_count']} 组"
        )
        return {"created_scenarios": created_scenarios, "match_result": result, "stats": stats}
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成失物招领演示数据（含程序化合成的照片）")
    parser.add_argument("--reset", action="store_true", help="先清空数据库与上传目录再生成")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()
    print("=" * 62)
    print("  失物招领系统 · 演示数据生成")
    print("=" * 62)
    seed(reset=args.reset, verbose=not args.quiet)
    print("=" * 62)
    print("  完成。启动服务： python run.py")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
