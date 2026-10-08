"""端到端验收脚本：``python smoke_test.py [base_url]``

对**正在运行的服务**做黑盒验收：真实 HTTP 请求、真实图片上传、真实会话 cookie，
覆盖页面渲染、REST API、鉴权、发布→匹配闭环与错误处理。

与 ``tests/test_lostfound.py`` 的分工：
* 单元测试用 Flask test_client，快、隔离、可断言内部结构；
* 本脚本打真实端口，验证 WSGI 之外的东西（静态资源、上传落盘、HTTP 状态码、
  多进程/线程下的行为差异），适合部署后自查。
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(BASE_DIR, "vendor"), BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image, ImageDraw  # noqa: E402

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000").rstrip("/")

PASS, FAIL = [], []


def check(label: str, condition: bool, detail: str = "") -> bool:
    (PASS if condition else FAIL).append(label)
    mark = "✓" if condition else "✗"
    print(f"  {mark} {label}" + (f"  —— {detail}" if detail and not condition else ""))
    return condition


class Session:
    """极简 HTTP 客户端，手动维护 cookie（标准库没有会话支持）。"""

    def __init__(self) -> None:
        self.cookies: dict[str, str] = {}

    def _request(self, method: str, path: str, *, data: bytes = None, headers: dict = None):
        # 请求行必须是 ASCII：中文查询参数需要先百分号编码
        url = BASE_URL + urllib.parse.quote(path, safe="/?&=%#[]@!$'()*+,;:~")
        request = urllib.request.Request(url, data=data, method=method)
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        if self.cookies:
            request.add_header("Cookie", "; ".join(f"{k}={v}" for k, v in self.cookies.items()))
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                for header in response.headers.get_all("Set-Cookie") or []:
                    pair = header.split(";", 1)[0]
                    if "=" in pair:
                        name, value = pair.split("=", 1)
                        self.cookies[name.strip()] = value.strip()
                return response.status, body, response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers

    def get(self, path: str):
        return self._request("GET", path)

    def json_get(self, path: str):
        status, body, _ = self.get(path)
        try:
            return status, json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return status, None

    def json_post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        status, body, _ = self._request(
            "POST", path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            return status, json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return status, None

    def post_multipart(self, path: str, fields: dict, files: list[tuple[str, str, bytes]]):
        boundary = "----lostfound" + uuid.uuid4().hex
        buffer = io.BytesIO()
        for name, value in fields.items():
            buffer.write(f"--{boundary}\r\n".encode())
            buffer.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            buffer.write(str(value).encode("utf-8"))
            buffer.write(b"\r\n")
        for name, filename, blob in files:
            buffer.write(f"--{boundary}\r\n".encode())
            buffer.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            buffer.write(b"Content-Type: image/jpeg\r\n\r\n")
            buffer.write(blob)
            buffer.write(b"\r\n")
        buffer.write(f"--{boundary}--\r\n".encode())
        status, body, _ = self._request(
            "POST", path, data=buffer.getvalue(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        try:
            return status, json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return status, None

    def delete(self, path: str):
        status, body, _ = self._request("DELETE", path)
        try:
            return status, json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return status, None


def photo(color, *, size=(420, 320), scale=0.6, background=(238, 238, 240), brightness=1.0):
    canvas = Image.new("RGB", size, background)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box = int(min(size) * scale)
    left, top = (size[0] - box) // 2, (size[1] - box) // 2
    draw.rounded_rectangle([left, top, left + box, top + box], radius=box // 6, fill=(*color, 255))
    canvas.paste(layer, (0, 0), layer)
    if brightness != 1.0:
        overlay = Image.new("RGB", size, (255, 255, 255) if brightness > 1 else (0, 0, 0))
        canvas = Image.blend(canvas, overlay, abs(brightness - 1.0) * 0.35)
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def main() -> int:
    print("=" * 74)
    print(f"  端到端验收：{BASE_URL}")
    print("=" * 74)

    anon = Session()

    # ---------------- 页面渲染 ---------------- #
    print("\n[1] 页面渲染")
    for path, expect in (
        ("/", 200),
        ("/browse?type=lost", 200),
        ("/browse?type=found", 200),
        ("/browse?type=found&q=耳机", 200),
        ("/login", 200),
        ("/algorithm", 200),
        ("/item/1", 200),
        ("/no-such-page", 404),
    ):
        status, body, headers = anon.get(path)
        check(f"GET {path} → {expect}", status == expect, f"实际 {status}")
        if expect == 200 and "text/html" in headers.get("Content-Type", ""):
            check(f"  {path} 返回 HTML 且包含导航", b"<nav" in body or b"site-header" in body)

    # ---------------- 静态资源 ---------------- #
    print("\n[2] 静态资源")
    for path in ("/static/css/style.css", "/static/js/common.js", "/static/js/publish.js", "/static/js/algorithm.js"):
        status, body, headers = anon.get(path)
        check(f"GET {path}", status == 200 and len(body) > 200, f"状态 {status} 长度 {len(body)}")

    # ---------------- 只读 API ---------------- #
    print("\n[3] 只读接口")
    status, data = anon.json_get("/api/stats")
    check("GET /api/stats", status == 200 and data and data.get("ok"), str(data)[:120])
    check("  统计包含总量字段", bool(data and "total_count" in data.get("stats", {})))

    status, data = anon.json_get("/api/categories")
    check("GET /api/categories 分类数 ≥ 10", status == 200 and len(data.get("categories", [])) >= 10)

    status, data = anon.json_get("/api/items?type=lost&limit=5")
    check("GET /api/items", status == 200 and data.get("ok"))
    check("  列表项字段完整", bool(data["items"]) and all(
        key in data["items"][0] for key in ("id", "title", "type", "tags", "images", "category_name")
    ))
    check("  图片特征已序列化（无 bytes）", data["items"][0]["images"] == [] or isinstance(
        data["items"][0]["images"][0].get("phash_bits_hex"), str
    ))

    status, data = anon.json_get("/api/items/1")
    check("GET /api/items/1", status == 200 and data.get("ok"))
    item1 = data.get("item", {})
    check("  含轮廓指纹", bool(item1.get("images")) and bool(item1["images"][0].get("shape_sig_hex")))

    status, data = anon.json_get("/api/items/1/matches")
    check("GET /api/items/1/matches", status == 200 and data.get("ok"))
    matches = data.get("matches", [])
    check("  返回候选与权重", "weights" in data)
    if matches:
        top = matches[0]
        check("  匹配含三通道得分", all(k in top for k in ("score", "image_score", "text_score", "reasons")))
        check("  匹配含可读理由", bool(top["reasons"]) and isinstance(top["reasons"][0], str))
        check("  按分数降序", all(matches[i]["score"] >= matches[i + 1]["score"] for i in range(len(matches) - 1)))

    status, data = anon.json_get("/api/matches/top?limit=5")
    check("GET /api/matches/top", status == 200 and data.get("ok"))

    # ---------------- 鉴权 ---------------- #
    print("\n[4] 鉴权")
    status, data = anon.json_get("/api/auth/me")
    check("未登录时 /api/auth/me 返回 null", status == 200 and data.get("user") is None)

    status, data = anon.json_post("/api/auth/login", {"username": "demo", "password": "wrong"})
    check("错误口令 → 401", status == 401)

    demo = Session()
    status, data = demo.json_post("/api/auth/login", {"username": "demo", "password": "demo1234"})
    check("正确口令 → 200", status == 200 and data.get("user", {}).get("username") == "demo")

    status, _body, _headers = demo.get("/publish")
    check("登录后可访问 /publish", status == 200)
    status, _body, _headers = demo.get("/admin")
    check("登录后可访问 /admin", status == 200)

    # urllib 默认跟随重定向，所以这里断言"最终落到登录页"（浏览器里的真实观感），
    # 而不是断言 302 本身。
    status, body, _headers = anon.get("/publish")
    check("未登录访问 /publish 最终落到登录页", status == 200 and "登录".encode() in body)

    # ---------------- 发布 → 匹配 闭环 ---------------- #
    print("\n[5] 发布 → 匹配 闭环（真实图片上传）")
    tag = uuid.uuid4().hex[:6]
    status, data = demo.post_multipart(
        "/api/items",
        {
            "type": "lost",
            "title": f"验收用黑色双肩包 {tag}",
            "category": "bag",
            "description": "验收脚本生成的记录：图书馆四楼落下的黑色双肩包，内有教材。",
            "tags": "黑色,双肩包,书包,图书馆,教材",
            "color_name": "黑色",
            "location": "图书馆四楼自习区",
            "contact": "13800000000",
        },
        [("images", "lost.jpg", photo((30, 32, 38), scale=0.6, brightness=0.95))],
    )
    check("发布寻物启事 → 201", status == 201 and data and data.get("ok"), str(data)[:160])
    lost_id = data["item"]["id"] if data and data.get("item") else None
    check("记录包含 1 张图片", bool(data) and len(data["item"]["images"]) == 1)

    status, data = demo.post_multipart(
        "/api/items",
        {
            "type": "found",
            "title": f"验收用图书馆捡到黑色背包 {tag}",
            "category": "bag",
            "description": "验收脚本生成的记录：图书馆四楼发现黑色背包，内有教材和电源适配器。",
            "tags": "黑色,背包,书包,图书馆,教材,电源适配器",
            "color_name": "黑色",
            "location": "图书馆",
            "contact": "13900000000",
        },
        [("images", "found.jpg", photo((30, 32, 38), scale=0.8, background=(214, 226, 240), brightness=1.08))],
    )
    check("发布失物招领 → 201", status == 201 and data.get("ok"))
    found_id = data["item"]["id"]
    found_matches = data.get("matches", [])
    check("发布时即时返回匹配", isinstance(found_matches, list))
    check("匹配命中刚发布的寻物启事", bool(found_matches) and found_matches[0]["item"]["id"] == lost_id,
          f"实际 {[m['item']['id'] for m in found_matches]}")

    if found_matches:
        best = found_matches[0]
        check("匹配得分超过疑似阈值 0.30", best["score"] >= 0.30, f"实际 {best['score']}")
        check("图像通道参与打分", best["image_score"] > 0.0, f"实际 {best['image_score']}")
        check("包含匹配理由", len(best["reasons"]) >= 1)
        print(f"      最佳匹配得分 {best['score']:.3f}（图像 {best['image_score']:.3f} "
              f"文本 {best['text_score']:.3f}）")
        for reason in best["reasons"][:3]:
            print(f"        · {reason}")

    # 反向查询
    status, data = anon.json_get(f"/api/items/{lost_id}/matches")
    check("反向查询同样命中", bool(data.get("matches")) and data["matches"][0]["item"]["id"] == found_id)

    # 图片确实可访问
    status, data = anon.json_get(f"/api/items/{lost_id}")
    thumb = data["item"]["images"][0]["thumb"]
    status, body, headers = anon.get(f"/static/uploads/{thumb}")
    check("上传的缩略图可访问", status == 200 and len(body) > 500, f"状态 {status} 字节 {len(body)}")

    # ---------------- 试算接口 ---------------- #
    print("\n[6] 试算接口")
    status, data = anon.json_post(
        "/api/match/text",
        {
            "left": {"title": "黑色双肩包", "tags": ["黑色", "双肩包"], "location": "图书馆"},
            "right": {"title": "图书馆捡到黑色背包", "tags": ["黑色", "背包"], "location": "图书馆"},
        },
    )
    check("POST /api/match/text", status == 200 and data.get("result", {}).get("score", 0) > 0)

    import base64

    def data_url(blob: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(blob).decode()

    blue = photo((30, 90, 200), scale=0.6)
    status, data = anon.json_post(
        "/api/match/pair",
        {
            "left": {"title": "蓝色物品 A", "tags": ["蓝色"], "images": [data_url(blue)]},
            "right": {
                "title": "蓝色物品 B",
                "tags": ["蓝色"],
                "images": [data_url(photo((30, 90, 200), scale=0.8, brightness=1.1))],
            },
        },
    )
    check("POST /api/match/pair 使用图像通道", status == 200 and data["result"]["image_score"] > 0,
          f"image_score={data.get('result', {}).get('image_score') if data else None}")

    status, data = anon.json_post(
        "/api/match/pair",
        {
            "left": {"title": "蓝色物品 A", "images": [data_url(blue)]},
            "right": {"title": "红色物品 B", "images": [data_url(photo((210, 60, 60), scale=0.6))]},
        },
    )
    same_color_score = 0.0
    check("异色物品得分更低", status == 200 and data["result"]["image_score"] < 0.85)

    # ---------------- 错误处理 ---------------- #
    print("\n[7] 错误处理")
    status, data = anon.json_post("/api/items", {"type": "lost", "title": ""})
    check("空标题 → 422", status == 422 and data and not data.get("ok"))

    status, data = anon.json_post("/api/items", {"type": "bad-type", "title": "测试标题"})
    check("非法类型 → 422", status == 422)

    status, data = anon.json_get("/api/items/999999")
    check("不存在记录 → 404", status == 404 and data and not data.get("ok"))

    status, data = anon.json_get("/api/no-such-endpoint")
    check("不存在接口 → 404 JSON", status == 404 and data is not None and not data.get("ok"))

    status, data = anon.delete(f"/api/items/{found_id}")
    check("未登录删除 → 401", status == 401)

    status, data = anon.json_get("/api/items?q=%27%20OR%201%3D1%20--")
    check("SQL 注入式输入安全", status == 200 and data.get("total") == 0)

    # ---------------- 清理 ---------------- #
    print("\n[8] 清理验收数据")
    if lost_id:
        status, data = demo.delete(f"/api/items/{lost_id}")
        check(f"删除寻物启事 #{lost_id}", status == 200 and data.get("ok"))
    if found_id:
        status, data = demo.delete(f"/api/items/{found_id}")
        check(f"删除失物招领 #{found_id}", status == 200 and data.get("ok"))
    status, data = anon.json_get(f"/api/items/{lost_id}")
    check("删除后记录不可访问", status == 404)

    # ---------------- 汇总 ---------------- #
    print("\n" + "=" * 74)
    print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
    if FAIL:
        print("  失败清单：")
        for name in FAIL:
            print(f"    ✗ {name}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
