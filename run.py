"""启动入口：``python run.py``

负责三件事：
1. 把 ``vendor/`` 加入 ``sys.path``（本项目自带依赖目录，见 ``bootstrap.py``），
   使得无需 pip / 虚拟环境也能直接运行；
2. 启动前自检 Flask / Pillow / numpy 是否就绪，缺失时给出明确的修复指引；
3. 初始化数据库（建表 + 默认分类 + 演示账号）并启动开发服务器。

环境变量：
    LOSTFOUND_HOST    监听地址，默认 127.0.0.1
    LOSTFOUND_PORT    监听端口，默认 5000
    LOSTFOUND_DEBUG   置 1 开启调试模式（自动重载）
    LOSTFOUND_DB      SQLite 文件路径
    LOSTFOUND_SECRET  session 密钥（生产环境务必设置）
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")

# vendor/ 必须在最前，保证本地依赖优先于系统安装的版本
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def preflight() -> None:
    """运行前依赖自检，缺失时打印可执行的修复命令后退出。"""
    missing: list[str] = []
    for import_name, dist_name in (("flask", "Flask"), ("PIL", "Pillow"), ("numpy", "numpy")):
        try:
            __import__(import_name)
        except ImportError:
            missing.append(dist_name)
    if not missing:
        return
    print("✗ 缺少依赖：" + "、".join(missing), file=sys.stderr)
    print("  修复方式（任选其一）：", file=sys.stderr)
    print("    1) python bootstrap.py          # 免 pip，自动下载解压到 vendor/", file=sys.stderr)
    print("    2) pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)


def pick_port(host: str, preferred: int, *, tries: int = 12) -> int:
    """端口被占用时自动向后寻找可用端口。

    Windows 上 5000 端口常被系统服务占用（而且被占用时 Flask 的报错不够直观），
    因此这里显式探测并给出友好提示，而不是直接崩掉。
    """
    import socket

    for offset in range(tries):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit(f"✗ {host} 上从 {preferred} 起连续 {tries} 个端口都被占用，请用 LOSTFOUND_PORT 指定其他端口")


def main() -> None:
    preflight()
    from lostfound.app import create_app  # 延迟导入，保证自检信息先于 Flask 报错出现

    app = create_app()
    host = os.environ.get("LOSTFOUND_HOST", "127.0.0.1")
    preferred = int(os.environ.get("LOSTFOUND_PORT", "5000"))
    debug = os.environ.get("LOSTFOUND_DEBUG", "0") == "1"
    port = pick_port(host, preferred)
    if port != preferred:
        print(f"! 端口 {preferred} 已被占用，自动改用 {port}")

    db_path = app.config["DATABASE"]
    print("=" * 62)
    print("  CQU-SRTP 失物招领系统  ·  图像感知匹配 + 文本标签匹配")
    print("=" * 62)
    print(f"  访问地址 : http://{host}:{port}")
    print(f"  数据库   : {db_path}")
    print(f"  上传目录 : {app.config['UPLOAD_FOLDER']}")
    print(f"  演示账号 : demo / demo1234   ·   管理员 admin / demo1234")
    print(f"  调试模式 : {'开启' if debug else '关闭'}")
    print("=" * 62)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
