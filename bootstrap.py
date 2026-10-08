"""``python bootstrap.py`` —— 零 pip 依赖的依赖安装器（Windows/离线友好）。

为什么需要它
------------
部分受管环境会拦截 pip 的**子进程**写文件（``pip-unpack-*/**.whl.metadata`` 报
``PermissionError: [Errno 13]``），而普通 Python 进程写同一目录却完全正常。
本脚本绕过 pip：直接用标准库 ``urllib`` 从 PyPI 拉取纯 Python wheel（zip 包），
再用 ``zipfile`` 解压到 ``vendor/`` 目录。

优点：不依赖 pip、不写临时目录、可重复执行（幂等）、可离线二次分发
（把 ``vendor/`` 一起拷贝即可）。

用法
----
    python bootstrap.py            # 安装缺失的依赖到 vendor/
    python bootstrap.py --check    # 只检查，不下载
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
PYPI_JSON = "https://pypi.org/pypi/{name}/json"

# 运行时依赖（Flask 及其全部纯 Python 依赖）
REQUIRED = [
    ("Flask", "flask"),
    ("Werkzeug", "werkzeug"),
    ("Jinja2", "jinja2"),
    ("MarkupSafe", "markupsafe"),
    ("itsdangerous", "itsdangerous"),
    ("click", "click"),
    ("blinker", "blinker"),
]

# 这些包通常由 numpy/Pillow 提供；缺失时给出提示，不在本脚本内安装（需要编译/二进制轮子）
OPTIONAL_BINARY = [("Pillow", "PIL"), ("numpy", "numpy")]


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def module_version(dist_name: str) -> str:
    """用 importlib.metadata 取版本，避免访问已废弃的 ``__version__`` 触发告警。"""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(dist_name)
    except Exception:  # noqa: BLE001 - 版本号仅用于展示
        return "?"


def http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "lostfound-bootstrap/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 固定为 pypi.org
        return response.read()


def platform_tags() -> list[str]:
    """当前解释器可接受的 wheel 平台标签（按优先级）。"""
    import sysconfig

    tags: list[str] = []
    try:
        tags.extend(sysconfig.get_platform().replace("-", "_").replace(".", "_").split(","))
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        import platform as _platform

        machine = (_platform.machine() or "AMD64").lower()
        tags.extend(["win_amd64", "win32", "win_arm64"] if machine in ("amd64", "x86_64") else ["win_arm64", "win_amd64", "win32"])
    else:
        tags.extend(["manylinux2014_x86_64", "manylinux_2_17_x86_64", "linux_x86_64"])
    tags.append("any")
    return tags


def python_tags() -> list[str]:
    """当前解释器可接受的 wheel Python 标签。"""
    major, minor = sys.version_info[:2]
    # sys.abiflags 仅在 POSIX 上存在，Windows 上需 getattr 兜底
    abiflags = getattr(sys, "abiflags", "") or ""
    abi = "t" if "t" in abiflags else ""
    return [f"cp{major}{minor}{abi}", f"cp{major}{minor}", f"py{major}{minor}", f"py{major}", "py2.py3"]


def pick_wheel(meta: dict) -> tuple[str, str] | None:
    """从 PyPI 元数据里挑一个**当前解释器可安装**的 wheel。

    解析 wheel 文件名规范 ``{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl``：

    * 优先纯 Python 轮子（``py3-none-any``）—— 最安全；
    * 否则选择与当前 CPython 版本 + 平台标签完全匹配的二进制轮子
      （例如 MarkupSafe 只发布二进制轮子，Windows/py3.12 需要
      ``cp312-cp312-win_amd64``）；
    * 绝不贪心选择「体积最小」的轮子 —— 那样会误装 iOS/Android 目标包
      （它们的文件名同样含 ``any`` 或 ``cp3xx`` 片段）。
    """
    files = meta.get("urls") or []
    wanted_py = python_tags()
    wanted_plat = platform_tags()
    candidates: list[tuple[int, int, int, str, str]] = []

    for entry in files:
        name = entry.get("filename", "")
        if not name.endswith(".whl") or entry.get("packagetype") != "bdist_wheel":
            continue
        stem = name[: -len(".whl")]
        parts = stem.split("-")
        if len(parts) < 5:
            continue
        python_tag, abi_tag, platform_tag = parts[-3], parts[-2], parts[-1]

        score = 0
        if abi_tag == "none" and platform_tag == "any":
            score = 100 if python_tag.startswith("py3") else 90
        else:
            # 平台标签可能是多标签（manylinux 风格用 . 分隔）
            plat_tokens = set(platform_tag.split("."))
            if plat_tokens & set(wanted_plat):
                score = 70
            elif platform_tag in wanted_plat:
                score = 70
            else:
                continue
            # Python 标签必须是精确的 cp{major}{minor}，避免跨版本二进制不兼容
            if python_tag not in wanted_py:
                continue
            score -= wanted_py.index(python_tag)
        if score <= 0:
            continue
        candidates.append((-score, int(entry.get("size") or 0), 0, name, entry.get("url")))

    if not candidates:
        return None
    candidates.sort()
    _neg, _size, _z, name, url = candidates[0]
    return name, url


def install_package(dist_name: str, import_name: str, *, dest: str, force: bool = False) -> bool:
    target = os.path.join(dest, import_name)
    if os.path.isdir(target) and not force:
        log(f"{dist_name:<14} 已存在，跳过（{import_name}/）")
        return True
    log(f"{dist_name:<14} 查询 PyPI ...")
    try:
        raw = http_get(PYPI_JSON.format(name=dist_name))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log(f"{dist_name:<14} ✗ 网络不可用：{exc}")
        return False

    meta = json.loads(raw.decode("utf-8"))
    picked = pick_wheel(meta)
    if not picked:
        log(f"{dist_name:<14} ✗ 未找到可用的 wheel")
        return False
    filename, url = picked
    log(f"{dist_name:<14} 下载 {filename} ...")
    try:
        blob = http_get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        log(f"{dist_name:<14} ✗ 下载失败：{exc}")
        return False

    os.makedirs(dest, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                # 跳过 dist-info 里的脚本与元数据目录以外的安装钩子
                if ".data/scripts/" in member or member.startswith("__pycache__"):
                    continue
                archive.extract(member, dest)
    except zipfile.BadZipFile as exc:
        log(f"{dist_name:<14} ✗ 解压失败：{exc}")
        return False
    log(f"{dist_name:<14} ✓ 已解压到 vendor/")
    return True


def check() -> int:
    sys.path.insert(0, VENDOR_DIR)
    missing = []
    for dist_name, import_name in REQUIRED + OPTIONAL_BINARY:
        try:
            __import__(import_name)
            log(f"{dist_name:<14} ✓ 可导入")
        except ImportError:
            log(f"{dist_name:<14} ✗ 缺失")
            missing.append((dist_name, import_name))
    if missing:
        log("缺失依赖：" + ", ".join(name for name, _ in missing))
        return 1
    log("全部依赖就绪 ✓")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="把 Flask 及其依赖解压到 vendor/（不需要 pip）")
    parser.add_argument("--check", action="store_true", help="只检查依赖是否可用")
    parser.add_argument("--force", action="store_true", help="即使已存在也重新下载")
    parser.add_argument("--dest", default=VENDOR_DIR, help="安装目录（默认 vendor/）")
    args = parser.parse_args()

    if args.check:
        return check()

    log(f"目标目录：{args.dest}")
    if os.path.isdir(args.dest) and args.force:
        log("--force：清空已有目录")
        shutil.rmtree(args.dest, ignore_errors=True)
    os.makedirs(args.dest, exist_ok=True)

    failed = []
    for dist_name, import_name in REQUIRED:
        if not install_package(dist_name, import_name, dest=args.dest, force=args.force):
            failed.append(dist_name)

    log("-" * 52)
    log("校验导入 ...")
    sys.path.insert(0, args.dest)
    for dist_name, import_name in REQUIRED + OPTIONAL_BINARY:
        try:
            __import__(import_name)
            version = module_version(dist_name)
            log(f"{dist_name:<14} ✓ import {import_name} (v{version})")
        except ImportError as exc:
            log(f"{dist_name:<14} ✗ 无法导入：{exc}")
            if dist_name not in failed:
                failed.append(dist_name)

    if failed:
        log("失败：" + ", ".join(failed))
        return 1
    log("完成。启动命令： python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
