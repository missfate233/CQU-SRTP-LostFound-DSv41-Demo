"""一键启动编排：``python serve.py``（Windows 上由 ``start.bat`` 调用）

为什么单独有这么一个文件，而不是把所有逻辑写在 ``start.bat`` 里
-----------------------------------------------------------------
``cmd.exe`` 用 **OEM 代码页**（中文 Windows 上是 936/GBK）解析批处理文件。
如果 .bat 以 UTF-8 保存且含中文，cmd 会把中文字节按 GBK 解码成乱码，
再把这些乱码当成命令去执行，于是出现满屏的
``'xx' is not recognized as an internal or external command``。

因此约定：**批处理文件只允许 ASCII**，所有中文提示都放在 Python 里。

模式
----
    python serve.py                 # 自检 → 首次自动灌演示数据 → 启动
    python serve.py --reset         # 启动前强制重建演示数据
    python serve.py --no-seed       # 首次运行也不灌演示数据（空库启动）
    python serve.py --probe         # 仅探测当前解释器是否可用（供 start.bat 选用）
    python serve.py --check         # 只做可移植性自检，不启动服务
    python serve.py --recreate-venv # 目录移动/改名后重建 .venv（修复失效的绝对路径）

关于「移动或改名项目文件夹」
----------------------------
项目源码本身**可任意移动改名**：所有路径都由文件位置推导（``BASE_DIR``），
没有硬编码的绝对路径。唯二需要注意的是：

1. ``.venv`` 内部记录的是**绝对路径**，移动后会失效 —— 用 ``--recreate-venv``
   重建，或干脆删掉 ``.venv``（本项目自带 ``vendor/`` 依赖，系统有 Python 就能跑）；
2. 数据库与上传目录用的是**相对项目的路径**，会跟着文件夹一起走，无需处理。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
DB_PATH = os.path.join(BASE_DIR, "data", "lostfound.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
VENV_DIR = os.path.join(BASE_DIR, ".venv")

for _path in (VENDOR_DIR, BASE_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# 需要在子进程里继承的环境变量（bootstrap/seed 也依赖 vendor 路径）
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": os.pathsep.join([VENDOR_DIR, BASE_DIR])}

REQUIRED_PACKAGES = (("flask", "Flask"), ("PIL", "Pillow"), ("numpy", "numpy"))


def _setup_console() -> None:
    """让 Windows 控制台能正确输出中文与 UTF-8 子进程输出。"""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _banner(text: str) -> None:
    print("=" * 62)
    print(f"  {text}")
    print("=" * 62)


def _display_width(text: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列，用于手工对齐而不用 ``str.ljust``。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _run(script: str, *args: str) -> int:
    """以当前解释器运行项目内脚本，实时转发输出。"""
    command = [sys.executable, os.path.join(BASE_DIR, script), *args]
    try:
        return subprocess.call(command, cwd=BASE_DIR, env=CHILD_ENV)
    except OSError as exc:
        print(f"✗ 无法执行 {script}：{exc}", file=sys.stderr)
        return 1


def missing_packages() -> list[str]:
    """返回缺失的依赖（供 --probe 与启动自检共用）。"""
    missing = []
    for import_name, dist_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(dist_name)
    return missing


def probe() -> int:
    """探测当前解释器能否跑起本项目；退出码 0 表示可用。

    供 ``start.bat`` 逐个候选解释器调用。移动文件夹后 ``.venv`` 会指向不存在的
    绝对路径，那时本探测会失败，启动器便自动改用 ``py -3`` 或 ``python``。
    """
    if sys.version_info < (3, 10):
        print(f"需要 Python 3.10 及以上，当前为 {sys.version.split()[0]}", file=sys.stderr)
        return 1
    missing = missing_packages()
    if missing:
        print("缺少依赖：" + "、".join(missing), file=sys.stderr)
        return 1
    return 0


def portability_check() -> int:
    """可移植性自检：打印路径来源、依赖位置、数据目录状态与潜在问题。"""
    _banner("可移植性自检")
    ok = True

    print("[解释器]")
    print(f"  可执行文件 : {sys.executable}")
    print(f"  版本       : {sys.version.split()[0]}")

    print("\n[路径解析]（全部由文件位置推导，无硬编码绝对路径）")
    for label, path in (
        ("项目根目录", BASE_DIR),
        ("vendor 依赖", VENDOR_DIR),
        ("数据库文件", DB_PATH),
        ("上传目录", UPLOAD_DIR),
    ):
        mark = "✓" if os.path.exists(path) else "○"
        # 中文字符在终端占两列，用 len() 补空格会错位，这里按显示宽度补齐
        pad = " " * max(0, 12 - _display_width(label))
        print(f"  {mark} {label}{pad}{path}")
    if not os.path.isdir(BASE_DIR):
        ok = False

    print("\n[依赖来源]")
    if os.path.isdir(VENDOR_DIR):
        print(f"  ✓ 项目自带 vendor/（{len(os.listdir(VENDOR_DIR))} 个包），无需 pip")
    else:
        print("  ○ 没有 vendor/ 目录 —— 请运行 python bootstrap.py，或使用 pip 安装")
    missing = missing_packages()
    if missing:
        print(f"  ✗ 仍缺少：{'、'.join(missing)}")
        ok = False
    else:
        print("  ✓ Flask / Pillow / numpy 均可导入")

    print("\n[数据]（存放在项目内部，移动文件夹时会一起带过去）")
    if os.path.isfile(DB_PATH):
        print(f"  ✓ 数据库已存在：{DB_PATH}")
    else:
        print("  ○ 数据库尚未创建，首次启动会自动生成演示数据")
    if os.path.isdir(UPLOAD_DIR):
        images = [n for n in os.listdir(UPLOAD_DIR) if n.lower().endswith(".jpg")]
        print(f"  ✓ 上传目录含 {len(images)} 张图片")

    print("\n[虚拟环境]")
    if os.path.isdir(VENV_DIR):
        cfg = os.path.join(VENV_DIR, "pyvenv.cfg")
        home = ""
        if os.path.isfile(cfg):
            for line in open(cfg, encoding="utf-8", errors="replace"):
                if line.lower().startswith("home"):
                    home = line.split("=", 1)[-1].strip()
        healthy = bool(home) and os.path.isfile(os.path.join(home, "python.exe"))
        print(f"  {'✓' if healthy else '✗'} .venv → {home or '(未记录 home)'}")
        if not healthy:
            print(".venv 失效：目录移动/改名后需重建；本项目可仅用系统 Python + vendor/。")
    else:
        print("  ○ 没有 .venv（本项目不依赖它，vendor/ 已足够）")

    print()
    if ok:
        print("结论：✓ 当前目录可以直接运行 —— python serve.py")
    else:
        print("结论：✗ 需要先处理上面的 ✗ 项")
    return 0 if ok else 1


def find_base_python() -> list[str] | None:
    """找一个可用的系统 Python 3 来重建 venv（返回命令前缀，如 ``['py', '-3']``）。"""
    candidates: list[list[str]] = []
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append([found])
    py_launcher = shutil.which("py")
    if py_launcher:
        candidates.append([py_launcher, "-3"])

    for command in candidates:
        try:
            result = subprocess.run(
                [*command, "-c", "import sys; print(sys.version_info[0])"],
                capture_output=True,
                text=True,
                timeout=30,
                env=CHILD_ENV,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout.strip() == "3":
            return command
    return None


def recreate_venv() -> int:
    """重建 .venv（修复因移动/改名文件夹而失效的绝对路径）。"""
    _banner("重建虚拟环境")
    base = find_base_python()
    if base is None:
        print("✗ 找不到可用的系统 Python 3。")
        print("  请先安装 Python 3.10+（安装时勾选 Add python.exe to PATH），")
        print("  或者直接使用 vendor/ 依赖：python bootstrap.py")
        return 1
    print(f"  使用解释器：{' '.join(base)}")

    if os.path.isdir(VENV_DIR):
        print("  删除旧的 .venv ...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        if os.path.isdir(VENV_DIR):
            print("✗ 无法删除旧 .venv（可能被占用），请关闭相关进程后重试。", file=sys.stderr)
            return 1

    print("  创建新的 .venv ...")
    result = subprocess.run([*base, "-m", "venv", VENV_DIR], cwd=BASE_DIR, env=CHILD_ENV)
    if result.returncode != 0:
        print("✗ venv 创建失败。可直接跳过虚拟环境：python bootstrap.py", file=sys.stderr)
        return 1

    scripts = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    new_python = os.path.join(VENV_DIR, scripts, exe)
    if not os.path.isfile(new_python):
        print(f"✗ 未找到新解释器：{new_python}", file=sys.stderr)
        return 1

    print("  ✓ 已重建。校验依赖 ...")
    check = subprocess.run(
        [new_python, "-c", "import flask, PIL, numpy"],
        cwd=BASE_DIR,
        env=CHILD_ENV,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        print("  ✓ 依赖已就绪（复用了系统的 site-packages）")
    else:
        print("  新环境里没有依赖，正在安装到 vendor/ ...")
        if _run("bootstrap.py") != 0:
            print("✗ 依赖安装失败，请检查网络后重试。", file=sys.stderr)
            return 1
        print("  ✓ 依赖已安装到 vendor/")

    print()
    print("完成。启动：start.bat   或   .venv\\Scripts\\python.exe serve.py")
    return 0


def main() -> int:
    _setup_console()
    parser = argparse.ArgumentParser(description="启动 / 自检 CQU-SRTP 失物招领系统")
    parser.add_argument("--no-seed", action="store_true", help="首次运行时不写入演示数据")
    parser.add_argument("--reset", action="store_true", help="启动前强制重建演示数据")
    parser.add_argument("--probe", action="store_true", help="仅探测当前解释器是否可用（供 start.bat 调用）")
    parser.add_argument("--check", action="store_true", help="只做可移植性自检，不启动服务")
    parser.add_argument("--recreate-venv", action="store_true", help="重建 .venv（目录移动后修复）")
    args = parser.parse_args()

    # 供启动器静默探测：不打印内容，只用退出码表达结果
    if args.probe:
        return probe()

    if args.recreate_venv:
        return recreate_venv()

    if args.check:
        return portability_check()

    _banner("CQU-SRTP 失物招领系统 · 启动自检")

    # ---- 1. 依赖 ----
    print("[1/3] 检查运行依赖 ...")
    missing = missing_packages()
    if missing:
        print(f"      缺少：{'、'.join(missing)}")
        print("      正在调用 bootstrap.py 自动安装（无需 pip）...")
        if _run("bootstrap.py") != 0:
            print()
            print("✗ 依赖安装失败。请手动执行其中之一：", file=sys.stderr)
            print("    python bootstrap.py", file=sys.stderr)
            print("    pip install -r requirements.txt", file=sys.stderr)
            return 1
        # 子进程安装完成后需清理模块缓存，才能看到新装进 vendor/ 的包
        for module in [m for m in list(sys.modules) if m.split(".")[0] in ("flask", "PIL", "numpy", "werkzeug", "jinja2")]:
            sys.modules.pop(module, None)
        still_missing = missing_packages()
        if still_missing:
            print(f"✗ 安装后仍缺少：{'、'.join(still_missing)}", file=sys.stderr)
            return 1
        print("      ✓ 依赖已就绪")
    else:
        print("      ✓ 依赖已就绪（Flask / Pillow / numpy）")

    # ---- 2. 数据 ----
    if args.reset:
        print("[2/3] 重建演示数据（--reset）...")
        if _run("seed.py", "--reset") != 0:
            return 1
    elif not os.path.isfile(DB_PATH):
        if args.no_seed:
            print("[2/3] 首次运行：跳过演示数据（--no-seed），将创建空数据库")
        else:
            print("[2/3] 首次运行：写入演示数据 ...")
            if _run("seed.py", "--reset") != 0:
                return 1
    else:
        print("[2/3] 数据库已存在，沿用现有数据（如需重建：python serve.py --reset）")

    # ---- 3. 启动 ----
    print("[3/3] 启动服务 ...")
    print()
    from run import main as run_server  # 延迟导入，确保自检信息先输出

    run_server()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已停止。")
        raise SystemExit(0)
