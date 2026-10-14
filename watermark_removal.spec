# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for the portable Windows GUI.

Build with:
    pyinstaller --noconfirm watermark_removal.spec           # onefile (default)
    $env:WMR_BUILD_MODE="onedir"; pyinstaller --noconfirm watermark_removal.spec

Output:
    onefile -> dist/WatermarkRemover.exe
    onedir  -> dist/WatermarkRemover/WatermarkRemover.exe

Onefile gives a single portable exe (best for the CPU build). The CUDA build is
packed as onedir because a single-file CUDA exe exceeds GitHub Release's 2 GB
per-asset limit; the whole folder is zipped by the CI workflow instead.

The core inference path (LaMa via torch) is bundled. Optional heavy back-ends
(transformers/Florence-2, paddle/paddleocr, STTN/ProPainter) are excluded and
degrade gracefully at runtime; see README.
"""

import os

from PyInstaller.utils.hooks import collect_all

MODE = os.environ.get("WMR_BUILD_MODE", "onefile").lower()

# UPX compression shrinks the bundle noticeably. Disable with WMR_UPX=0.
# A few runtime/Qt DLLs are excluded because compressing them can trip
# antivirus false-positives or loader edge cases.
UPX = os.environ.get("WMR_UPX", "1").lower() not in ("0", "false", "no")
UPX_EXCLUDE = [
    "Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "Qt6Network.dll",
    "ucrtbase.dll", "vcruntime140.dll", "vcruntime140_1.dll",
]


def _resolve_icon():
    """Return a path to assets/app.ico, generating it from Pillow if missing."""
    from pathlib import Path

    ico = Path("assets/app.ico")
    if not ico.exists():
        try:
            import sys

            sys.path.insert(0, str(Path("assets").resolve()))
            from make_icon import build_ico

            build_ico(ico)
        except Exception as exc:  # noqa: BLE001 - icon is best-effort
            print(f"[spec] icon generation skipped: {exc}")
            return None
    return str(ico) if ico.exists() else None


ICON = _resolve_icon()

datas = []
binaries = []
hiddenimports = []

# Fully collect the packages that ship native/runtime data files.
for pkg in ("PySide6", "numpy", "cv2", "PIL", "torch"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Bundle the app icon so the running window/taskbar can reuse it.
if ICON:
    datas.append((ICON, "assets"))

# Modules imported lazily/conditionally inside factory functions - make sure
# PyInstaller still bundles them (their optional deps stay excluded).
hiddenimports += [
    "src.detection.florence2",
    "src.detection.paddleocr_det",
    "src.detection.manual",
    "src.inpaint.sttn",
    "src.inpaint.propainter",
    "src.inpaint.opencv_fallback",
]
hiddenimports = list(dict.fromkeys(hiddenimports))


a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "paddle",
        "paddleocr",
        "transformers",
        "accelerate",
        "torchvision",
        "torchaudio",
        "einops",
        "scipy",
        "matplotlib",
        "tkinter",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

if MODE == "onedir":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="WatermarkRemover",
        debug=False,
        strip=False,
        upx=UPX,
        upx_exclude=UPX_EXCLUDE,
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        icon=ICON,
    )
    COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=UPX,
        upx_exclude=UPX_EXCLUDE,
        name="WatermarkRemover",
    )
else:  # onefile
    EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="WatermarkRemover",
        debug=False,
        strip=False,
        upx=UPX,
        upx_exclude=UPX_EXCLUDE,
        console=False,
        disable_windowed_traceback=False,
        runtime_tmpdir=None,
        icon=ICON,
    )
