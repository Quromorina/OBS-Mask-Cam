# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os

# onnxruntime の DLL を収集
ort_binaries = collect_dynamic_libs('onnxruntime')

# customtkinter のテーマデータを収集
ctk_datas = collect_data_files('customtkinter')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=ort_binaries,
    datas=[
        ('yolov8n-face.onnx', '.'),          # ONNXモデル
    ] + ctk_datas,
    hiddenimports=[
        'onnxruntime',
        'customtkinter',
        'pyvirtualcam',
        'pygrabber',
        'pygrabber.dshow_graph',
        'comtypes',
        'PIL',
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'ultralytics',
        'mediapipe',
        'matplotlib', 'scipy', 'polars',
        'numpy.f2py', 'pandas',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OBSMaskCam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OBSMaskCam',
)
