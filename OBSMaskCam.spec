# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
import os

# customtkinter のテーマデータを収集
ctk_datas = collect_data_files('customtkinter')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 顔検出モデル（YuNet / OpenCV Zoo, MIT License）。推論は OpenCV の DNN で行う
        ('face_detection_yunet_2023mar.onnx', '.'),
        # モデルのライセンス文（MIT の条件として配布物に同梱する）
        ('licenses/YuNet-LICENSE.txt', 'licenses'),
    ] + ctk_datas,
    hiddenimports=[
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
        'onnxruntime',  # 顔検出は OpenCV の DNN に移したので不要（環境に残っていても同梱しない）
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
    upx=False,  # UPX圧縮はウイルス対策ソフトの誤検知を招きやすいため使わない
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,  # UPX圧縮はウイルス対策ソフトの誤検知を招きやすいため使わない
    upx_exclude=[],
    name='OBSMaskCam',
)
