# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'cv2',
        'matplotlib',
        'numpy',
        'scipy',
        'segment_anything',
        'torch',
        'torch._C',
        'torchvision',
        'PIL',
        'tkinter',
        'requests',
        'tqdm'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add torch binaries manually after analysis
import os
import torch
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_lib_path):
    for file in os.listdir(torch_lib_path):
        if file.endswith('.dll'):
            a.binaries.append((os.path.join(torch_lib_path, file), 'torch/lib'))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='WheatgrassAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # IMPORTANT: Disable UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)