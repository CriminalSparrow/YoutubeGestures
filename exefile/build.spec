# build.spec

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

# Путь к PyQt5 платформенным плагинам
qt_platform_plugins = [
    (os.path.join(
        '..', 'gesture_env310', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins', 'platforms'),
     'platforms')
]

# Добавляем .pth файл из папки exefile
additional_datas = [
    ('trained_res18_full_train_all_gestures.pth', '.')
]

datas = [
    (os.path.join('..', 'gesture_env310', 'Lib', 'site-packages', 'mediapipe', 'modules', 'hand_landmark'), 'mediapipe/modules/hand_landmark'),
    (os.path.join('..', 'gesture_env310', 'Lib', 'site-packages', 'mediapipe', 'modules', 'palm_detection'), 'mediapipe/modules/palm_detection'),
]


# Скрипт
script_path = 'gesture_controller_gui.py'

block_cipher = None

a = Analysis(
    [script_path],
    pathex=[os.path.abspath('.')], 
    binaries=[],
    datas=qt_platform_plugins + additional_datas + datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YoutubeGesturesApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YoutubeGesturesApp'
)
