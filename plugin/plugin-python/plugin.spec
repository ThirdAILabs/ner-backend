# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules
import os
import transformers
import torch

datas=[]
datas += copy_metadata('torch')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('importlib_metadata')
datas += collect_data_files('transformers', include_py_files=True)

transformers_models_path = os.path.join(os.path.dirname(transformers.__file__), 'models')
datas.append((transformers_models_path, 'transformers/models'))


# Get torch library path
torch_lib_path = os.path.dirname(torch.__file__)
torch_dylib_path = os.path.join(torch_lib_path, 'lib')

# Add torch dynamic libraries with full paths
torch_binaries = []
torch_dylib_datas = []
dylib_names = [
    'libtorch_python.dylib',
    'libtorch.dylib',
    'libomp.dylib',
    'libtorch_cpu.dylib',
    'libc10.dylib',
    'libshm.dylib'
]

for lib in dylib_names:
    lib_path = os.path.join(torch_dylib_path, lib)
    if os.path.exists(lib_path):
        torch_binaries.append((lib_path, '.'))
        torch_dylib_datas.append((lib_path, '.'))
    else:
        # Try looking in the main torch directory
        lib_path = os.path.join(torch_lib_path, lib)
        if os.path.exists(lib_path):
            torch_binaries.append((lib_path, '.'))
            torch_dylib_datas.append((lib_path, '.'))
        else:
            print(f"Warning: Could not find {lib}")

# Add dylibs to datas
datas.extend(torch_dylib_datas)

needed_hiddenimports = [
    'torch',
    'tqdm',
    'regex',
    'sacremoses',
    'requests',
    'packaging',
    'filelock',
    'numpy',
    'tokenizers',
    'importlib_metadata',
    'transformers',          # Base transformers package
    'transformers.models'  # Explicitly include models, good practice
]
# Add all submodules found within the 'transformers' package
needed_hiddenimports += collect_submodules('transformers')
# Ensure the list is unique if there are overlaps
needed_hiddenimports = list(set(needed_hiddenimports))

a = Analysis(
    ['plugin.py'],
    pathex=[],
    binaries=torch_binaries,  # Add torch binaries here
    datas=datas,
    hiddenimports=needed_hiddenimports,
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='plugin',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='plugin',
)
