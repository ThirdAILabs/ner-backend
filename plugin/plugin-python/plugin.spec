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

torch_lib_path = os.path.dirname(torch.__file__)
torch_dylib_path = os.path.join(torch_lib_path, 'lib')

# Standard Homebrew path for libomp on Intel Macs
homebrew_intel_libomp_path = '/usr/local/opt/libomp/lib/libomp.dylib'

# Add torch dynamic libraries with full paths
torch_binaries = []
torch_dylib_datas = []
dylib_names = [
    'libtorch_python.dylib',
    'libtorch.dylib',
    # 'libomp.dylib', # We will handle this specially
    'libtorch_cpu.dylib',
    'libc10.dylib',
    'libshm.dylib'
]

for lib_name in dylib_names:
    lib_path_in_torch_lib = os.path.join(torch_dylib_path, lib_name)
    lib_path_in_torch_main = os.path.join(torch_lib_path, lib_name)

    if os.path.exists(lib_path_in_torch_lib):
        torch_binaries.append((lib_path_in_torch_lib, '.'))
        torch_dylib_datas.append((lib_path_in_torch_lib, '.'))
        print(f"Found {lib_name} in {torch_dylib_path}")
    elif os.path.exists(lib_path_in_torch_main):
        torch_binaries.append((lib_path_in_torch_main, '.'))
        torch_dylib_datas.append((lib_path_in_torch_main, '.'))
        print(f"Found {lib_name} in {torch_lib_path}")
    else:
        print(f"Warning: Could not find {lib_name} in PyTorch directories")

# --- Special handling for libomp.dylib ---
libomp_name = 'libomp.dylib'
found_libomp = False

# 1. Check in torch_dylib_path (torch/lib)
libomp_path_in_torch_lib = os.path.join(torch_dylib_path, libomp_name)
if os.path.exists(libomp_path_in_torch_lib):
    torch_binaries.append((libomp_path_in_torch_lib, '.'))
    torch_dylib_datas.append((libomp_path_in_torch_lib, '.'))
    print(f"Found {libomp_name} in {torch_dylib_path}")
    found_libomp = True
else:
    # 2. Check in torch_lib_path (main torch directory)
    libomp_path_in_torch_main = os.path.join(torch_lib_path, libomp_name)
    if os.path.exists(libomp_path_in_torch_main):
        torch_binaries.append((libomp_path_in_torch_main, '.'))
        torch_dylib_datas.append((libomp_path_in_torch_main, '.'))
        print(f"Found {libomp_name} in {torch_lib_path}")
        found_libomp = True
    else:
        # 3. Check standard Homebrew path for Intel Macs
        if os.path.exists(homebrew_intel_libomp_path):
            torch_binaries.append((homebrew_intel_libomp_path, '.'))
            torch_dylib_datas.append((homebrew_intel_libomp_path, '.'))
            print(f"Found {libomp_name} at Homebrew path: {homebrew_intel_libomp_path}")
            found_libomp = True
        else:
            print(f"Warning: Could not find {libomp_name} in PyTorch directories or at {homebrew_intel_libomp_path}")
            print(f"Please verify the location of {libomp_name} and update the script if necessary.")


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
