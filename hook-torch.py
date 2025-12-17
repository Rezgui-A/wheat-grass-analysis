# hook-torch.py
import os
import glob
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, get_package_paths

# Get torch installation path
torch_path = get_package_paths('torch')[0]
torch_lib = os.path.join(torch_path, 'lib')

binaries = []
hiddenimports = []

# Add ALL DLLs from torch/lib
if os.path.exists(torch_lib):
    for dll in glob.glob(os.path.join(torch_lib, '*.dll')):
        binaries.append((dll, 'torch/lib'))

# Also collect via standard method
binaries += collect_dynamic_libs('torch')

# Hidden imports
hiddenimports += collect_submodules('torch')
hiddenimports += ['torch._C', 'torch._dl', 'torch._dynamo', 'torch._inductor']

# Add torchvision
try:
    import torchvision
    binaries += collect_dynamic_libs('torchvision')
    hiddenimports += collect_submodules('torchvision')
except:
    pass

# Add segment_anything
try:
    hiddenimports += collect_submodules('segment_anything')
except:
    pass

# MKL and other dependencies
mkl_files = []
for root, dirs, files in os.walk(torch_lib):
    for file in files:
        if file.startswith('mkl_') or file.startswith('libiomp') or file.startswith('vcomp'):
            mkl_files.append((os.path.join(root, file), '.'))
binaries += mkl_files