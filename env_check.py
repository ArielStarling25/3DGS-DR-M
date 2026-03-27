import os, sys
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
import torch
import simple_knn
from tetranerf.utils.extension import cpp
print('[SUCCESS] All submodules loaded successfully!')

if torch.cuda.is_available():
    print("[SUCCESS] torch CUDA is Enabled")
    # Optionally, print the number of GPUs and their names
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Installed CUDA version: {torch.version.cuda}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("[FAIL] CUDA is NOT enabled or accessible by PyTorch.")

try:
    from torch.utils.tensorboard import SummaryWriter
    print("[SUCCESS] Tensorboard is Enabled")
except ImportError as e:
    print("[FAIL] Tensorboard is NOT enabled")
    print(e)

try:
    from pytorch3d.io import IO
    print("[SUCCESS] PyTorch3D is Enabled")
except ImportError as e:    
    print("[FAIL] PyTorch3D is NOT enabled")
    print(e)

import GPUtil
all_gpus = GPUtil.getGPUs()

print("--- RAW GPU DATA ---")
for gpu in all_gpus:
    print(f"GPU ID: {gpu.id}")
    print(f"Name: {gpu.name}")
    print(f"Current Load: {gpu.load * 100:.1f}%")
    print(f"Current Memory Usage: {gpu.memoryUtil * 100:.1f}%")
    print(f"Total Memory: {gpu.memoryTotal} MB")
    print("--------------------")

available_gpus = GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5)
print(f"\nGPUs available (under 50% usage): {available_gpus}")

print("[CHECK] GPUtil Works!")