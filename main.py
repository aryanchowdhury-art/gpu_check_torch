import torch

if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU (Mac M1/M2/M3)
elif torch.backends.rocm.is_available():
    device = "rocm"  # AMD GPU (Linux only)
else:
    device = "cpu"  # No GPU, use CPU

print("Using device:", device)
