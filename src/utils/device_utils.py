# src/core/device_utils.py
import torch

def norm_device(device) -> torch.device:
    # int -> cuda:<int>
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    s = str(device).strip().lower()

    # "0" -> cuda:0
    if s.isdigit():
        return torch.device(f"cuda:{s}" if torch.cuda.is_available() else "cpu")

    # "gpu"/"cuda" -> cuda:0
    if s in ("gpu", "cuda"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # "cpu", "cuda:1", ...
    return torch.device(s)
