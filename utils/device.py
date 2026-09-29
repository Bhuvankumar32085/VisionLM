"""Device selection and hardware capability detection utilities."""

from typing import Any, Dict, Optional
import torch
import torchvision


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Returns the torch.device to use throughout the application.

    Args:
        preferred_device: Optional device name (e.g. 'cuda', 'cuda:0', 'cpu').
                          If None or 'auto', automatically selects CUDA if available.

    Returns:
        torch.device instance.
    """
    if preferred_device is not None and preferred_device != "auto":
        device = torch.device(preferred_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({preferred_device}) but is not available on this system.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_gpu_memory_mb() -> Optional[float]:
    """Returns the allocated GPU memory in MB if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


def get_gpu_name() -> Optional[str]:
    """Returns the name of the active GPU if available."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def get_hardware_info() -> Dict[str, Any]:
    """Inspects and returns system hardware, CUDA, and PyTorch capabilities."""
    info: Dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "compute_capability": None,
        "total_vram_gb": 0.0,
        "total_vram_mb": 0.0,
        "bf16_supported": False,
        "fp16_supported": False,
    }

    if torch.cuda.is_available():
        curr_device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(curr_device)
        info["gpu_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["total_vram_gb"] = props.total_memory / (1024**3)
        info["total_vram_mb"] = props.total_memory / (1024**2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        # FP16 tensor core support exists on compute capability >= 7.0
        info["fp16_supported"] = props.major >= 7

    return info
