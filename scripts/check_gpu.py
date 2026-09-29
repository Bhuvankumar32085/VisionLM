"""GPU and CUDA environment verification script."""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.device import get_hardware_info


def check_gpu() -> bool:
    """Checks and prints PyTorch, Torchvision, and CUDA GPU details."""
    info = get_hardware_info()

    print("=" * 60)
    print("GPU & HARDWARE ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"PyTorch Version:         {info['pytorch_version']}")
    print(f"Torchvision Version:     {info['torchvision_version']}")
    print(f"CUDA Available:          {info['cuda_available']}")

    if not info["cuda_available"]:
        print("\n[WARNING/ERROR] CUDA is NOT available to PyTorch.")
        print("Possible causes:")
        print("1. NVIDIA drivers are not installed or need an update.")
        print("2. PyTorch CPU-only version was accidentally installed.")
        print("3. No compatible NVIDIA GPU was detected.")
        print("=" * 60)
        return False

    print(f"CUDA Version:            {info['cuda_version']}")
    print(f"GPU Count:               {info['device_count']}")
    print(f"GPU Model:               {info['gpu_name']}")
    print(f"Compute Capability:      {info['compute_capability']}")
    print(f"Total VRAM:              {info['total_vram_gb']:.2f} GB ({info['total_vram_mb']:.0f} MB)")
    print(f"BF16 Precision Support:  {info['bf16_supported']}")
    print(f"FP16 Precision Support:  {info['fp16_supported']}")
    print("=" * 60)
    print("STATUS: Hardware verification PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
