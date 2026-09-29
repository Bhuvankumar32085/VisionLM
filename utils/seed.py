"""Seed and reproducibility utilities."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python random, NumPy, PyTorch, and CUDA.

    Args:
        seed: The integer seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
