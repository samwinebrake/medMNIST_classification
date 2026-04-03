import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Seed common random number generators for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Random seed applied to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
