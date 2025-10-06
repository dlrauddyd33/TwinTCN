"""
Utility functions for seed setting, parameter counting, and preprocessing.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(m: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _zscore(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Apply z-score normalization."""
    mu = np.nanmean(a, axis=(0, 1), keepdims=True)
    sd = np.nanstd(a, axis=(0, 1), keepdims=True)
    return (a - mu) / (sd + eps)


def preprocess_viewcor_one_item(item) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess one data item for ViewCor format.
    
    Args:
        item: Tuple of (data, meta) where data is (T, >=8) array
        
    Returns:
        X: (T, 7) feature array [v_diff, vmin, soc, t_diff, tmin, cur, mileage]
        label: int label (0 or 1)
        car_id: int car identifier
    """
    data = item[0].astype(np.float32)
    data = data[:, :8]
    seq = data.shape[0]
    
    volt = data[:, 0:1]
    cur = data[:, 1:2]
    soc = data[:, 2:3]
    vmin = data[:, 3:4]
    vmax = data[:, 4:5]
    tmin = data[:, 5:6]
    tmax = data[:, 6:7]

    meta = item[1]
    lb = int(meta['label'][0])
    car_id = int(meta['car'])
    mile = float(meta['mileage'])
    miles = np.array([[mile] for _ in range(seq)])
    
    X = np.concatenate([vmax - vmin, vmin, soc, tmax - tmin, tmin, cur, miles], axis=1)  # (T, 7)
    
    return X, lb, car_id
