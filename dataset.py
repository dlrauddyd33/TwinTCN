"""
Dataset classes for battery data loading and preprocessing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from utils import _zscore, preprocess_viewcor_one_item


class DatasetPKL_ViewCor_Pos(Dataset):
    """
    Dataset for loading preprocessed PKL files.
    Each file contains a list of (data[T,>=8], meta{label,...}) objects.
    
    Returns:
        x: (C, L) tensor - features
        y: int - label
        car: int - car identifier
    """
    
    def __init__(self, file_paths, mean=None, std=None, seq_len=128):
        self.samples = []
        self.labels = []
        self.car_list = []
        self.mile_list = []

        L = int(seq_len)

        for path in file_paths:
            arr = np.load(path, allow_pickle=True)
            for item in arr:
                X, y, car_id = preprocess_viewcor_one_item(item)
                self.samples.append(X)
                self.labels.append(y)
                self.car_list.append(car_id)

        if len(self.samples) == 0:
            raise RuntimeError("No samples extracted. Check seq_len or input files.")

        self.samples = np.stack(self.samples, axis=0)  # (N, L, C)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.car_list = np.array(self.car_list, dtype=np.int64)

        self.mu = np.nanmean(self.samples, axis=(0, 1), keepdims=True)
        self.sd = np.nanstd(self.samples, axis=(0, 1), keepdims=True)

        if (mean is not None) and (std is not None):
            self.samples = (self.samples - mean) / (std + 1e-10)
        else:
            self.samples = _zscore(self.samples)

        self.in_ch = self.samples.shape[-1]
        self.seq_len = L

    def get_mean_std(self):
        """Return mean and std for normalization."""
        return self.mu, self.sd

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xw = self.samples[idx]  # (L, C)
        y = self.labels[idx]
        car = self.car_list[idx]
        x = torch.from_numpy(xw).transpose(0, 1).contiguous()  # (C, L)
        y = torch.tensor(int(y), dtype=torch.long)
        car = torch.tensor(int(car), dtype=torch.long)

        return x, y, car
