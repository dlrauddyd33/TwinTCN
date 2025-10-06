"""
Five-fold cross-validation data splitting utilities.
"""

import os
import re
import glob
import numpy as np
from typing import Optional, List, Tuple


CAR_RE = re.compile(r"car_(\d+)\.pkl$")


def get_car_label_from_pkl(pkl_path: str) -> int:
    """Extract label from a PKL file."""
    arr = np.load(pkl_path, allow_pickle=True)
    for item in arr:
        meta = item[1]
        if isinstance(meta, dict) and 'label' in meta:
            val = meta['label']
            if isinstance(val, (list, np.ndarray)):
                val = val[0]
            try:
                return int(val)
            except Exception:
                pass
    return 0


def load_fold_hint(fold_file: Optional[str]):
    """
    Load fold hint file containing IND/OOD car splits.
    
    Args:
        fold_file: Path to .npz file containing car splits
        
    Returns:
        Dictionary with ind/ood car lists or None
    """
    if fold_file is None:
        return None
    d = np.load(fold_file, allow_pickle=True).item()
    
    def _tolist(x):
        return [int(v) for v in list(np.array(x).ravel())]
    
    for a, b in [("ind_sorted", "ood_sorted"), ("ind", "ood"), 
                 ("normal", "abnormal"), ("norm", "abnorm"), 
                 ("ind_list", "ood_list")]:
        if a in d and b in d:
            A = _tolist(d[a])
            B = _tolist(d[b])
            if A is not None and B is not None:
                return {a: A, b: B}
    return None


def build_five_fold_filelists(brand_dir: str, fold_index: int, folds_seed: int,
                               fold_file: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Build train/validation file lists for 5-fold cross-validation.
    
    Args:
        brand_dir: Directory containing car_*.pkl files
        fold_index: Fold index (0-4)
        folds_seed: Random seed for reproducibility
        fold_file: Path to fold hint file
        
    Returns:
        train_files: List of training file paths
        valid_files: List of validation file paths
    """
    files = sorted(glob.glob(os.path.join(brand_dir, "car_*.pkl")))

    if len(files) == 0:
        raise RuntimeError(f"No PKL files under {brand_dir}")

    hint = load_fold_hint(fold_file)
    if hint is None:
        raise RuntimeError("fold_file is required or does not contain valid keys (ind_sorted/ood_sorted, etc.).")

    def car_from_path(p):
        m = CAR_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else None

    # Extract ind/ood lists from hint
    key_pairs = [("ind_sorted", "ood_sorted"), ("ind", "ood"), 
                 ("normal", "abnormal"), ("norm", "abnorm"), 
                 ("ind_list", "ood_list")]
    ind_list, ood_list = None, None
    for a, b in key_pairs:
        if a in hint and b in hint:
            ind_list = list(map(int, hint[a]))
            ood_list = list(map(int, hint[b]))
            break
    if ind_list is None or ood_list is None:
        raise RuntimeError("Could not find ind/ood lists in fold_file.")

    if not (0 <= fold_index < 5):
        raise ValueError("--fold_index must be between 0 and 4")

    # 5-fold split: split IND cars evenly, all OOD cars go to validation
    n = len(ind_list)
    k0 = (fold_index * n) // 5
    k1 = ((fold_index + 1) * n) // 5
    val_cars = ind_list[k0:k1] + ood_list
    train_cars = ind_list[:k0] + ind_list[k1:]

    train_files = [p for p in files if car_from_path(p) in train_cars]
    valid_files = [p for p in files if car_from_path(p) in val_cars]
    return train_files, valid_files
