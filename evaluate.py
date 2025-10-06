"""
Evaluation script for computing F1-score, Precision, Recall, and Accuracy.
Uses threshold-based classification on car-level aggregated reconstruction errors.
"""

import os
import copy
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils import set_seed, count_params
from dataset import DatasetPKL_ViewCor_Pos
from data_split import build_five_fold_filelists
from models import TCNAE
from train import TrainConfig, build_loaders


# ============================
# Helper Functions
# ============================

def _aggregate_by_car(scores: np.ndarray, labels: np.ndarray, cars: np.ndarray):
    """
    Aggregate scores and labels by car ID.
    
    Args:
        scores: Window-level reconstruction errors
        labels: Window-level labels
        cars: Window-level car IDs
        
    Returns:
        car_ids: List of car IDs
        agg_scores: Car-level average scores
        agg_labels: Car-level majority vote labels
    """
    d_scores: Dict[int, List[float]] = {}
    d_labels: Dict[int, List[int]] = {}
    for s, t, c in zip(scores, labels, cars):
        d_scores.setdefault(int(c), []).append(float(s))
        d_labels.setdefault(int(c), []).append(int(t))
    car_ids = sorted(d_scores.keys())
    agg_scores = np.array([np.mean(d_scores[c]) for c in car_ids], dtype=np.float64)
    # Majority vote (round in case of tie)
    agg_labels = np.array([int(round(np.mean(d_labels[c]))) for c in car_ids], dtype=np.int64)
    return car_ids, agg_scores, agg_labels


@torch.no_grad()
def _collect_window_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Collect reconstruction error scores for each window.
    
    Args:
        model: Trained TCNAE model
        loader: DataLoader
        device: torch device
        
    Returns:
        scores: Window-level reconstruction errors
        labels: Window-level labels
        cars: Window-level car IDs
    """
    model.eval()
    mse = nn.MSELoss(reduction='none')
    scores, labels, cars = [], [], []
    for x, y, car in loader:
        x = x.to(device).float()  # (B, C, L)
        y = y.numpy().astype(np.int64)
        car = car.numpy().astype(np.int64)
        recon, z, z1, z2 = model(x)  # recon: (B, C, L)
        # Window-level reconstruction error (MSE sum)
        diff = (recon - x).reshape(x.size(0), -1)
        s = torch.sum(diff * diff, dim=1).detach().cpu().numpy()
        scores.append(s.astype(np.float64))
        labels.append(y)
        cars.append(car)
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    cars = np.concatenate(cars, axis=0)
    return scores, labels, cars


# ============================
# Threshold Computation
# ============================

@torch.no_grad()
def compute_threshold_from_train(model: nn.Module,
                                 train_loader: DataLoader,
                                 device: torch.device,
                                 mode: str = 'quantile',
                                 tau_q: float = 0.99,
                                 k: float = 3.0) -> float:
    """
    Compute threshold from training data (normal samples).
    
    Args:
        model: Trained model
        train_loader: Training data loader (normal samples only)
        device: torch device
        mode: 'quantile' or 'gauss'
        tau_q: Quantile value for threshold (0.5~0.9999)
        k: Multiplier for Gaussian mode (mu + k*sigma)
        
    Returns:
        tau: Threshold value
    """
    scores, labels, cars = _collect_window_scores(model, train_loader, device)
    # Aggregate by car (training data should contain only normal samples)
    _, agg_scores, _ = _aggregate_by_car(scores, labels, cars)

    if mode == 'gauss':
        mu = float(np.mean(agg_scores))
        sd = float(np.std(agg_scores, ddof=1))
        tau = mu + k * sd
    else:
        # Default: quantile
        q = float(np.clip(tau_q, 0.5, 0.9999))
        tau = float(np.quantile(agg_scores, q=q))
    return tau


# ============================
# Evaluation with Threshold
# ============================

@torch.no_grad()
def evaluate_f1_with_threshold(model: nn.Module,
                                valid_loader: DataLoader,
                                device: torch.device,
                                tau: float) -> Dict[str, float]:
    """
    Evaluate F1-score, Precision, Recall, and Accuracy using threshold.
    
    Args:
        model: Trained model
        valid_loader: Validation data loader
        device: torch device
        tau: Threshold value
        
    Returns:
        Dictionary containing evaluation metrics
    """
    scores, labels, cars = _collect_window_scores(model, valid_loader, device)
    _, agg_scores, agg_labels = _aggregate_by_car(scores, labels, cars)
    y_true = agg_labels
    y_pred = (agg_scores > tau).astype(np.int64)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    supp_pos = int(np.sum(y_true == 1))
    supp_neg = int(np.sum(y_true == 0))
    return {
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'accuracy': float(acc),
        'support_pos': supp_pos,
        'support_neg': supp_neg,
        'threshold': float(tau)
    }


# ============================
# Load Best Model and Evaluate
# ============================

def load_best_and_eval_f1(cfg: TrainConfig,
                           tau_mode: str = 'quantile',
                           tau_q: float = 0.99,
                           tau_k: float = 3.0,
                           best_path: Optional[str] = None):
    """
    Load best checkpoint, compute threshold from training data, and evaluate F1-score.
    
    Args:
        cfg: Training configuration
        tau_mode: Threshold mode ('quantile' or 'gauss')
        tau_q: Quantile value for threshold
        tau_k: Multiplier for Gaussian mode
        best_path: Path to best checkpoint (default: cfg.out_dir/best.pt)
        
    Returns:
        Dictionary containing evaluation results
    """
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl, valid_dl, in_ch = build_loaders(cfg)
    model = TCNAE(
        in_dim=in_ch,
        hidden=cfg.atcn_model_ch,
        out_ch=in_ch,
        levels=cfg.atcn_levels,
        kernel_size=cfg.atcn_kernel,
        dropout=cfg.corr_gate_dropout
    ).to(device)

    if best_path is None:
        best_path = os.path.join(cfg.out_dir, 'best.pt')
    ckpt = torch.load(best_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # Compute threshold from training data (car-level aggregated scores)
    tau = compute_threshold_from_train(model, train_dl, device,
                                       mode=tau_mode, tau_q=tau_q, k=tau_k)

    # Evaluate F1 on validation data (car-level)
    res = evaluate_f1_with_threshold(model, valid_dl, device, tau)

    # Console output
    print("==== Threshold-based Evaluation (car-level) ====")
    print(f"tau_mode={tau_mode}, tau_q={tau_q}, tau_k={tau_k}")
    print(f"threshold τ = {res['threshold']:.6f}")
    print(f"F1={res['f1']:.4f} | Precision={res['precision']:.4f} | Recall={res['recall']:.4f} | Acc={res['accuracy']:.4f} "
          f"| support(pos)={res['support_pos']} / (neg)={res['support_neg']}")

    # Save results to text file
    os.makedirs(cfg.out_dir, exist_ok=True)
    txt_path = os.path.join(cfg.out_dir, "eval_result.txt")
    with open(txt_path, "w") as f:
        f.write("==== Threshold-based Evaluation (car-level) ====\n")
        f.write(f"tau_mode={tau_mode}, tau_q={tau_q}, tau_k={tau_k}\n")
        f.write(f"threshold τ = {res['threshold']:.6f}\n")
        f.write(
            f"F1={res['f1']:.4f} | Precision={res['precision']:.4f} | "
            f"Recall={res['recall']:.4f} | Acc={res['accuracy']:.4f} | "
            f"support(pos)={res['support_pos']} / (neg)={res['support_neg']}\n"
        )

    print(f"[INFO] Results saved to {txt_path}")
    return res


# ============================
# Main Entry Point
# ============================

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # IO
    p.add_argument('--train_dir', type=str, default='')
    p.add_argument('--valid_dir', type=str, default='')
    p.add_argument('--brand_dir', type=str, default='./five_fold_utils2/battery_brand1')
    p.add_argument('--out_dir', type=str, default='./runs/atcn_cls')
    # Data
    p.add_argument('--data_format', type=str, choices=['pkl'], default='pkl')
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--stride', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)
    # ATCN model
    p.add_argument('--atcn_model_ch', type=int, default=64)
    p.add_argument('--atcn_levels', type=int, default=6)
    p.add_argument('--atcn_kernel', type=int, default=3)
    p.add_argument('--atcn_dropout', type=float, default=0.0)
    p.add_argument('--corr_gate_dropout', type=float, default=0.1)
    # Optimization
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    # Checkpoint
    p.add_argument('--resume', type=str, default=None)
    # 5-fold
    p.add_argument('--five_fold', type=bool, default=True)
    p.add_argument('--fold_index', type=int, default=0)
    p.add_argument('--folds_seed', type=int, default=42)
    p.add_argument('--fold_file', type=str, default='./five_fold_utils2/ind_odd_dict1.npz.npy')
    p.add_argument('--run_all_folds', type=bool, default=True)
    # Threshold & F1 eval options
    p.add_argument('--tau_mode', type=str, default='quantile', choices=['quantile', 'gauss'],
                   help='Thresholding mode: quantile (default) or gauss(mu+k*sigma)')
    p.add_argument('--tau_q', type=float, default=0.90, help='Quantile q for threshold (0.5~0.9999).')
    p.add_argument('--tau_k', type=float, default=3.0, help='k in mu+k*sigma when tau_mode=gauss.')

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))
    
    if cfg.five_fold and cfg.run_all_folds:
        for k in range(5):
            cfg_k = copy.deepcopy(cfg)
            cfg_k.fold_index = k
            cfg_k.out_dir = os.path.join(cfg.out_dir, f"fold{k}")
            _ = load_best_and_eval_f1(cfg_k,
                                       tau_mode=args.tau_mode,
                                       tau_q=args.tau_q,
                                       tau_k=args.tau_k,
                                       best_path=os.path.join(cfg_k.out_dir, 'best.pt'))
    else:
        _ = load_best_and_eval_f1(cfg,
                                   tau_mode=args.tau_mode,
                                   tau_q=args.tau_q,
                                   tau_k=args.tau_k)
