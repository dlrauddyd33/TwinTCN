"""
Training script for TCN Autoencoder on battery data.
"""

import os
import time
import copy
import argparse
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from utils import set_seed, count_params
from dataset import DatasetPKL_ViewCor_Pos
from data_split import build_five_fold_filelists
from models import TCNAE, CrossViewInfoNCE


# ============================
# Configuration
# ============================

@dataclass
class TrainConfig:
    """Training configuration parameters."""
    # Data
    train_dir: str = ""
    valid_dir: str = ""
    brand_dir: Optional[str] = None
    data_format: str = 'pkl'
    seq_len: int = 128
    batch_size: int = 64
    num_workers: int = 0
    
    # Model (ATCN)
    atcn_model_ch: int = 128
    atcn_levels: int = 5
    atcn_kernel: int = 3
    atcn_dropout: float = 0.1
    corr_gate_dropout: float = 0.1

    # Optimization
    lr: float = 1e-2
    weight_decay: float = 1e-1
    epochs: int = 20
    seed: int = 42
    
    # Checkpoint
    out_dir: str = './runs/atcn_cls'
    resume: Optional[str] = None
    
    # 5-fold cross-validation
    five_fold: bool = False
    fold_index: int = 0
    folds_seed: int = 42
    fold_file: Optional[str] = None
    run_all_folds: bool = False

    #evaluation
    tau_mode: str = 'quantile'
    tau_q: float = 0.90
    tau_k: float = 3.0

# ============================
# Data Loading
# ============================

def build_loaders(cfg: TrainConfig):
    """Build train and validation data loaders."""
    tr_files, va_files = build_five_fold_filelists(
        brand_dir=cfg.brand_dir,
        fold_index=cfg.fold_index,
        folds_seed=cfg.seed,
        fold_file=cfg.fold_file,
    )

    if len(tr_files) == 0 or len(va_files) == 0:
        raise RuntimeError('No PKL files found.')
    
    train_ds = DatasetPKL_ViewCor_Pos(tr_files, seq_len=cfg.seq_len)
    train_mean, train_std = train_ds.get_mean_std()
    valid_ds = DatasetPKL_ViewCor_Pos(va_files, mean=train_mean, std=train_std,
                                      seq_len=cfg.seq_len)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)

    return train_dl, valid_dl, train_ds.in_ch


# ============================
# Evaluation
# ============================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, 
             out_dir: str, epoch: int):
    """
    Evaluate model on validation set.
    
    Returns:
        auc: Area under ROC curve (car-level)
        avg_loss: Average loss
    """
    mse = nn.MSELoss()
    infonce = CrossViewInfoNCE()
    model.eval()
    
    y_all, car_all, logit1_all = [], [], []
    total_loss = 0
    n_batches = 0
    
    for x, y, car in loader:
        B = x.size(0)
        n_batches += B
        x = x.to(device).float()  # (B, C, L)
        y = y.to(device)

        x_recon, z, z1, z2 = model(x)
        score = torch.sum((x_recon.reshape(B, -1) - x.reshape(B, -1)) ** 2, dim=1)
        loss = mse(x_recon, x) + infonce(z1, z2)
        total_loss += loss.item()
        logit1_all.append(score.detach().cpu())

        y_all.append(y.cpu())
        car_all.append(car.cpu())

    car_alls = torch.cat(car_all).numpy().astype(int)
    y_true = torch.cat(y_all).numpy().astype(int)
    logit1_all = torch.cat(logit1_all).numpy()
    car_ids = car_alls

    # Car-level aggregation: average reconstruction error per car
    car_to_scores: Dict[int, list] = {}
    car_to_labels: Dict[int, list] = {}
    for s, t, c in zip(logit1_all, y_true, car_ids):
        car_to_scores.setdefault(c, []).append(float(s))
        car_to_labels.setdefault(c, []).append(int(t))

    cars = sorted(car_to_scores.keys())
    agg_scores = np.array([np.mean(car_to_scores[c]) for c in cars], dtype=float)
    agg_labels = np.array([int(round(np.mean(car_to_labels[c]))) for c in cars], dtype=int)
    auc = float(roc_auc_score(agg_labels, agg_scores))

    return auc, total_loss / n_batches


# ============================
# Training
# ============================

def train_one(cfg: TrainConfig):
    """Train model for one fold."""
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM {prop.total_memory/1e9:.1f} GB")

    train_dl, valid_dl, in_ch = build_loaders(cfg)
    
    # Dynamic Weight Averaging (DWA) preparation
    T = getattr(cfg, 'dwa_T', 2.0)
    eps = 1e-12
    hist_mse = []
    hist_cont = []
    lam_recon = 1.0
    lam_cont = 1.0

    # Create model
    model = TCNAE(
        in_dim=in_ch,
        hidden=cfg.atcn_model_ch,
        out_ch=in_ch,
        levels=cfg.atcn_levels,
        kernel_size=cfg.atcn_kernel,
        dropout=cfg.corr_gate_dropout
    ).to(device)

    print(f"Model params: {count_params(model)/1e6:.2f}M | in_ch={in_ch}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)

    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, 'metrics.csv')
    if not os.path.isfile(metrics_csv):
        with open(metrics_csv, 'w') as f:
            f.write('timestamp,epoch,train_loss,val_loss,val_auc,lr\n')

    best_loss = 1e9
    best_auc = 0
    patient = 0

    mse = nn.MSELoss()
    infoce = CrossViewInfoNCE()

    for ep in range(cfg.epochs):
        # Dynamic Weight Averaging
        if len(hist_mse) >= 2 and len(hist_cont) >= 2:
            r_mse = hist_mse[-1] / (hist_mse[-2] + eps)
            r_cont = hist_cont[-1] / (hist_cont[-2] + eps)
            w = torch.softmax(torch.tensor([r_mse, r_cont], dtype=torch.float32) / T, dim=0)
            lam_recon = float(w[0].item())
            lam_cont = float(w[1].item())
        else:
            lam_recon = 1.0
            lam_cont = 1.0

        model.train()
        t0 = time.time()
        tot_loss = 0.0
        n_samples = 0
        sum_mse = 0.0
        sum_cont = 0.0
        n_batches = 0
        
        for x, y, car_id in train_dl:
            x = x.to(device).float()  # (B, C, L)
            car_id = car_id.to(device)
            y = y.to(device)

            recon, z, z1, z2 = model(x)
            mse_loss = mse(recon, x)
            cont_loss = infoce(z1, z2)
            loss = lam_recon * mse_loss + lam_cont * cont_loss
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            sum_mse += mse_loss.item()
            sum_cont += cont_loss.item()
            B = x.size(0)
            tot_loss += loss.item()
            n_samples += B
            n_batches += 1

        sched.step()
        
        # Record epoch average losses
        epoch_mse = sum_mse / max(n_batches, 1)
        epoch_cont = sum_cont / max(n_batches, 1)
        hist_mse.append(epoch_mse)
        hist_cont.append(epoch_cont)
        
        val_auc, val_loss = evaluate(model, valid_dl, device, out_dir, ep)
        dt = time.time() - t0
        lr = sched.get_last_lr()[0]
        print(f"Epoch {ep+1:03d}/{cfg.epochs} | train_loss {tot_loss/n_samples:.5f} | "
              f"val_loss {val_loss:.5f} | val_auc {val_auc:.5f} | lr {lr:.2e} | {dt:.1f}s "
              f"(λ_r {lam_recon:.3f}, λ_c {lam_cont:.3f})")

        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(metrics_csv, 'a') as f:
            f.write(f"{ts},{ep+1},{tot_loss/n_samples:.6f},{val_loss:.6f},{val_auc:.6f},{lr:.8f}\n")

        # Save checkpoint
        torch.save({'epoch': ep+1, 'model': model.state_dict(), 'optim': opt.state_dict(),
                    'sched': sched.state_dict(), 'val_loss': val_loss, 'cfg': cfg.__dict__},
                   os.path.join(out_dir, 'last.pt'))

        improved = (val_auc > best_auc)
        if improved:
            best_auc = val_auc
            best_loss = val_loss
            torch.save({'epoch': ep+1, 'model': model.state_dict(), 'optim': opt.state_dict(),
                        'sched': sched.state_dict(), 'best_loss': best_loss, 'cfg': cfg.__dict__},
                       os.path.join(out_dir, 'best.pt'))
            print(f"Saved best checkpoint: best_auc={best_auc:.4f}")
            patient = 0
        else:
            patient += 1
            if patient > 10:
                print(f"[epoch {ep+1}] early stop (patience 10)")
                break


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

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))
    
    if cfg.five_fold and cfg.run_all_folds:
        for k in range(5):
            cfg_k = copy.deepcopy(cfg)
            cfg_k.fold_index = k
            cfg_k.out_dir = os.path.join(cfg.out_dir, f"fold{k}")
            train_one(cfg_k)
    else:
        train_one(cfg)
