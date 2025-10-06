"""
Correlation computation utilities for multi-scale analysis.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def _corr_uppertri_from_seq(x_seq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute upper triangular correlation coefficients from sequence.
    
    Args:
        x_seq: (B, L, C) tensor (already standardized)
        eps: Small constant for numerical stability
        
    Returns:
        vec: (B, C*(C-1)//2) correlation vector
    """
    assert x_seq.dim() == 3, f"x_seq should be (B,L,C), got {x_seq.shape}"
    device = x_seq.device
    dtype = x_seq.dtype

    B, L, C = x_seq.shape
    x_centered = x_seq - x_seq.mean(dim=1, keepdim=True)  # (B, L, C)
    
    # Covariance
    cov = torch.einsum('blc,bld->bcd', x_centered, x_centered)  # (B, C, C)
    cov = cov / (max(L - 1, 1) + eps)
    
    # Standard deviation
    d = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(eps))  # (B, C)
    denom = torch.einsum('bc,bd->bcd', d, d)  # (B, C, C)
    corr = cov / denom
    
    # Upper triangular vectorization
    idx_i, idx_j = torch.triu_indices(C, C, offset=1, device=device)
    vec = corr[:, idx_i, idx_j]  # (B, P)
    return vec.to(dtype)


def compute_multiscale_corr_seq(x_bcl: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute multi-scale correlation sequences.
    
    Args:
        x_bcl: (B, C, L) tensor
        
    Returns:
        Dictionary with keys 'L', 'L2', 'L4', 'L8' containing correlation sequences
        Each value is (B, n_chunks, P) where P = C*(C-1)//2
    """
    assert x_bcl.dim() == 3, f"x_bcl should be (B,C,L), got {x_bcl.shape}"
    device = x_bcl.device
    dtype = x_bcl.dtype

    B, C, L = x_bcl.shape
    x_blc = x_bcl.permute(0, 2, 1)  # (B, L, C)
    out: Dict[str, torch.Tensor] = {}

    for k, name in [(1, 'L'), (2, 'L2'), (4, 'L4'), (8, 'L8')]:
        seg = max(L // k, 2)
        n_chunks = max(L // seg, 1)
        vec_list = []
        for i in range(n_chunks):
            s0, s1 = i * seg, (i + 1) * seg
            if s1 > L:
                break
            xs = x_blc[:, s0:s1, :]
            vec = _corr_uppertri_from_seq(xs)
            vec_list.append(vec)
        if len(vec_list) == 0:
            xs = x_blc[:, -seg:, :]
            vec_list = [_corr_uppertri_from_seq(xs)]
            n_chunks = 1
        seq = torch.stack(vec_list, dim=1).to(device=device, dtype=dtype)  # (B, n_chunks, P)
        out[name] = seq
    return out


def corr_chunks_for_rf(x_bcl: torch.Tensor, seg: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute correlation chunks for receptive field alignment.
    
    Args:
        x_bcl: (B, C, L) tensor
        seg: Segment length (receptive field size)
        eps: Small constant for numerical stability
        
    Returns:
        (B, n_chunks, P) correlation tensor where P = C*(C-1)//2
    """
    assert x_bcl.dim() == 3, f"x_bcl should be (B,C,L), got {x_bcl.shape}"
    device = x_bcl.device
    dtype = x_bcl.dtype

    B, C, L = x_bcl.shape
    seg = max(int(seg), 2)
    n_chunks = max(L // seg, 1)

    x_blc = x_bcl.permute(0, 2, 1)  # (B, L, C)
    vec_list = []
    for i in range(n_chunks):
        s0, s1 = i * seg, (i + 1) * seg
        if s1 > L:
            break
        xs = x_blc[:, s0:s1, :]  # (B, seg, C)
        vec = _corr_uppertri_from_seq(xs, eps=eps)  # (B, P)
        vec_list.append(vec)

    if len(vec_list) == 0:
        xs = x_blc[:, -seg:, :]
        vec_list = [_corr_uppertri_from_seq(xs, eps=eps)]
        n_chunks = 1

    out = torch.stack(vec_list, dim=1).to(device=device, dtype=dtype)  # (B, n_chunks, P)
    return out
