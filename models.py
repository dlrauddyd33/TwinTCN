"""
Neural network models: TCN blocks, Encoder, Decoder, and Autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ============================
# TCN Blocks
# ============================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding."""
    
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.pad, dilation=dilation, bias=bias)
    
    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        return y


class TCNBlock(nn.Module):
    """Single TCN block with residual connection."""
    
    def __init__(self, ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, ch)
        self.norm2 = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
    
    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + r


class TCNStack(nn.Module):
    """
    Stack of TCN blocks with multiple dilation levels.
    
    Args:
        in_ch: Input channels
        model_ch: Model hidden channels
        levels: Number of dilation levels
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(self, in_ch, model_ch, levels=5, kernel_size=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, model_ch, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNBlock(model_ch, kernel_size, dilation=2**i, dropout=dropout) 
            for i in range(levels)
        ])
        self.fuse = nn.Linear(model_ch * levels, model_ch)
        self.out_norm = nn.LayerNorm(model_ch)
        self.levels = levels

    def forward(self, x_btci, return_per_level: bool = False):
        """
        Args:
            x_btci: (B, T, C) input tensor
            return_per_level: If True, return list of per-level features
            
        Returns:
            If return_per_level: List[(B, L, H)] * levels
            Else: (B, T, H) fused features
        """
        x = x_btci.transpose(1, 2)  # (B, C_in, T)
        h0 = self.in_proj(x)  # (B, H, T)
        feats = []
        h = h0
        for b in self.blocks:
            h = b(h)  # (B, H, T)
            feats.append(h.transpose(1, 2))  # (B, T, H)

        if return_per_level:
            return feats  # List[(B, L, H)] * levels

        h_cat = torch.cat(feats, dim=-1)  # (B, T, H*L)
        h = self.fuse(h_cat)  # (B, T, H)
        h = self.out_norm(h)
        return h


# ============================
# Contrastive Loss
# ============================

class CrossViewInfoNCE(nn.Module):
    """InfoNCE contrastive loss for cross-view learning."""
    
    def __init__(self, temperature: float = 0.07,
                 learnable: bool = True,
                 tau_bounds=(0.02, 0.5)):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(temperature)))
            self.tau_bounds = tau_bounds
        else:
            self.register_buffer('tau_buf', torch.tensor(float(temperature)))

    def _tau(self):
        if self.learnable:
            lo, hi = self.tau_bounds
            return self.log_tau.exp().clamp(min=lo, max=hi)
        return self.tau_buf

    def forward(self, z_a, z_b):
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        tau = self._tau()
        logits = z_a @ z_b.t() / tau
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_ab + loss_ba)


# ============================
# Encoder (GateTCN)
# ============================

class GateTCN(nn.Module):
    """
    TCN-based encoder with optional correlation gating.
    
    Args:
        in_dim: Input feature dimension
        model_ch: Model hidden channels
        tcn_levels: Number of TCN levels
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        use_corr: Whether to use correlation features (legacy)
        corr_hidden: Hidden size for correlation LSTM
        corr_layers: Number of LSTM layers
        corr_dropout: Dropout for correlation LSTM
        use_corr_align: Whether to use RF-aligned correlation gating
        corr_gate_dropout: Dropout for correlation gating
    """
    
    @staticmethod
    def receptive_field(kernel_size=3, levels=5):
        return 1 + 2 * (kernel_size - 1) * (2**levels - 1)

    @staticmethod
    def rf_per_level(kernel_size=3, levels=5) -> List[int]:
        return [1 + 2 * (kernel_size - 1) * (2**(i + 1) - 1) for i in range(levels)]

    def __init__(self, in_dim=6, model_ch=128, tcn_levels=5, kernel_size=3,
                 dropout=0.1,
                 use_corr: bool = False,
                 corr_hidden: int = 64, corr_layers: int = 1, corr_dropout: float = 0.0,
                 use_corr_align: bool = True, corr_gate_dropout: float = 0.0):
        super().__init__()
        self.use_corr = use_corr
        self.use_corr_align = use_corr_align
        self.model_ch = model_ch

        # TCN
        self.tcn = TCNStack(in_dim, model_ch, tcn_levels, kernel_size, dropout)
        self.tcn_levels = tcn_levels
        self.kernel_size = kernel_size
        self.rf_list = self.rf_per_level(kernel_size, tcn_levels)

        # Correlation gating (optional, not used in current implementation)
        C_dummy = in_dim
        P = (C_dummy * (C_dummy - 1)) // 2
        self.P = P
        if self.use_corr_align:
            self.corr2h_per_level = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(P),
                    nn.Linear(P, model_ch),
                    nn.ReLU(),
                    nn.Dropout(corr_gate_dropout)
                ) for _ in range(tcn_levels)
            ])
            self.gate_per_level = nn.ModuleList([
                nn.Conv1d(model_ch * 2, 1, kernel_size=1) for _ in range(tcn_levels)
            ])
            self.fuse_levels = nn.Linear(model_ch * tcn_levels, model_ch)
            self.post_norm = nn.LayerNorm(model_ch)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(model_ch),
            nn.Linear(model_ch, model_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_ch // 2, model_ch // 4)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_bcl: torch.Tensor, return_corr: bool = False):
        """
        Args:
            x_bcl: (B, C, L) input tensor
            return_corr: Whether to return correlation debug info
            
        Returns:
            logits: (B, H//4) output logits
        """
        assert x_bcl.dim() == 3, f"x_bcl should be (B,C,L), got {x_bcl.shape}"
        B, C, L = x_bcl.shape

        # Get per-level features from TCN
        feats_list: List[torch.Tensor] = self.tcn(x_bcl.transpose(1, 2), return_per_level=True)

        # Fuse levels
        h_cat = torch.cat(feats_list, dim=-1)  # (B, L, H*levels)
        h_seq = self.fuse_levels(h_cat)  # (B, L, H)
        h_seq = self.post_norm(h_seq)  # (B, L, H)
        h = h_seq.transpose(1, 2)  # (B, H, L)
        
        # Pool & head
        h_vec = self.pool(h).squeeze(-1)  # (B, H)
        logits = self.head(h_vec)  # (B, H//4)
        return logits


# ============================
# Decoder
# ============================

class Decoder(nn.Module):
    """
    Reconstructs (B, C_out, T) from latent vector z: (B, H).
    
    Args:
        hidden: Latent dimension H
        out_ch: Output channels C_out
        levels: Number of TCN blocks
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        seq_len: Sequence length T
    """
    
    def __init__(self, hidden: int = 128, out_ch: int = 7, levels: int = 5,
                 kernel_size: int = 3, dropout: float = 0.0, seq_len: int = 128):
        super().__init__()
        self.hidden = hidden
        self.out_ch = out_ch
        self.seq_len = seq_len

        # (B, H) -> (B, H*T) -> reshape -> (B, H, T)
        self.expand = nn.Linear(hidden, hidden * seq_len)

        # Preparation conv
        self.pre = nn.Conv1d(hidden, hidden, kernel_size=1)

        # Dilated TCN stack
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            for i in range(levels)
        ])

        # Normalization + final projection
        self.norm = nn.GroupNorm(1, hidden)
        self.head = nn.Conv1d(hidden, out_ch, kernel_size=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, H) latent vector
            
        Returns:
            y: (B, C_out, T) reconstructed sequence
        """
        assert z.dim() == 2, f"z should be (B,H), got {z.shape}"
        B, H = z.shape
        assert H == self.hidden, f"latent dim mismatch: expected {self.hidden}, got {H}"

        # (B, H) -> (B, H*T) -> (B, H, T)
        x = self.expand(z)  # (B, H*T)
        x = x.view(B, H, self.seq_len)  # (B, H, T)

        # Preparation conv
        x = self.pre(x)  # (B, H, T)

        # TCN stack
        for b in self.blocks:
            x = b(x)  # (B, H, T)

        x = self.norm(x)  # (B, H, T)

        # Channel projection
        y = self.head(x)  # (B, C_out, T)
        return y


# ============================
# Autoencoder
# ============================

class TCNAE(nn.Module):
    """
    TCN-based Autoencoder with dual encoders for voltage and temperature features.
    
    Args:
        in_dim: Input feature dimension
        hidden: Hidden dimension
        out_ch: Output channels (for reconstruction)
        levels: Number of TCN levels
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(self, in_dim: int = 7, hidden: int = 128, out_ch: int = 7,
                 levels: int = 5, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder1 = GateTCN(
            in_dim=in_dim // 2, model_ch=hidden, tcn_levels=levels,
            kernel_size=kernel_size, dropout=dropout
        )
        self.encoder2 = GateTCN(
            in_dim=in_dim // 2 + 1, model_ch=hidden, tcn_levels=levels,
            kernel_size=kernel_size, dropout=dropout
        )
        self.decoder = Decoder(
            hidden=hidden, out_ch=in_dim
        )

        self.proj = nn.Linear(hidden // 2, hidden)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, L) input tensor
            
        Returns:
            out: (B, C, L) reconstructed tensor
            z_: (B, H) projected latent
            z1: (B, H//4) encoder1 output
            z2: (B, H//4) encoder2 output
        """
        # Split voltage and temperature features
        v_x = x[:, :3, :]
        t_x = x[:, 3:, :]
        z1 = self.encoder1(v_x)  # (B, H//4)
        z2 = self.encoder2(t_x)  # (B, H//4)
        z = torch.cat([z1, z2], dim=-1)  # (B, H//2)
        z_ = self.proj(z)  # (B, H)
        out = self.decoder(z_)
        return out, z_, z1, z2
