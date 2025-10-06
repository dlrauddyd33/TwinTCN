# TwinTCN: Correlation-Gated Temporal Convolutions with Twin Encoders

TCN-based Autoencoder for battery anomaly detection using multi-brand vehicle data.

## Overview

This project implements a Temporal Convolutional Network (TCN) autoencoder for detecting anomalies in battery management systems. The model uses voltage, current, temperature, and SOC data from electric vehicles to identify abnormal battery behavior.

## Project Structure

```
TwinTCN/
├── Dataset_split.py          # Data preprocessing and splitting
├── utils.py                  # Utility functions (seed, normalization, etc.)
├── dataset.py                # PyTorch Dataset class
├── data_split.py             # 5-fold cross-validation utilities
├── correlation.py            # Correlation analysis functions
├── models.py                 # Neural network models (TCN, Encoder, Decoder)
├── train.py                  # Training script with 5-fold CV
├── evaluate.py               # Evaluation script (F1, Precision, Recall)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For Google Colab:
```python
!pip install -r requirements.txt
```

### 2. Download Dataset
```
wget --content-disposition \
--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36" \
https://figshare.com/ndownloader/articles/23659323/versions/1
unzip -qq "23659323.zip" -d "./"
for name_b in battery_brand1.tar.gz battery_brand2.tar.gz battery_brand3.tar.gz; do
    tar -xzvf "$name_b" -C "./"
done

```

For Google Colab:
```
!wget --content-disposition --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"  https://figshare.com/ndownloader/articles/23659323/versions/1
!unzip -qq "23659323.zip" -d "./"
for name_b in ["battery_brand1.tar.gz","battery_brand2.tar.gz","battery_brand3.tar.gz"]:
    !tar -xzvf $name_b -C "./"
```

This will download and extract:
- `battery_brand1.tar.gz`
- `battery_brand2.tar.gz`
- `battery_brand3.tar.gz`

**Note:** Requires `wget`, `unzip`, and `tar` commands. On Windows, use WSL or Git Bash.

### 3. Preprocess Data

```
python Dataset_split.py
```

For Google Colab:
```
!python Dataset_split.py
```

This creates:
- `five_fold_utils/battery_brand1/` - Car-level PKL files
- `five_fold_utils/battery_brand2/` - Car-level PKL files
- `five_fold_utils/battery_brand3/` - Car-level PKL files
- `five_fold_utils/ind_odd_dict1.npz.npy` - IND/OOD splits for brand1
- `five_fold_utils/ind_odd_dict2.npz.npy` - IND/OOD splits for brand2
- `five_fold_utils/ind_odd_dict3.npz.npy` - IND/OOD splits for brand3

### 4. Train Model

```
python train.py --brand_dir ./five_fold_utils/battery_brand1 \
                --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                --fold_index 0 \
                --run_all_folds False
```
For Google Colab:
**Single fold:**
```
!python train.py --brand_dir ./five_fold_utils/battery_brand1 \
                --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                --fold_index 0 \
                --run_all_folds False
```

**All 5 folds:**
```
python train.py --brand_dir ./five_fold_utils/battery_brand1 \
                --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                --run_all_folds True
```
For Google Colab:
```
!python train.py --brand_dir ./five_fold_utils/battery_brand1 \
                --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                --run_all_folds True
```

## Data Format

### Input Features (7 channels)
1. **v_diff** - Voltage difference (vmax - vmin)
2. **vmin** - Minimum voltage
3. **soc** - State of Charge
4. **t_diff** - Temperature difference (tmax - tmin)
5. **tmin** - Minimum temperature
6. **cur** - Current
7. **mileage** - Vehicle mileage

### Data Structure
- **Sequence length:** 128 time steps
- **Input shape:** (Batch, 7, 128)
- **Labels:** 0 (normal), 1 (fault)

## Model Architecture

### TCN Autoencoder (TCNAE)
- **Encoder 1:** Processes voltage features (3 channels)
- **Encoder 2:** Processes temperature + current features (4 channels)
- **Decoder:** Reconstructs original 7-channel input
- **Loss:** MSE reconstruction + InfoNCE contrastive loss

### Key Components
- **TCN Blocks:** Dilated causal convolutions with residual connections
- **Levels:** 6 (default), dilation rates: 1, 2, 4, 8, 16, 32
- **Receptive Field:** **253 time steps** for k=3, L=6 (two convs per block)
- **Hidden Dimension:** 64 (default)

## Training Configuration

### Brand-Specific Optimal Hyperparameters

Each battery brand has different optimal hyperparameters:

| Brand | atcn_levels | atcn_model_ch | Dataset Path |
|-------|-------------|---------------|--------------|
| Brand 1 | 6 | 64 | `./five_fold_utils/battery_brand1` |
| Brand 2 | 5 | 32 | `./five_fold_utils/battery_brand2` |
| Brand 3 | 4 | 64 | `./five_fold_utils/battery_brand3` |

**Example usage for each brand:**

```
# Brand 1
python train.py --atcn_levels 6 --atcn_model_ch 64 \
                --brand_dir ./five_fold_utils/battery_brand1 \
                --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy

# Brand 2
python train.py --atcn_levels 5 --atcn_model_ch 32 \
                --brand_dir ./five_fold_utils/battery_brand2 \
                --fold_file ./five_fold_utils/ind_odd_dict2.npz.npy

# Brand 3
python train.py --atcn_levels 4 --atcn_model_ch 64 \
                --brand_dir ./five_fold_utils/battery_brand3 \
                --fold_file ./five_fold_utils/ind_odd_dict3.npz.npy
```

For Google Colab:
```
# Brand 1
!python train.py --atcn_levels 6 --atcn_model_ch 64 \
                --brand_dir ./five_fold_utils/battery_brand1 \
                --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy

# Brand 2
!python train.py --atcn_levels 5 --atcn_model_ch 32 \
                --brand_dir ./five_fold_utils/battery_brand2 \
                --fold_file ./five_fold_utils/ind_odd_dict2.npz.npy

# Brand 3
!python train.py --atcn_levels 4 --atcn_model_ch 64 \
                --brand_dir ./five_fold_utils/battery_brand3 \
                --fold_file ./five_fold_utils/ind_odd_dict3.npz.npy
```

### Default Hyperparameters
```python
--batch_size 128
--seq_len 128
--stride 32
--lr 1e-3
--weight_decay 1e-2
--epochs 200
--atcn_model_ch 64
--atcn_levels 6
--atcn_kernel 3
--corr_gate_dropout 0.1
```

### Optimization
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingLR (eta_min=1e-6)
- **Gradient Clipping:** Max norm 1.0
- **Early Stopping:** Patience 10 epochs

## Evaluation Metrics

### Training Metrics
- **Car-level AUC:** Aggregates sample-level reconstruction errors per vehicle
- **Reconstruction Loss:** MSE between input and reconstructed sequences
- **Contrastive Loss:** InfoNCE between voltage and temperature embeddings

### Evaluation Metrics (F1-Score)

After training, use `evaluate.py` to compute threshold-based classification metrics:

**Metrics:**
- **F1-Score:** Harmonic mean of precision and recall
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **Accuracy:** Correct predictions / Total predictions

**Threshold Computation:**
- **Quantile mode (default):** τ = Q_q(scores) where q=0.90
- **Gaussian mode:** τ = μ + k×σ where k=3.0

**Usage:**
```
# Evaluate all 5 folds with quantile threshold (q=0.90)
python evaluate.py --brand_dir ./five_fold_utils/battery_brand1 \
                   --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                   --out_dir ./runs/atcn_cls \
                   --tau_mode quantile \
                   --tau_q 0.90

# Evaluate with Gaussian threshold (μ + 3σ)
python evaluate.py --brand_dir ./five_fold_utils/battery_brand1 \
                   --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                   --out_dir ./runs/atcn_cls \
                   --tau_mode gauss \
                   --tau_k 3.0

# Evaluate single fold
python evaluate.py --brand_dir ./five_fold_utils/battery_brand1 \
                   --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                   --out_dir ./runs/atcn_cls/fold0 \
                   --fold_index 0 \
                   --run_all_folds False
```


For Google Colab:
```
# Evaluate all 5 folds with quantile threshold (q=0.90)
!python evaluate.py --brand_dir ./five_fold_utils/battery_brand1 \
                   --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                   --out_dir ./runs/atcn_cls \
                   --tau_mode quantile \
                   --tau_q 0.90

# Evaluate with Gaussian threshold (μ + 3σ)
!python evaluate.py --brand_dir ./five_fold_utils/battery_brand1 \
                   --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                   --out_dir ./runs/atcn_cls \
                   --tau_mode gauss \
                   --tau_k 3.0

# Evaluate single fold
!python evaluate.py --brand_dir ./five_fold_utils/battery_brand1 \
                   --fold_file ./five_fold_utils/ind_odd_dict1.npz.npy \
                   --out_dir ./runs/atcn_cls/fold0 \
                   --fold_index 0 \
                   --run_all_folds False
```

**Output:**
- Console output with metrics
- `eval_result.txt` saved in each fold directory

## 5-Fold Cross-Validation

The dataset is split at the **car level** to prevent data leakage:
- **IND (In-Distribution):** Normal battery samples, split into 5 folds
- **OOD (Out-of-Distribution):** Abnormal samples, all used for validation

Each fold:
- **Training:** 80% of IND cars
- **Validation:** 20% of IND cars + 100% of OOD cars

## Output Structure

```
runs/atcn_cls/
├── fold0/
│   ├── best.pt          # Best model checkpoint
│   ├── last.pt          # Last epoch checkpoint
│   └── metrics.csv      # Training metrics
├── fold1/
├── fold2/
├── fold3/
└── fold4/
```

### Metrics CSV Format
```
timestamp,epoch,train_loss,val_loss,val_auc,lr
2025-09-30 20:00:00,1,0.123456,0.234567,0.85,0.001
```
