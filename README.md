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



## Table 1. Training schedule and runtime setup

| Item            | Setting                                                                                 |
|-----------------|-----------------------------------------------------------------------------------------|
| Optimizer       | AdamW (lr = \(1 \times 10^{-3}\), weight decay = \(1 \times 10^{-2}\))                 |
| LR scheduler    | CosineAnnealingLR (\(T_{\max} =\) epochs, \(\eta_{\min} = 1 \times 10^{-6}\))          |
| Epochs          | 200 (max)                                                                              |
| Early stopping  | Stop if *val AUROC* shows no improvement for 10 epochs                                 |
| Checkpointing   | Save on *val AUROC* improvement (best-only)                                            |
| Batch size      | 128                                                                                    |
| Grad clipping   | \(\lVert g \rVert_2 \le 1.0\)                                                           |
| Seed            | 42                                                                                      |
| Environment     | Colab GPU: NVIDIA T4 16 GB, ~2 vCPU, 52 GB RAM                                         |


## Five-fold Protocol and Split Summaries

Tables2 summarize, for each dataset, the 5-fold splits (counts of normal/faulty vehicles in train/validation) and validation length statistics (normal/faulty μ±σ) plus fault ratios/ranges for the unsupervised and supervised settings, respectively.


### Table 2. Per-dataset fold summaries

Lengths are measured in snippets.

#### Unsupervised setting (Train: Normal, Val: Normal+Faulty)

| Dataset  | Tr-N Veh | Val-N Veh | Val-F Veh | Val-N snippets μ±σ / Fault% \[min–max] |
|----------|----------|-----------|-----------|-----------------------------------------|
| Dataset1 | 168      | 33–34     | 30        | 91,942 ± 12,553 / 13.7%–18.2%          |
| Dataset2 | 33       | 6–7       | 16        | 16,592 ± 5,122 / 83.6%–93.6%           |
| Dataset3 | 91       | 18–19     | 9         | 5,565 ± 888 / 20.2%–28.7%              |

#### Supervised setting (Train/Val: Normal+Faulty)

| Dataset  | Tr (N/F) Veh        | Val (N/F) Veh      | Val-N snippets μ±σ      | Val-F snippets μ±σ; Fault% μ±σ \[min–max]           |
|----------|---------------------|--------------------|-------------------------|-----------------------------------------------------|
| Dataset1 | 134–135 / 24        | 33–34 / 6          | 91,942 ± 12,252         | 3,406 ± 1,503; 3.5 ± 1.2 \[1.4–4.7]                 |
| Dataset2 | 26–27 / 12–13       | 6–7 / 3–4          | 16,592 ± 10,427         | 22,257 ± 18,873; 55.0 ± 26.8 \[12.7–86.1]          |
| Dataset3 | 72–73 / 7–8         | 18–19 / 1–2        | 5,565 ± 593             | 355 ± 174; 6.1 ± 2.9 \[1.3–10.1]                   |


### Protocol

**Protocol (vehicle-level, DyAD-style).**  
We use *vehicle-level* five-fold cross-validation (seed = 42) to avoid cross-vehicle leakage.

- **Unsupervised:** train on normal vehicles only; validation includes normal + faulty (OOD).  
- **Supervised:** train/validation on normal + faulty.  
- Segments immediately preceding failures are excluded by the dataset protocol.  


### Notes

- **Abbrev.**  
  Veh: number of vehicles; “snippets”: snippet counts; Tr-N/Val-N/Val-F: train/val normal/fault vehicles.

- **Fault% (validation).**  
  \[
  \text{Fault\%} = \frac{\text{Val\_FaultLen}}{\text{Val\_NormLen} + \text{Val\_FaultLen}}
  \]
  (length-based).

- **Aggregation.**  
  Table&nbsp;2 (top) reports *min–max* across folds; Table&nbsp;2 (bottom) reports *mean ± sd* across folds and also *min–max* in brackets.

- **Indices.**  
  Vehicle-level, non-overlapping train/val per fold; identical indices shared by all baselines.




## F1/Recall/Precision Details

### Table 3. F1/Recall/Precision at quantile thresholds

F1/Recall/Precision at quantile thresholds \( p \in \{0.80, 0.85, 0.90, 0.95\} \) across three datasets. Values are mean ± std in percent.

| Model    | p   | D1 F1              | D1 Recall          | D1 Precision        | D2 F1              | D2 Recall          | D2 Precision        | D3 F1              | D3 Recall          | D3 Precision        |
|----------|-----|--------------------|--------------------|---------------------|--------------------|--------------------|---------------------|--------------------|--------------------|---------------------|
| TwinTCN  | p80 | 79.48% ± 3.18%     | 84.67% ± 3.80%     | 75.28% ± 6.85%      | 85.94% ± 2.14%     | 87.50% ± 4.42%     | 84.83% ± 5.51%      | 78.21% ± 7.57%     | 95.56% ± 6.09%     | 67.19% ± 11.50%     |
| TwinTCN  | p85 | 83.11% ± 2.72%     | 83.33% ± 2.36%     | 83.06% ± 5.14%      | 87.58% ± 3.55%     | 87.50% ± 4.42%     | 88.12% ± 7.55%      | 79.35% ± 9.90%     | 88.89% ± 7.86%     | 73.20% ± 15.05%     |
| TwinTCN  | p90 | 82.13% ± 3.26%     | 78.00% ± 6.50%     | 87.67% ± 8.36%      | 88.14% ± 2.99%     | 87.50% ± 4.42%     | 89.36% ± 7.76%      | 76.60% ± 6.03%     | 80.00% ± 11.46%    | 77.06% ± 16.67%     |
| TwinTCN  | p95 | 80.47% ± 2.33%     | 72.67% ± 2.79%     | 90.47% ± 5.86%      | 90.21% ± 2.13%     | 86.25% ± 5.23%     | 95.15% ± 6.65%      | 63.54% ± 7.17%     | 51.11% ± 9.56%     | 87.43% ± 13.29%     |
| DyAD     | p80 | 37.70% ± 3.13%     | 40.72% ± 1.69%     | 35.52% ± 5.70%      | 84.59% ± 1.07%     | 74.77% ± 2.16%     | 97.44% ± 0.89%      | 52.99% ± 4.20%     | 49.64% ± 8.38%     | 58.19% ± 6.75%      |
| DyAD     | p85 | 37.93% ± 2.89%     | 38.14% ± 1.85%     | 38.23% ± 6.15%      | 83.09% ± 1.51%     | 72.35% ± 2.71%     | 97.68% ± 0.97%      | 49.46% ± 5.71%     | 43.49% ± 9.79%     | 59.44% ± 6.88%      |
| DyAD     | p90 | 37.96% ± 2.54%     | 34.21% ± 2.46%     | 43.39% ± 6.99%      | 79.96% ± 2.73%     | 67.53% ± 4.18%     | 98.24% ± 0.98%      | 43.45% ± 9.21%     | 35.10% ± 11.17%    | 61.49% ± 8.01%      |
| DyAD     | p95 | 37.10% ± 2.20%     | 29.33% ± 1.24%     | 51.92% ± 10.19%     | 71.91% ± 6.02%     | 56.84% ± 7.75%     | 98.89% ± 1.08%      | 25.63% ± 8.70%     | 16.48% ± 7.13%     | 64.60% ± 5.75%      |
| TranAD   | p80 | 71.15% ± 3.34%     | 83.33% ± 0.00%     | 62.22% ± 5.11%      | 85.47% ± 3.27%     | 87.50% ± 0.00%     | 83.71% ± 6.03%      | 42.24% ± 6.58%     | 35.55% ± 4.97%     | 52.48% ± 10.60%     |
| TranAD   | p85 | 69.37% ± 2.82%     | 76.67% ± 0.00%     | 63.47% ± 4.70%      | 86.00% ± 3.35%     | 87.50% ± 0.00%     | 84.74% ± 6.18%      | 45.03% ± 3.78%     | 33.33% ± 0.00%     | 72.00% ± 18.91%     |
| TranAD   | p90 | 71.79% ± 3.38%     | 73.33% ± 0.00%     | 70.54% ± 6.52%      | 87.51% ± 3.12%     | 83.75% ± 5.59%     | 91.87% ± 2.48%      | 46.37% ± 3.57%     | 33.33% ± 0.00%     | 79.00% ± 20.12%     |
| TranAD   | p95 | 72.03% ± 1.73%     | 60.00% ± 0.00%     | 90.26% ± 5.28%      | 77.33% ± 6.19%     | 65.00% ± 9.48%     | 96.57% ± 4.80%      | 7.27% ± 16.26%     | 4.44% ± 9.94%      | 20.00% ± 44.72%     |
| TimesNet | p80 | 60.27% ± 6.34%     | 64.00% ± 8.63%     | 57.13% ± 5.35%      | 83.08% ± 2.02%     | 88.75% ± 2.80%     | 78.23% ± 4.04%      | 71.14% ± 14.02%    | 66.67% ± 15.72%    | 77.85% ± 15.43%     |
| TimesNet | p85 | 53.26% ± 7.52%     | 49.33% ± 11.16%    | 59.36% ± 7.14%      | 78.32% ± 2.18%     | 78.75% ± 3.42%     | 78.28% ± 5.93%      | 15.27% ± 15.46%    | 8.89% ± 9.30%      | 60.00% ± 54.77%     |
| TimesNet | p90 | 46.64% ± 11.17%    | 35.33% ± 13.46%    | 76.75% ± 11.58%     | 73.25% ± 7.41%     | 68.75% ± 8.84%     | 78.91% ± 8.29%      | 0.00% ± 0.00%      | 0.00% ± 0.00%      | 0.00% ± 0.00%       |
| TimesNet | p95 | 26.77% ± 15.76%    | 17.33% ± 13.21%    | 88.50% ± 16.92%     | 45.91% ± 17.46%    | 33.75% ± 15.69%    | 77.64% ± 12.25%     | 0.00% ± 0.00%      | 0.00% ± 0.00%      | 0.00% ± 0.00%       |
| WACformer| p80 | 35.03% ± 20.55%    | 43.33% ± 25.27%    | 30.91% ± 21.05%     | 80.00% ± 14.14%    | 86.67% ± 18.26%    | 81.91% ± 26.17%     | 61.33% ± 23.64%    | 100.00% ± 0.00%    | 48.33% ± 30.28%     |
| WACformer| p85 | 33.47% ± 20.92%    | 36.67% ± 21.73%    | 32.00% ± 21.68%     | 59.78% ± 36.52%    | 60.00% ± 36.52%    | 64.00% ± 43.36%     | 63.43% ± 26.22%    | 90.00% ± 22.36%    | 53.00% ± 30.56%     |
| WACformer| p90 | 12.86% ± 21.67%    | 10.00% ± 14.91%    | 22.50% ± 43.66%     | 33.14% ± 45.43%    | 28.33% ± 38.91%    | 40.00% ± 54.77%     | 74.76% ± 23.79%    | 80.00% ± 27.39%    | 78.00% ± 30.33%     |
| WACformer| p95 | 12.86% ± 21.67%    | 10.00% ± 14.91%    | 22.50% ± 43.66%     | 33.14% ± 45.43%    | 28.33% ± 38.91%    | 40.00% ± 54.77%     | 74.76% ± 23.79%    | 80.00% ± 27.39%    | 78.00% ± 30.33%     |
| DCdetector| p80| 14.10% ± 19.30%    | 10.70% ± 14.60%    | 20.80% ± 28.80%     | 18.00% ± 16.50%    | 11.30% ± 10.30%    | 47.00% ± 45.20%     | 31.20% ± 3.50%     | 28.90% ± 9.90%     | 38.80% ± 10.30%     |
| DCdetector| p85| 8.00% ± 11.00%     | 5.30% ± 7.30%      | 16.20% ± 22.30%     | 4.30% ± 5.90%      | 2.50% ± 3.40%      | 16.70% ± 23.60%     | 5.70% ± 12.80%     | 4.40% ± 9.90%      | 8.00% ± 17.90%      |
| DCdetector| p90| 3.40% ± 5.00%      | 2.00% ± 3.00%      | 13.30% ± 18.30%     | 4.50% ± 6.10%      | 2.50% ± 3.40%      | 26.70% ± 43.50%     | 0.00% ± 0.00%      | 0.00% ± 0.00%      | 0.00% ± 0.00%       |
| DCdetector| p95| 0.00% ± 0.00%      | 0.00% ± 0.00%      | 0.00% ± 0.00%       | 0.00% ± 0.00%      | 0.00% ± 0.00%      | 0.00% ± 0.00%       | 0.00% ± 0.00%      | 0.00% ± 0.00%      | 0.00% ± 0.00%       |
| GDN      | p80 | 16.17% ± 6.74%     | 12.92% ± 7.59%     | 24.86% ± 5.09%      | 18.48% ± 27.29%    | 12.96% ± 21.29%    | 91.36% ± 3.28%      | 36.77% ± 13.38%    | 34.49% ± 13.97%    | 48.13% ± 20.17%     |
| GDN      | p85 | 13.87% ± 6.39%     | 10.26% ± 6.75%     | 26.25% ± 5.18%      | 16.84% ± 27.23%    | 11.87% ± 20.80%    | 91.81% ± 2.71%      | 35.93% ± 13.24%    | 33.03% ± 13.65%    | 48.73% ± 20.69%     |
| GDN      | p90 | 11.51% ± 4.23%     | 7.59% ± 3.42%      | 27.54% ± 5.67%      | 15.33% ± 26.95%    | 10.87% ± 20.42%    | 91.38% ± 2.99%      | 34.01% ± 12.77%    | 30.24% ± 13.14%    | 49.72% ± 21.41%     |
| GDN      | p95 | 8.91% ± 3.90%      | 5.48% ± 2.78%      | 29.85% ± 6.95%      | 13.02% ± 23.13%    | 8.70% ± 16.34%     | 91.37% ± 2.90%      | 32.15% ± 12.34%    | 27.47% ± 12.05%    | 50.29% ± 22.15%     |
| LSTM-AE  | p80 | 13.70% ± 1.10%     | 10.80% ± 0.90%     | 19.00% ± 2.40%      | 10.20% ± 2.20%     | 5.40% ± 1.30%      | 87.80% ± 4.40%      | 54.60% ± 3.10%     | 52.10% ± 2.30%     | 58.40% ± 10.20%     |
| LSTM-AE  | p85 | 12.60% ± 0.70%     | 9.50% ± 0.40%      | 19.00% ± 2.30%      | 7.50% ± 1.30%      | 3.90% ± 0.70%      | 87.60% ± 4.80%      | 53.40% ± 3.00%     | 49.10% ± 1.50%     | 59.30% ± 9.60%      |
| LSTM-AE  | p90 | 11.70% ± 0.90%     | 8.50% ± 0.70%      | 19.10% ± 2.30%      | 6.50% ± 0.90%      | 3.40% ± 0.50%      | 89.80% ± 5.00%      | 51.70% ± 2.40%     | 45.70% ± 2.10%     | 60.40% ± 9.90%      |
| LSTM-AE  | p95 | 10.60% ± 0.60%     | 7.30% ± 0.30%      | 19.50% ± 2.00%      | 5.60% ± 0.50%      | 2.90% ± 0.30%      | 87.60% ± 5.40%      | 50.00% ± 2.00%     | 42.60% ± 1.80%     | 61.40% ± 9.40%      |
| AE       | p80 | 24.70% ± 0.90%     | 27.60% ± 1.50%     | 22.40% ± 1.00%      | 16.10% ± 6.10%     | 9.00% ± 3.90%      | 89.90% ± 1.50%      | 53.60% ± 5.30%     | 60.90% ± 7.60%     | 48.60% ± 6.90%      |
| AE       | p85 | 23.40% ± 1.10%     | 24.30% ± 2.00%     | 22.60% ± 1.00%      | 10.90% ± 1.40%     | 5.80% ± 0.80%      | 89.30% ± 3.10%      | 52.70% ± 5.40%     | 56.20% ± 7.70%     | 50.40% ± 6.90%      |
| AE       | p90 | 21.90% ± 1.10%     | 21.20% ± 1.90%     | 22.80% ± 1.10%      | 9.30% ± 1.00%      | 4.90% ± 0.60%      | 89.10% ± 3.60%      | 51.30% ± 4.60%     | 51.60% ± 5.90%     | 51.90% ± 7.30%      |
| AE       | p95 | 20.50% ± 1.20%     | 18.60% ± 1.50%     | 23.00% ± 1.30%      | 8.40% ± 0.50%      | 4.40% ± 0.30%      | 89.00% ± 4.00%      | 49.20% ± 3.40%     | 46.30% ± 3.60%     | 53.40% ± 7.40%      |
| SVDD     | p80 | 0.10% ± 0.10%      | 0.10% ± 0.10%      | 46.30% ± 17.40%     | 67.70% ± 30.40%    | 64.70% ± 37.30%    | 87.30% ± 7.00%      | 37.20% ± 7.50%     | 42.60% ± 8.60%     | 33.40% ± 7.90%      |
| SVDD     | p85 | 0.10% ± 0.10%      | 0.10% ± 0.10%      | 46.30% ± 17.40%     | 66.90% ± 31.30%    | 64.00% ± 37.60%    | 87.20% ± 7.40%      | 36.10% ± 7.20%     | 39.70% ± 8.60%     | 33.60% ± 7.90%      |
| SVDD     | p90 | 0.10% ± 0.10%      | 0.10% ± 0.10%      | 46.30% ± 17.40%     | 66.00% ± 32.50%    | 63.40% ± 38.80%    | 87.10% ± 7.70%      | 35.40% ± 7.10%     | 38.10% ± 8.80%     | 33.70% ± 7.90%      |
| SVDD     | p95 | 0.10% ± 0.10%      | 0.10% ± 0.10%      | 46.30% ± 17.40%     | 64.20% ± 34.40%    | 62.10% ± 40.20%    | 87.20% ± 7.90%      | 33.70% ± 7.00%     | 34.30% ± 8.30%     | 33.80% ± 7.90%      |


### Protocol Clarification

**Thresholding.**  
For each \( p \in \{0.80, 0.85, 0.90, 0.95\} \), we set a fixed absolute threshold
\[
\tau_p = Q_p\big(\{s_i^{\text{train}}\}\big)
\]
using the *upper* p-quantile of the *training* score distribution (larger scores indicate more anomalous). The same \(\tau_p\) is then applied to the validation split. Consequently, the validation predicted–positive rate is *not* constrained to \(1 - p\).

**Aggregation level.**  
Window scores are averaged per vehicle; the decision rule is
\[
\hat{y} = \mathbf{1}[\,s > \tau_p\,]
\]
at the vehicle level.

**Metrics.**  
Precision \(P\), Recall \(R\), and
\[
F1 = \frac{2PR}{P + R}
\]
are computed per vehicle. AUROC is reported separately as a threshold-free auxiliary metric.


### Dataset-Specific Context (Unsupervised Validation)

- **Dataset1:** Validation abnormal-length ratio ≈ 13.7%–18.2% (moderate). F1 often peaks at an *intermediate* \(p\) (e.g., p85) where \(P\) and \(R\) are comparable.

- **Dataset2:** Validation is *heavily abnormal* (83.6%–93.6%). With train-quantile thresholding, \(\tau_p\) can still yield *high recall* even at p95, because \(\tau_p\) is calibrated on training (typically normal-dominant) while many validation sequences exceed it.

- **Dataset3:** Validation abnormal ratio is *low-to-moderate* (20.2%–28.7%). As \(p\) increases, the typical \(P \uparrow\), \(R \downarrow\) trade-off appears; the F1 optimum emerges near the balance point (often at intermediate \(p\)).

Supervised 5-fold splits have different base rates per fold; hence the F1-optimal \(p\) may shift. Reporting multiple \(p\)’s is informative.



## On Seemingly Odd Entries

- **High recall at p95 on Dataset2.**  
  This is consistent with train-quantile thresholding: \(\tau_{0.95}\) is set on training and can be relatively low for a validation set dominated by abnormal sequences.

- **Identical rows at p90 and p95.**  
  Quantile ties may give \(Q_{0.90} = Q_{0.95}\); with the strict comparator \(s > \tau_p\), the predicted-positive set can be identical.

- **Zero P/R/F1 at high p.**  
  This occurs when all validation scores are ≤ \(\tau_p\) under \(s > \tau_p\). Using \(s \ge \tau_p\) is a robust alternative.



