# Explicit Uncertainty Decomposition for Probabilistic Load Forecasting with Calibrated Latent Diffusion

This repository contains the source code for the paper:

> **Explicit Uncertainty Decomposition for Probabilistic Load Forecasting with Calibrated Latent Diffusion**
>
> Ye Yang and Yuan Tian
>
> *Submitted to Reliability Engineering & System Safety*

## Overview

We propose a probabilistic load forecasting framework that achieves explicit decomposition of predictive uncertainty into epistemic and aleatoric components. The framework integrates:

1. **iTransformer Encoder** — captures cross-variate dependencies via inverted attention
2. **Latent Space Diffusion Module** — introduces controlled stochasticity for Monte Carlo uncertainty estimation
3. **Dual-Branch Uncertainty Head** — directly predicts both epistemic and aleatoric uncertainties
4. **End-to-End Calibration Loss** — aligns epistemic uncertainty with prediction errors during training

![Framework](docs/framework.png)

## Repository Structure

```
├── models/
│   └── diffload_uncertainty_v3_enhanced.py   # Main model: training and evaluation
├── experiments/
│   ├── run_baselines.py                      # Baselines: iTransformer, MC Dropout, Deep Ensemble
│   ├── run_gp_baseline.py                    # Gaussian Process baseline
│   ├── ablation_explicit_head.py             # Ablation: explicit head vs MC-only vs explicit-only
│   ├── run_calib_sensitivity.py              # Ablation: λ_calib sensitivity (Table 5)
│   └── run_gamma_ablation.py                 # Ablation: learnable scaling factors
├── utils/
│   └── advanced_visualization.py             # Visualization utilities for figures
├── data/                                     # Place datasets here (see below)
├── requirements.txt
└── README.md
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA (recommended for training)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

We evaluate on two publicly available datasets:

| Dataset | Period | Source |
|---------|--------|--------|
| GEFCom2014 | 2004–2014 | [Global Energy Forecasting Competition 2014](https://www.sciencedirect.com/science/article/pii/S0169207016000133) |
| ISO-NE COVID-19 | 2017–2020 | [Day-ahead Electricity Demand Forecasting Competition](https://ieee-dataport.org/competitions/day-ahead-electricity-demand-forecasting-post-covid-paradigm) |

### Data Preparation

1. Download the datasets from the sources above.
2. Preprocess into NumPy arrays with shape `(n_samples, seq_len, n_features)` for inputs and `(n_samples,)` for labels.
3. Place the `.npy` files in the `data/` directory:

```
data/
├── train_data.npy      # Training inputs,  shape: (N_train, seq_len, n_features)
├── train_label.npy     # Training targets, shape: (N_train,)
├── val_data.npy        # Validation inputs
├── val_label.npy       # Validation targets
├── test_data.npy       # Test inputs
└── test_label.npy      # Test targets
```

**Note:** The default data path in the scripts is `../GEF_data/`. You can either create a symlink (`ln -s data/ ../GEF_data`) or update the paths in the scripts. We plan to release a preprocessing script in a future update.

## Usage

### Train and evaluate the proposed method

```bash
cd models
python diffload_uncertainty_v3_enhanced.py \
    --mode both \
    --epochs 300 \
    --batch_size 256 \
    --lr 5e-4 \
    --d_model 64 \
    --n_heads 4 \
    --n_layers 2 \
    --diff_steps 5 \
    --mc_samples 100 \
    --lambda_diff 1.0 \
    --lambda_calib 0.1
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `both` | `train`, `test`, or `both` |
| `--epochs` | 300 | Number of training epochs |
| `--d_model` | 64 | Hidden dimension of iTransformer |
| `--diff_steps` | 5 | Number of diffusion steps |
| `--mc_samples` | 100 | Monte Carlo samples at inference |
| `--lambda_diff` | 1.0 | Weight of diffusion loss |
| `--lambda_calib` | 0.1 | Weight of calibration loss |

### Run baseline comparisons (Table 6)

```bash
cd experiments

# iTransformer (No UQ), MC Dropout, Deep Ensemble
python run_baselines.py

# Gaussian Process
python run_gp_baseline.py
```

### Ablation studies

```bash
cd experiments

# Effect of explicit uncertainty head (Table 4)
python ablation_explicit_head.py --variant mc_only
python ablation_explicit_head.py --variant explicit_only

# Calibration loss sensitivity (Table 5)
python run_calib_sensitivity.py

# Scaling factor ablation
python run_gamma_ablation.py
```

## Results

### Overall Performance (GEFCom2014)

| Method | MAE | MAPE (%) | CRPS | ECE | PICP₉₀ |
|--------|-----|----------|------|-----|--------|
| iTransformer (No UQ) | 117.16 | 3.53 | — | — | — |
| Gaussian Process | 142.00 | 4.22 | 104.88 | 0.069 | 78.9% |
| DeepAR | 117.46 | 3.50 | 84.34 | 0.029 | 87.9% |
| MC Dropout | 112.73 | 3.38 | 88.83 | 0.217 | 57.0% |
| Deep Ensemble | 113.54 | 3.41 | 97.03 | 0.343 | 35.5% |
| DiffLoad | 110.43 | 3.30 | 86.21 | 0.212 | 59.2% |
| **Ours** | **109.54** | **3.26** | **79.39** | **0.021** | **88.8%** |

## Citation

If you find this code useful, please cite:

```bibtex
@article{yang2026explicit,
  title={Explicit Uncertainty Decomposition for Probabilistic Load Forecasting with Calibrated Latent Diffusion},
  author={Yang, Ye and Tian, Yuan},
  journal={Reliability Engineering \& System Safety},
  year={2026},
  note={Under review}
}
```

## License

This project is released for academic research purposes. Please contact the authors for commercial use.
