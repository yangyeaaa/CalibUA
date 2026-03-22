# diffload_uncertainty_v3_enhanced.py
"""
DiffLoad Uncertainty Decomposition v3 Enhanced
=====================================================================
Based on the original v3 code with enhanced output display.

Key improvements from v2:
1. Multi-step diffusion sampling for better epistemic estimation
2. Ensemble of hidden states for variance amplification
3. Learnable epistemic scaling factor
4. Proper ECE calculation from parametric distribution
5. NLL-based calibration loss

Enhanced features:
- Complete Pinball Loss metrics (75%, 50%, 25%)
- Rich formatted console output with tables
- Enhanced 20-panel visualization
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, cauchy
import properscoring as ps

import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -------------------------
# Device
# -------------------------
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# Diffusion Schedule
# =========================
class DiffusionSchedule:
    def __init__(self, num_steps: int, device: torch.device):
        self.num_steps = num_steps
        self.device = device
        self._init_schedule()
    
    def _init_schedule(self):
        betas = torch.linspace(-6, 6, self.num_steps).to(self.device)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_prod = alphas_prod
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        
        print(f"Diffusion schedule: {self.num_steps} steps")
    
    def q_sample(self, x_0: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion with noise return."""
        noise = torch.randn_like(x_0)
        x_t = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        return x_t, noise
    
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: int, 
                 add_noise: bool = True) -> torch.Tensor:
        """Reverse diffusion step."""
        x = x.to(self.device)
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        t_tensor = torch.tensor([t]).to(self.device)
        eps_theta = model(x, t_tensor)
        
        mean = (1 / (1 - self.betas[t]).sqrt()) * (x - coeff * eps_theta)
        
        if add_noise and t > 0:
            z = torch.randn_like(x)
            sigma_t = self.betas[t].sqrt()
            return mean + sigma_t * z
        return mean
    
    def p_sample_loop(self, model: nn.Module, x_T: torch.Tensor, 
                      start_t: int = None) -> torch.Tensor:
        """Full reverse diffusion from timestep start_t to 0."""
        if start_t is None:
            start_t = self.num_steps - 1
        
        x = x_T
        for t in reversed(range(start_t + 1)):
            x = self.p_sample(model, x, t, add_noise=(t > 0))
        return x
    
    def compute_loss(self, model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_steps, size=(batch_size // 2,)).to(self.device)
        t = torch.cat([t, self.num_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)
        
        noise = torch.randn_like(x_0)
        x_noisy = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        
        predicted_noise = model(x_noisy, t.squeeze(-1))
        return F.mse_loss(predicted_noise, noise)


# =========================
# Utility Functions
# =========================
def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_folders(root_path: str, run_id: int) -> Tuple[str, str]:
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(f"{root_path}/models", exist_ok=True)
    os.makedirs(f"{root_path}/results", exist_ok=True)
    
    model_path = f"{root_path}/models/run_{run_id}"
    result_path = f"{root_path}/results/run_{run_id}"
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
    return model_path, result_path


# =========================
# Model Components
# =========================
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps: int, hidden_dim: int, mlp_dim: int = 128):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        ])
        
        self.time_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, mlp_dim),
            nn.Embedding(n_steps, mlp_dim),
            nn.Embedding(n_steps, mlp_dim),
        ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for idx, time_emb in enumerate(self.time_embeddings):
            t_emb = time_emb(t.to(x.device))
            x = self.layers[2 * idx](x)
            x = x + t_emb
            x = self.layers[2 * idx + 1](x)
        return self.layers[-1](x)


class iTransformerEncoder(nn.Module):
    def __init__(self, seq_len: int, n_variates: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.variate_embedding = nn.Linear(seq_len, d_model)
        self.variate_pos = nn.Parameter(torch.zeros(1, n_variates, d_model))
        nn.init.trunc_normal_(self.variate_pos, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        tokens = self.variate_embedding(x)
        tokens = tokens + self.variate_pos
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        tokens = tokens + self.temporal_ffn(tokens)
        return tokens[:, 0, :]


class ExplicitUncertaintyHead(nn.Module):
    """
    Uncertainty head with EXPLICIT epistemic and aleatoric branches.
    
    Key insight: Instead of relying solely on diffusion sampling variance,
    we directly predict both uncertainty components.
    """
    
    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Main prediction (deterministic part)
        self.main_head = nn.Linear(hidden_dim, output_dim)
        
        # Residual mu
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        
        # Aleatoric uncertainty (data noise) - bounded
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # Epistemic uncertainty (model uncertainty) - explicitly predicted
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # Learnable scaling factors
        self.log_aleatoric_scale = nn.Parameter(torch.tensor(0.0))
        self.log_epistemic_scale = nn.Parameter(torch.tensor(-1.0))  # Start smaller
        
        # Bounds
        self.sigma_min = 0.01
        self.sigma_max = 3.0
    
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_feat = self.shared(hidden)
        
        main_value = self.main_head(shared_feat).squeeze(-1)
        residual_mu = self.mu_head(shared_feat).squeeze(-1)
        
        # Aleatoric: bounded sigmoid output
        aleatoric_raw = self.aleatoric_head(shared_feat).squeeze(-1)
        aleatoric_scale = torch.exp(self.log_aleatoric_scale)
        aleatoric = torch.sigmoid(aleatoric_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min
        aleatoric = aleatoric * aleatoric_scale
        
        # Epistemic: also bounded but with separate scale
        epistemic_raw = self.epistemic_head(shared_feat).squeeze(-1)
        epistemic_scale = torch.exp(self.log_epistemic_scale)
        epistemic = torch.sigmoid(epistemic_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min
        epistemic = epistemic * epistemic_scale
        
        return {
            'main_value': main_value,
            'residual_mu': residual_mu,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
        }
    
    def get_scales(self) -> Dict[str, float]:
        return {
            'aleatoric_scale': torch.exp(self.log_aleatoric_scale).item(),
            'epistemic_scale': torch.exp(self.log_epistemic_scale).item(),
        }


class DiffLoadV3(nn.Module):
    """
    DiffLoad v3 with explicit uncertainty decomposition.
    
    Key changes:
    1. Epistemic is explicitly predicted, not just from MC variance
    2. Diffusion still adds stochasticity, but epistemic head learns the magnitude
    3. Loss includes terms for both uncertainty types
    """
    
    def __init__(self, seq_len: int, n_variates: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, num_diff_steps: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_diff_steps = num_diff_steps
        
        self.encoder = iTransformerEncoder(
            seq_len=seq_len,
            n_variates=n_variates,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        self.diffusion_net = MLPDiffusion(num_diff_steps, d_model)
        self.uncertainty_head = ExplicitUncertaintyHead(d_model, output_dim=1)
        self.diff_schedule: Optional[DiffusionSchedule] = None
    
    def set_diffusion_schedule(self, schedule: DiffusionSchedule):
        self.diff_schedule = schedule
    
    def forward(self, x: torch.Tensor, use_diffusion: bool = True
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional diffusion.
        
        Returns dict with all uncertainty components.
        """
        hidden = self.encoder(x)
        
        diff_loss = torch.tensor(0.0).to(x.device)
        
        if use_diffusion and self.diff_schedule is not None:
            # Add noise
            hidden_noisy, _ = self.diff_schedule.q_sample(hidden, self.num_diff_steps - 1)
            
            # Compute diffusion loss
            diff_loss = self.diff_schedule.compute_loss(self.diffusion_net, hidden)
            
            # Denoise
            hidden_denoised = self.diff_schedule.p_sample(
                self.diffusion_net, hidden_noisy, self.num_diff_steps - 1
            )
        else:
            hidden_denoised = hidden
        
        # Get uncertainty predictions
        outputs = self.uncertainty_head(hidden_denoised)
        outputs['diff_loss'] = diff_loss
        outputs['hidden'] = hidden_denoised
        
        return outputs
    
    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100
                                  ) -> Dict[str, torch.Tensor]:
        """
        Prediction with proper uncertainty quantification.
        
        Combines:
        1. Explicitly predicted epistemic/aleatoric
        2. MC sampling variance (adds to epistemic)
        """
        self.eval()
        
        # Collect samples
        main_samples = []
        mu_samples = []
        pred_samples = []
        aleatoric_samples = []
        epistemic_samples = []
        
        for _ in range(n_samples):
            outputs = self.forward(x, use_diffusion=True)
            
            main_samples.append(outputs['main_value'].cpu())
            mu_samples.append(outputs['residual_mu'].cpu())
            pred_samples.append((outputs['main_value'] + outputs['residual_mu']).cpu())
            aleatoric_samples.append(outputs['aleatoric'].cpu())
            epistemic_samples.append(outputs['epistemic'].cpu())
        
        # Stack
        main_samples = torch.stack(main_samples, dim=0)
        mu_samples = torch.stack(mu_samples, dim=0)
        pred_samples = torch.stack(pred_samples, dim=0)
        aleatoric_samples = torch.stack(aleatoric_samples, dim=0)
        epistemic_samples = torch.stack(epistemic_samples, dim=0)
        
        # Point prediction
        prediction = pred_samples.mean(dim=0)
        
        # Aleatoric: mean of predicted aleatoric
        aleatoric_pred = aleatoric_samples.mean(dim=0)
        
        # Epistemic: combine predicted epistemic + MC variance
        epistemic_pred = epistemic_samples.mean(dim=0)
        mc_variance = pred_samples.var(dim=0)
        mc_std = torch.sqrt(mc_variance)
        
        # Total epistemic = predicted + MC-based (they capture different aspects)
        # Use sqrt of sum of squares (assuming independence)
        epistemic_total = torch.sqrt(epistemic_pred**2 + mc_std**2)
        
        # Total uncertainty
        total_std = torch.sqrt(aleatoric_pred**2 + epistemic_total**2)
        
        # Epistemic ratio
        epistemic_ratio = epistemic_total / (total_std + 1e-8)
        
        return {
            'prediction': prediction,
            'aleatoric': aleatoric_pred,
            'epistemic_pred': epistemic_pred,
            'epistemic_mc': mc_std,
            'epistemic_total': epistemic_total,
            'total_std': total_std,
            'epistemic_ratio': epistemic_ratio,
            'mc_samples': pred_samples.numpy(),
            'scales': self.uncertainty_head.get_scales(),
        }


# =========================
# Loss Functions
# =========================
class DecomposedUncertaintyLoss(nn.Module):
    """
    Loss function that properly trains both uncertainty components.
    
    Key insight: 
    - Aleatoric should capture residual variance
    - Epistemic should be higher when prediction is wrong
    """
    
    def __init__(self, lambda_diff: float = 1.0, lambda_calib: float = 0.1):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_calib = lambda_calib
    
    def forward(self, target: torch.Tensor, outputs: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        main_value = outputs['main_value']
        mu = outputs['residual_mu']
        aleatoric = outputs['aleatoric']
        epistemic = outputs['epistemic']
        diff_loss = outputs['diff_loss']
        
        # Prediction
        prediction = main_value + mu
        residual = target - prediction
        
        # Total variance for NLL
        total_var = aleatoric**2 + epistemic**2 + 1e-6
        total_std = torch.sqrt(total_var)
        
        # Gaussian NLL with total uncertainty
        nll = 0.5 * (torch.log(total_var) + residual**2 / total_var)
        nll_loss = nll.mean()
        
        # Calibration loss: epistemic should correlate with |error|
        # Idea: if |residual| > total_std, epistemic should be higher
        abs_residual = torch.abs(residual)
        normalized_error = abs_residual / (total_std + 1e-6)
        
        # Soft calibration: encourage epistemic to scale with error magnitude
        # When error is large, epistemic should be large
        calib_loss = F.mse_loss(
            epistemic,
            (abs_residual * 0.5).detach()  # Target: half of absolute error
        )
        
        # Total loss
        total_loss = nll_loss + self.lambda_diff * diff_loss + self.lambda_calib * calib_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'nll_loss': nll_loss.item(),
            'diff_loss': diff_loss.item(),
            'calib_loss': calib_loss.item(),
        }


# =========================
# Metrics
# =========================
class UncertaintyMetrics:
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        return np.mean(ps.crps_gaussian(y_true, mu, sigma))
    
    @staticmethod
    def crps_empirical(y_true: np.ndarray, mc_samples: np.ndarray) -> float:
        crps_values = []
        for i in range(len(y_true)):
            samples = mc_samples[:, i]
            crps_values.append(ps.crps_ensemble(y_true[i], samples))
        return np.mean(crps_values)
    
    @staticmethod
    def pinball_loss(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                     quantiles: List[float]) -> float:
        losses = []
        for i in range(len(y_true)):
            ppf_values = norm.ppf(quantiles, loc=mu[i], scale=sigma[i])
            for q, pred_q in zip(quantiles, ppf_values):
                error = y_true[i] - pred_q
                loss = np.maximum((q - 1) * error, q * error)
                losses.append(loss)
        return np.mean(losses)
    
    @staticmethod
    def coverage_metrics(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray
                         ) -> Dict[str, float]:
        results = {}
        for conf in [0.50, 0.75, 0.90, 0.95]:
            z = norm.ppf((1 + conf) / 2)
            lower = mu - z * sigma
            upper = mu + z * sigma
            
            picp = np.mean((y_true >= lower) & (y_true <= upper))
            mpiw = np.mean(upper - lower)
            
            results[f'PICP_{int(conf*100)}'] = picp
            results[f'MPIW_{int(conf*100)}'] = mpiw
            results[f'Gap_{int(conf*100)}'] = picp - conf
        
        return results
    
    @staticmethod
    def ece_parametric(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                       n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
        """ECE from parametric Gaussian distribution."""
        confidence_levels = np.linspace(0.1, 0.95, n_bins)
        observed_coverage = []
        
        for conf in confidence_levels:
            z = norm.ppf((1 + conf) / 2)
            lower = mu - z * sigma
            upper = mu + z * sigma
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            observed_coverage.append(coverage)
        
        observed_coverage = np.array(observed_coverage)
        ece = np.mean(np.abs(confidence_levels - observed_coverage))
        
        return ece, confidence_levels, observed_coverage
    
    @staticmethod
    def uncertainty_quality(epistemic: np.ndarray, aleatoric: np.ndarray,
                            errors: np.ndarray) -> Dict[str, float]:
        total = np.sqrt(epistemic**2 + aleatoric**2)
        abs_errors = np.abs(errors)
        
        # Correlations
        corr_total = np.corrcoef(total, abs_errors)[0, 1]
        corr_epi = np.corrcoef(epistemic, abs_errors)[0, 1]
        corr_ale = np.corrcoef(aleatoric, abs_errors)[0, 1]
        
        # Spearman rank correlation (more robust)
        from scipy.stats import spearmanr
        spearman_total, _ = spearmanr(total, abs_errors)
        spearman_epi, _ = spearmanr(epistemic, abs_errors)
        
        return {
            'corr_total_error': corr_total,
            'corr_epistemic_error': corr_epi,
            'corr_aleatoric_error': corr_ale,
            'spearman_total_error': spearman_total,
            'spearman_epistemic_error': spearman_epi,
            'epistemic_ratio': np.mean(epistemic / (total + 1e-8)),
            'mean_epistemic': np.mean(epistemic),
            'mean_aleatoric': np.mean(aleatoric),
            'mean_total': np.mean(total),
            'std_epistemic': np.std(epistemic),
            'std_aleatoric': np.std(aleatoric),
        }


# =========================
# Dataset
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.data)


# =========================
# Enhanced Visualization
# =========================
def plot_results(y_true: np.ndarray, results: Dict, metrics: Dict, save_path: str):
    """Enhanced comprehensive visualization with 20 panels."""
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    n_points = min(300, len(y_true))
    t = np.arange(n_points)
    
    pred = results['prediction'][:n_points]
    epi = results['epistemic'][:n_points]
    ale = results['aleatoric'][:n_points]
    total = results['total_std'][:n_points]
    errors = y_true[:n_points] - pred
    abs_errors = np.abs(errors)
    
    # =====================
    # Row 1: Main predictions and metrics
    # =====================
    
    # 1. Main prediction plot (spans 3 columns)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(t, y_true[:n_points], 'b-', label='Ground Truth', lw=2, alpha=0.8)
    ax1.plot(t, pred, 'r--', label='Prediction', lw=1.5, alpha=0.8)
    ax1.fill_between(t, pred - 1.96*total, pred + 1.96*total,
                    alpha=0.2, color='gray', label='95% CI')
    ax1.fill_between(t, pred - total, pred + total,
                    alpha=0.15, color='blue', label='68% CI')
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Load Value', fontsize=11)
    ax1.set_title('Predictions with Confidence Intervals', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Metrics summary box
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.axis('off')
    
    text = "═══ PERFORMANCE ═══\n\n"
    text += f"MAE:     {metrics['MAE']:.2f}\n"
    text += f"MAPE:    {metrics['MAPE']:.2f}%\n"
    text += f"RMSE:    {metrics['RMSE']:.2f}\n\n"
    text += "═══ PROBABILISTIC ═══\n\n"
    text += f"CRPS(G): {metrics['CRPS_Gaussian']:.2f}\n"
    text += f"CRPS(E): {metrics['CRPS_Empirical']:.2f}\n\n"
    text += "═══ PINBALL LOSS ═══\n\n"
    text += f"PB 75%:  {metrics['Pinball_75']:.2f}\n"
    text += f"PB 50%:  {metrics['Pinball_50']:.2f}\n"
    text += f"PB 25%:  {metrics['Pinball_25']:.2f}\n\n"
    text += "═══ CALIBRATION ═══\n\n"
    text += f"ECE:     {metrics['ECE']:.4f}\n"
    
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray'))
    
    # =====================
    # Row 2: Uncertainty decomposition
    # =====================
    
    # 3. Stacked uncertainty over time
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.stackplot(t, ale, epi, labels=['Aleatoric', 'Epistemic'],
                 colors=['#ff7f0e', '#2ca02c'], alpha=0.7)
    ax3.set_xlabel('Time Step', fontsize=10)
    ax3.set_ylabel('Uncertainty (σ)', fontsize=10)
    ax3.set_title('Decomposed Uncertainty (Stacked Area)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Uncertainty components separately
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.plot(t, epi, 'g-', label=f'Epistemic (μ={np.mean(epi):.1f})', lw=1.5, alpha=0.8)
    ax4.plot(t, ale, color='orange', label=f'Aleatoric (μ={np.mean(ale):.1f})', lw=1.5, alpha=0.8)
    ax4.plot(t, total, 'b--', label=f'Total (μ={np.mean(total):.1f})', lw=1.5, alpha=0.6)
    ax4.set_xlabel('Time Step', fontsize=10)
    ax4.set_ylabel('Uncertainty (σ)', fontsize=10)
    ax4.set_title('Uncertainty Components Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # =====================
    # Row 3: Epistemic analysis and error distribution
    # =====================
    
    # 5. Epistemic ratio over time
    ax5 = fig.add_subplot(gs[2, 0])
    ratio = results['epistemic_ratio'][:n_points]
    ax5.plot(t, ratio, 'purple', alpha=0.7, lw=1.5)
    ax5.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='Equal (0.5)')
    ax5.fill_between(t, 0, ratio, alpha=0.3, color='purple')
    ax5.set_xlabel('Time Step', fontsize=10)
    ax5.set_ylabel('Ratio', fontsize=10)
    ax5.set_title('Epistemic / Total Ratio', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Error distribution with fit
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(errors, bins=50, alpha=0.7, color='steelblue', density=True, edgecolor='white')
    mu_fit, std_fit = np.mean(errors), np.std(errors)
    x_fit = np.linspace(errors.min(), errors.max(), 100)
    ax6.plot(x_fit, norm.pdf(x_fit, mu_fit, std_fit), 'r-', lw=2, 
            label=f'N({mu_fit:.1f}, {std_fit:.1f})')
    ax6.axvline(x=0, color='green', ls='--', lw=2, alpha=0.7)
    ax6.set_xlabel('Prediction Error', fontsize=10)
    ax6.set_ylabel('Density', fontsize=10)
    ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Uncertainty histograms
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(ale, bins=30, alpha=0.6, color='orange', 
            label=f'Aleatoric', density=True, edgecolor='white')
    ax7.hist(epi, bins=30, alpha=0.6, color='green', 
            label=f'Epistemic', density=True, edgecolor='white')
    ax7.set_xlabel('Uncertainty Value', fontsize=10)
    ax7.set_ylabel('Density', fontsize=10)
    ax7.set_title('Uncertainty Distributions', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Scatter: True vs Predicted
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.scatter(y_true[:n_points], pred, alpha=0.4, s=15, c='blue', edgecolors='none')
    lims = [min(y_true[:n_points].min(), pred.min()),
            max(y_true[:n_points].max(), pred.max())]
    ax8.plot(lims, lims, 'r--', lw=2, label='Perfect')
    r2 = np.corrcoef(y_true[:n_points], pred)[0, 1] ** 2
    ax8.set_xlabel('Ground Truth', fontsize=10)
    ax8.set_ylabel('Prediction', fontsize=10)
    ax8.set_title(f'True vs Predicted (R²={r2:.4f})', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # =====================
    # Row 4: Uncertainty quality and calibration
    # =====================
    
    # 9. Total Uncertainty vs Error
    ax9 = fig.add_subplot(gs[3, 0])
    ax9.scatter(total, abs_errors, alpha=0.4, s=15, c='blue', edgecolors='none')
    z = np.polyfit(total, abs_errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(total.min(), total.max(), 100)
    corr_total = np.corrcoef(total, abs_errors)[0, 1]
    ax9.plot(x_line, p(x_line), 'r-', lw=2, label=f'Fit (r={corr_total:.3f})')
    ax9.set_xlabel('Total Uncertainty (σ)', fontsize=10)
    ax9.set_ylabel('|Error|', fontsize=10)
    ax9.set_title('Total Uncertainty vs |Error|', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    # 10. Epistemic vs Error
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.scatter(epi, abs_errors, alpha=0.4, s=15, c='green', edgecolors='none')
    z_epi = np.polyfit(epi, abs_errors, 1)
    p_epi = np.poly1d(z_epi)
    x_line_epi = np.linspace(epi.min(), epi.max(), 100)
    corr_epi = np.corrcoef(epi, abs_errors)[0, 1]
    ax10.plot(x_line_epi, p_epi(x_line_epi), 'r-', lw=2, label=f'Fit (r={corr_epi:.3f})')
    ax10.set_xlabel('Epistemic Uncertainty', fontsize=10)
    ax10.set_ylabel('|Error|', fontsize=10)
    ax10.set_title('Epistemic vs |Error|', fontsize=12, fontweight='bold')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    # 11. Calibration curve
    ax11 = fig.add_subplot(gs[3, 2])
    expected = metrics['calib_expected']
    observed = metrics['calib_observed']
    
    ax11.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect')
    ax11.plot(expected, observed, 'bo-', lw=2, ms=8, 
            label=f'Model (ECE={metrics["ECE"]:.4f})')
    ax11.fill_between(expected, expected, observed, alpha=0.3, color='blue')
    ax11.set_xlabel('Expected Confidence', fontsize=10)
    ax11.set_ylabel('Observed Coverage', fontsize=10)
    ax11.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax11.legend(fontsize=9)
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.set_aspect('equal')
    ax11.grid(True, alpha=0.3)
    
    # 12. Coverage bar chart
    ax12 = fig.add_subplot(gs[3, 3])
    levels = [50, 75, 90, 95]
    expected_cov = [c/100 for c in levels]
    observed_cov = [metrics[f'PICP_{c}'] for c in levels]
    
    x_pos = np.arange(len(levels))
    width = 0.35
    bars1 = ax12.bar(x_pos - width/2, expected_cov, width, label='Expected', 
                     color='lightblue', edgecolor='steelblue')
    bars2 = ax12.bar(x_pos + width/2, observed_cov, width, label='Observed', 
                     color='steelblue', edgecolor='navy')
    
    # Add value labels
    for bar, val in zip(bars2, observed_cov):
        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax12.set_xticks(x_pos)
    ax12.set_xticklabels([f'{c}%' for c in levels])
    ax12.set_xlabel('Confidence Level', fontsize=10)
    ax12.set_ylabel('Coverage', fontsize=10)
    ax12.set_title('Coverage Analysis', fontsize=12, fontweight='bold')
    ax12.legend(fontsize=9)
    ax12.set_ylim(0, 1.1)
    ax12.grid(True, alpha=0.3, axis='y')
    
    # =====================
    # Row 5: Residuals and Pinball analysis
    # =====================
    
    # 13. Residuals with uncertainty bands
    ax13 = fig.add_subplot(gs[4, 0:2])
    ax13.plot(t, errors, 'g-', alpha=0.7, lw=1)
    ax13.fill_between(t, -2*total, 2*total, alpha=0.15, color='red', label='±2σ (95%)')
    ax13.fill_between(t, -total, total, alpha=0.25, color='blue', label='±1σ (68%)')
    ax13.axhline(y=0, color='black', ls='-', lw=1)
    ax13.set_xlabel('Time Step', fontsize=10)
    ax13.set_ylabel('Residual (True - Pred)', fontsize=10)
    ax13.set_title('Residuals with Uncertainty Bands', fontsize=12, fontweight='bold')
    ax13.legend(loc='upper right', fontsize=9)
    ax13.grid(True, alpha=0.3)
    
    # 14. Pinball Loss comparison
    ax14 = fig.add_subplot(gs[4, 2])
    pinball_levels = ['75%', '50%', '25%']
    pinball_values = [metrics['Pinball_75'], metrics['Pinball_50'], metrics['Pinball_25']]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    bars = ax14.bar(pinball_levels, pinball_values, color=colors, edgecolor='black', alpha=0.8)
    for bar, val in zip(bars, pinball_values):
        ax14.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax14.set_xlabel('Prediction Interval', fontsize=10)
    ax14.set_ylabel('Pinball Loss (Winkler Score)', fontsize=10)
    ax14.set_title('Pinball Loss by Interval', fontsize=12, fontweight='bold')
    ax14.grid(True, alpha=0.3, axis='y')
    
    # 15. Summary statistics table
    ax15 = fig.add_subplot(gs[4, 3])
    ax15.axis('off')
    
    # Create summary table
    summary_text = "═══════════════════════════════\n"
    summary_text += "     DECOMPOSITION SUMMARY     \n"
    summary_text += "═══════════════════════════════\n\n"
    summary_text += f"{'Component':<12} {'Mean':>8} {'Std':>8}\n"
    summary_text += f"{'-'*12} {'-'*8} {'-'*8}\n"
    summary_text += f"{'Epistemic':<12} {metrics['UQ_mean_epistemic']:>8.2f} {metrics['UQ_std_epistemic']:>8.2f}\n"
    summary_text += f"{'Aleatoric':<12} {metrics['UQ_mean_aleatoric']:>8.2f} {metrics['UQ_std_aleatoric']:>8.2f}\n"
    summary_text += f"{'Total':<12} {metrics['UQ_mean_total']:>8.2f} {'-':>8}\n\n"
    summary_text += f"Epistemic Ratio: {metrics['UQ_epistemic_ratio']:.4f}\n\n"
    summary_text += "═══════════════════════════════\n"
    summary_text += "      CORRELATION ANALYSIS     \n"
    summary_text += "═══════════════════════════════\n\n"
    summary_text += f"Corr(Total, |Err|):    {metrics['UQ_corr_total_error']:+.4f}\n"
    summary_text += f"Corr(Epi, |Err|):      {metrics['UQ_corr_epistemic_error']:+.4f}\n"
    summary_text += f"Corr(Ale, |Err|):      {metrics['UQ_corr_aleatoric_error']:+.4f}\n"
    summary_text += f"Spearman(Total):       {metrics['UQ_spearman_total_error']:+.4f}\n"
    
    ax15.text(0.05, 0.95, summary_text, transform=ax15.transAxes, fontsize=9,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.9, edgecolor='gray'))
    
    plt.suptitle('DiffLoad v3 - Comprehensive Uncertainty Decomposition Results', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive results saved to: {save_path}")


# =========================
# Training
# =========================
def train_model(model: DiffLoadV3, train_loader: DataLoader,
                val_loader: DataLoader, diff_schedule: DiffusionSchedule,
                save_path: str, epochs: int = 300, patience: int = 15,
                lr: float = 5e-4, lambda_diff: float = 1.0, lambda_calib: float = 0.1):
    
    model.set_diffusion_schedule(diff_schedule)
    model.to(device)
    
    loss_fn = DecomposedUncertaintyLoss(lambda_diff=lambda_diff, lambda_calib=lambda_calib)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )
    
    history = {'train_loss': [], 'val_rmse': [], 'nll': [], 'diff': [], 'calib': []}
    
    best_val_rmse = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Training with decomposed loss (λ_diff={lambda_diff}, λ_calib={lambda_calib})...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': [], 'nll': [], 'diff': [], 'calib': []}
        
        for data, label in train_loader:
            data = data.to(device)
            label = label.squeeze().to(device)
            
            outputs = model(data, use_diffusion=True)
            total_loss, loss_dict = loss_fn(label, outputs)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses['total'].append(loss_dict['total_loss'])
            epoch_losses['nll'].append(loss_dict['nll_loss'])
            epoch_losses['diff'].append(loss_dict['diff_loss'])
            epoch_losses['calib'].append(loss_dict['calib_loss'])
        
        # Validation
        model.eval()
        val_rmse_list = []
        
        with torch.no_grad():
            for data, label in val_loader:
                data = data.to(device)
                label = label.squeeze().to(device)
                
                outputs = model(data, use_diffusion=False)
                pred = outputs['main_value'] + outputs['residual_mu']
                rmse = torch.sqrt(torch.mean((label - pred) ** 2)).item()
                val_rmse_list.append(rmse)
        
        train_loss = np.mean(epoch_losses['total'])
        val_rmse = np.mean(val_rmse_list)
        
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_rmse)
        history['nll'].append(np.mean(epoch_losses['nll']))
        history['diff'].append(np.mean(epoch_losses['diff']))
        history['calib'].append(np.mean(epoch_losses['calib']))
        
        scheduler.step(val_rmse)
        
        scales = model.uncertainty_head.get_scales()
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse': val_rmse,
                'scales': scales,
            }, f"{save_path}/best_model.pth")
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val: {val_rmse:.4f} | "
                  f"Scales: ale={scales['aleatoric_scale']:.3f}, epi={scales['epistemic_scale']:.3f} | ✓")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val: {val_rmse:.4f} | "
                      f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    print(f"\nTraining completed in {time.time() - start_time:.2f}s")
    print(f"Best validation RMSE: {best_val_rmse:.4f}")
    
    # Plot history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_rmse'], color='orange')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['nll'], label='NLL', color='blue')
    axes[1, 0].plot(history['calib'], label='Calib', color='green')
    axes[1, 0].set_title('NLL & Calibration Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['diff'], color='red')
    axes[1, 1].set_title('Diffusion Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_history.png", dpi=150)
    plt.close()
    
    return history


# =========================
# Testing with Enhanced Output
# =========================
def test_model(model: DiffLoadV3, test_data: np.ndarray, test_labels: np.ndarray,
               sd_data: StandardScaler, sd_label: StandardScaler,
               diff_schedule: DiffusionSchedule, save_path: str, n_mc: int = 100):
    
    model.set_diffusion_schedule(diff_schedule)
    model.to(device)
    model.eval()
    
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    
    print(f"\n{'='*60}")
    print(f"Running inference with {n_mc} MC samples...")
    print(f"{'='*60}")
    
    start_time = time.time()
    results = model.predict_with_uncertainty(test_tensor, n_samples=n_mc)
    print(f"Inference completed in {time.time() - start_time:.2f}s")
    print(f"Scales: {results['scales']}")
    
    # Extract and transform
    pred = results['prediction'].numpy()
    epi = results['epistemic_total'].numpy()
    ale = results['aleatoric'].numpy()
    total = results['total_std'].numpy()
    epi_ratio = results['epistemic_ratio'].numpy()
    mc_samples = results['mc_samples']
    
    # Inverse transform
    scale = sd_label.scale_[0]
    mean = sd_label.mean_[0]
    
    pred_orig = pred * scale + mean
    test_orig = test_labels * scale + mean
    epi_orig = epi * scale
    ale_orig = ale * scale
    total_orig = total * scale
    mc_orig = mc_samples * scale + mean
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    metrics = {}
    
    # Point prediction
    metrics['MAE'] = UncertaintyMetrics.mae(test_orig, pred_orig)
    metrics['MAPE'] = UncertaintyMetrics.mape(test_orig, pred_orig)
    metrics['RMSE'] = UncertaintyMetrics.rmse(test_orig, pred_orig)
    
    # Probabilistic
    metrics['CRPS_Gaussian'] = UncertaintyMetrics.crps_gaussian(test_orig, pred_orig, total_orig)
    metrics['CRPS_Empirical'] = UncertaintyMetrics.crps_empirical(test_orig, mc_orig)
    
    # Pinball Loss (all three levels)
    metrics['Pinball_75'] = UncertaintyMetrics.pinball_loss(
        test_orig, pred_orig, total_orig, [0.125, 0.875]) / 0.25
    metrics['Pinball_50'] = UncertaintyMetrics.pinball_loss(
        test_orig, pred_orig, total_orig, [0.25, 0.75]) / 0.5
    metrics['Pinball_25'] = UncertaintyMetrics.pinball_loss(
        test_orig, pred_orig, total_orig, [0.375, 0.625]) / 0.75
    
    # Coverage
    coverage = UncertaintyMetrics.coverage_metrics(test_orig, pred_orig, total_orig)
    for k, v in coverage.items():
        metrics[k] = v
    
    # ECE
    ece, expected, observed = UncertaintyMetrics.ece_parametric(test_orig, pred_orig, total_orig)
    metrics['ECE'] = ece
    metrics['calib_expected'] = expected
    metrics['calib_observed'] = observed
    
    # Uncertainty quality
    errors = test_orig - pred_orig
    uq = UncertaintyMetrics.uncertainty_quality(epi_orig, ale_orig, errors)
    for k, v in uq.items():
        metrics[f'UQ_{k}'] = v
    
    # ==========================================
    # ENHANCED FORMATTED OUTPUT
    # ==========================================
    print(f"\n{'═'*70}")
    print(f"{'TEST RESULTS SUMMARY':^70}")
    print(f"{'═'*70}")
    
    print(f"\n{'─'*70}")
    print(f"{'POINT PREDICTION METRICS':^70}")
    print(f"{'─'*70}")
    print(f"  {'MAE':<35} {metrics['MAE']:>15.4f}")
    print(f"  {'MAPE (%)':<35} {metrics['MAPE']:>15.4f}")
    print(f"  {'RMSE':<35} {metrics['RMSE']:>15.4f}")
    
    print(f"\n{'─'*70}")
    print(f"{'PROBABILISTIC METRICS (CRPS)':^70}")
    print(f"{'─'*70}")
    print(f"  {'CRPS (Gaussian)':<35} {metrics['CRPS_Gaussian']:>15.4f}")
    print(f"  {'CRPS (Empirical/MC)':<35} {metrics['CRPS_Empirical']:>15.4f}")
    
    print(f"\n{'─'*70}")
    print(f"{'PINBALL LOSS / WINKLER SCORE':^70}")
    print(f"{'─'*70}")
    print(f"  {'Interval':<20} {'Quantiles':<25} {'Score':>15}")
    print(f"  {'-'*20} {'-'*25} {'-'*15}")
    print(f"  {'75% PI':<20} {'α = 0.125, 0.875':<25} {metrics['Pinball_75']:>15.4f}")
    print(f"  {'50% PI':<20} {'α = 0.25, 0.75':<25} {metrics['Pinball_50']:>15.4f}")
    print(f"  {'25% PI':<20} {'α = 0.375, 0.625':<25} {metrics['Pinball_25']:>15.4f}")
    
    print(f"\n{'─'*70}")
    print(f"{'PREDICTION INTERVAL COVERAGE':^70}")
    print(f"{'─'*70}")
    print(f"  {'CI Level':<12} {'Expected':>10} {'Observed':>10} {'Gap':>10} {'MPIW':>12} {'Status':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    for c in [50, 75, 90, 95]:
        picp = metrics[f'PICP_{c}']
        gap = metrics[f'Gap_{c}']
        mpiw = metrics[f'MPIW_{c}']
        status = "✓" if abs(gap) < 0.05 else ("↑ Over" if gap > 0 else "↓ Under")
        print(f"  {c}% CI{'':<6} {c/100:>10.2f} {picp:>10.4f} {gap:>+10.4f} {mpiw:>12.2f} {status:>8}")
    
    print(f"\n{'─'*70}")
    print(f"{'UNCERTAINTY CALIBRATION':^70}")
    print(f"{'─'*70}")
    print(f"  {'Expected Calibration Error (ECE)':<45} {metrics['ECE']:>12.4f}")
    quality = 'Excellent' if metrics['ECE'] < 0.05 else 'Good' if metrics['ECE'] < 0.1 else 'Fair' if metrics['ECE'] < 0.2 else 'Poor'
    print(f"  {'Calibration Quality':<45} {quality:>12}")
    
    print(f"\n{'─'*70}")
    print(f"{'UNCERTAINTY DECOMPOSITION QUALITY':^70}")
    print(f"{'─'*70}")
    print(f"  {'Pearson Corr (Total Unc, |Error|)':<45} {metrics['UQ_corr_total_error']:>+12.4f}")
    print(f"  {'Pearson Corr (Epistemic, |Error|)':<45} {metrics['UQ_corr_epistemic_error']:>+12.4f}")
    print(f"  {'Pearson Corr (Aleatoric, |Error|)':<45} {metrics['UQ_corr_aleatoric_error']:>+12.4f}")
    print(f"  {'Spearman Rank (Total, |Error|)':<45} {metrics['UQ_spearman_total_error']:>+12.4f}")
    print(f"  {'Spearman Rank (Epistemic, |Error|)':<45} {metrics['UQ_spearman_epistemic_error']:>+12.4f}")
    
    print(f"\n{'─'*70}")
    print(f"{'UNCERTAINTY DECOMPOSITION STATISTICS':^70}")
    print(f"{'─'*70}")
    print(f"  {'Component':<18} {'Mean':>12} {'Std':>12} {'Ratio':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Epistemic':<18} {metrics['UQ_mean_epistemic']:>12.2f} {metrics['UQ_std_epistemic']:>12.2f} {metrics['UQ_epistemic_ratio']:>12.4f}")
    print(f"  {'Aleatoric':<18} {metrics['UQ_mean_aleatoric']:>12.2f} {metrics['UQ_std_aleatoric']:>12.2f} {1-metrics['UQ_epistemic_ratio']:>12.4f}")
    print(f"  {'Total':<18} {metrics['UQ_mean_total']:>12.2f} {'-':>12} {'1.0000':>12}")
    
    print(f"\n{'═'*70}\n")
    
    # Prepare results
    results_output = {
        'prediction': pred_orig,
        'epistemic': epi_orig,
        'aleatoric': ale_orig,
        'total_std': total_orig,
        'epistemic_ratio': epi_ratio,
        'mc_samples': mc_orig,
    }
    
    # Save
    np.savez(f"{save_path}/predictions.npz",
             prediction=pred_orig, true=test_orig,
             epistemic=epi_orig, aleatoric=ale_orig, total=total_orig)
    
    metrics_save = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    pd.DataFrame([metrics_save]).to_csv(f"{save_path}/metrics.csv", index=False)
    
    # Plot
    print("Generating visualizations...")
    plot_results(test_orig, results_output, metrics, f"{save_path}/comprehensive_results.png")
    
    print(f"All results saved to: {save_path}")
    
    return metrics, results_output


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description='DiffLoad v3 Enhanced')
    parser.add_argument("--mode", type=str, default="both", choices=["train", "test", "both"])
    parser.add_argument("--diff_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mc_samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_diff", type=float, default=1.0)
    parser.add_argument("--lambda_calib", type=float, default=0.1)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"DiffLoad v3 Enhanced - Explicit Uncertainty Decomposition")
    print(f"{'='*70}")
    print(f"Model: d={args.d_model}, h={args.n_heads}, l={args.n_layers}")
    print(f"Loss: λ_diff={args.lambda_diff}, λ_calib={args.lambda_calib}")
    print(f"{'='*70}\n")
    
    diff_schedule = DiffusionSchedule(args.diff_steps, device)
    root_path = f"./diffload_v3_enhanced"
    
    for run_id in range(1, args.runs + 1):
        print(f"\n{'#'*70}")
        print(f"# Run {run_id}/{args.runs}")
        print(f"{'#'*70}")
        
        setup_seed(run_id)
        model_path, result_path = create_folders(root_path, run_id)
        
        # Load data
        print("\nLoading data...")
        train_data = np.load('../GEF_data/train_data.npy')#COV
        train_label = np.load('../GEF_data/train_label.npy')
        val_data = np.load('../GEF_data/val_data.npy')
        val_label = np.load('../GEF_data/val_label.npy')
        test_data = np.load('../GEF_data/test_data.npy')
        test_label = np.load('../GEF_data/test_label.npy')
        
        seq_len = train_data.shape[1]
        n_var = train_data.shape[2]
        print(f"Data: {train_data.shape}")
        
        # Standardize
        sd_data = StandardScaler().fit(train_data.reshape(-1, n_var))
        sd_label = StandardScaler().fit(train_label.reshape(-1, 1))
        
        train_norm = sd_data.transform(train_data.reshape(-1, n_var)).reshape(-1, seq_len, n_var)
        val_norm = sd_data.transform(val_data.reshape(-1, n_var)).reshape(-1, seq_len, n_var)
        test_norm = sd_data.transform(test_data.reshape(-1, n_var)).reshape(-1, seq_len, n_var)
        
        train_label_norm = sd_label.transform(train_label.reshape(-1, 1)).flatten()
        val_label_norm = sd_label.transform(val_label.reshape(-1, 1)).flatten()
        test_label_norm = sd_label.transform(test_label.reshape(-1, 1)).flatten()
        
        train_ds = TimeSeriesDataset(train_norm, train_label_norm)
        val_ds = TimeSeriesDataset(val_norm, val_label_norm)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        # Model
        model = DiffLoadV3(
            seq_len=seq_len, n_variates=n_var,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            num_diff_steps=args.diff_steps, dropout=0.1
        )
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if args.mode in ["train", "both"]:
            train_model(
                model, train_loader, val_loader, diff_schedule,
                model_path, epochs=args.epochs, lr=args.lr,
                lambda_diff=args.lambda_diff, lambda_calib=args.lambda_calib
            )
        
        if args.mode in ["test", "both"]:
            ckpt = torch.load(f"{model_path}/best_model.pth")
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"\nLoaded model from epoch {ckpt['epoch']}")
            
            test_model(
                model, test_norm, test_label_norm,
                sd_data, sd_label, diff_schedule,
                result_path, n_mc=args.mc_samples
            )
    
    print(f"\n{'='*70}")
    print("Completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
