"""
Ablation Study: Effect of Learnable Scaling Factors
====================================================

运行方式：
    python run_gamma_ablation.py

将自动运行5个变体的实验并输出汇总结果。
"""

import os
import sys
import random
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import properscoring as ps

import time

# -------------------------
# Device
# -------------------------
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# 基础组件（从原文件复制）
# =========================

def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
    
    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        return x_t, noise
    
    def p_sample(self, model, x, t, add_noise=True):
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
    
    def compute_loss(self, model, x_0):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_steps, size=(batch_size // 2,)).to(self.device)
        t = torch.cat([t, self.num_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)
        noise = torch.randn_like(x_0)
        x_noisy = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        predicted_noise = model(x_noisy, t.squeeze(-1))
        return F.mse_loss(predicted_noise, noise)


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, hidden_dim, mlp_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
        ])
        self.time_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, mlp_dim),
            nn.Embedding(n_steps, mlp_dim),
            nn.Embedding(n_steps, mlp_dim),
        ])
    
    def forward(self, x, t):
        for idx, time_emb in enumerate(self.time_embeddings):
            t_emb = time_emb(t.to(x.device))
            x = self.layers[2 * idx](x)
            x = x + t_emb
            x = self.layers[2 * idx + 1](x)
        return self.layers[-1](x)


class iTransformerEncoder(nn.Module):
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.variate_embedding = nn.Linear(seq_len, d_model)
        self.variate_pos = nn.Parameter(torch.zeros(1, n_variates, d_model))
        nn.init.trunc_normal_(self.variate_pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        tokens = self.variate_embedding(x)
        tokens = tokens + self.variate_pos
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        tokens = tokens + self.temporal_ffn(tokens)
        return tokens[:, 0, :]


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))
    
    def __len__(self):
        return len(self.data)


# =========================
# 可配置的 UncertaintyHead
# =========================
class ConfigurableUncertaintyHead(nn.Module):
    """
    支持不同γ配置的UncertaintyHead
    
    gamma_config: 
        'both_learnable' - 两个γ都可学习 (默认)
        'both_fixed_zero' - 两个γ都固定为0
        'both_fixed_init' - γ_ale=0, γ_epi=-1 固定
        'ale_learnable_only' - 只有γ_ale可学习
        'epi_learnable_only' - 只有γ_epi可学习
    """
    
    def __init__(self, hidden_dim, output_dim=1, gamma_config='both_learnable'):
        super().__init__()
        self.gamma_config = gamma_config
        
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.main_head = nn.Linear(hidden_dim, output_dim)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # 根据配置设置γ
        if gamma_config == 'both_learnable':
            self.log_aleatoric_scale = nn.Parameter(torch.tensor(0.0))
            self.log_epistemic_scale = nn.Parameter(torch.tensor(-1.0))
        elif gamma_config == 'both_fixed_zero':
            self.register_buffer('log_aleatoric_scale', torch.tensor(0.0))
            self.register_buffer('log_epistemic_scale', torch.tensor(0.0))
        elif gamma_config == 'both_fixed_init':
            self.register_buffer('log_aleatoric_scale', torch.tensor(0.0))
            self.register_buffer('log_epistemic_scale', torch.tensor(-1.0))
        elif gamma_config == 'ale_learnable_only':
            self.log_aleatoric_scale = nn.Parameter(torch.tensor(0.0))
            self.register_buffer('log_epistemic_scale', torch.tensor(-1.0))
        elif gamma_config == 'epi_learnable_only':
            self.register_buffer('log_aleatoric_scale', torch.tensor(0.0))
            self.log_epistemic_scale = nn.Parameter(torch.tensor(-1.0))
        
        self.sigma_min = 0.01
        self.sigma_max = 3.0
    
    def forward(self, hidden):
        shared_feat = self.shared(hidden)
        
        main_value = self.main_head(shared_feat).squeeze(-1)
        residual_mu = self.mu_head(shared_feat).squeeze(-1)
        
        aleatoric_raw = self.aleatoric_head(shared_feat).squeeze(-1)
        aleatoric_scale = torch.exp(self.log_aleatoric_scale)
        aleatoric = torch.sigmoid(aleatoric_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min
        aleatoric = aleatoric * aleatoric_scale
        
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
    
    def get_scales(self):
        return {
            'aleatoric_scale': torch.exp(self.log_aleatoric_scale).item(),
            'epistemic_scale': torch.exp(self.log_epistemic_scale).item(),
        }


class DiffLoadGammaAblation(nn.Module):
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4, n_layers=2,
                 num_diff_steps=5, dropout=0.1, gamma_config='both_learnable'):
        super().__init__()
        self.d_model = d_model
        self.num_diff_steps = num_diff_steps
        
        self.encoder = iTransformerEncoder(
            seq_len=seq_len, n_variates=n_variates,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        self.diffusion_net = MLPDiffusion(num_diff_steps, d_model)
        self.uncertainty_head = ConfigurableUncertaintyHead(d_model, output_dim=1, gamma_config=gamma_config)
        self.diff_schedule = None
    
    def set_diffusion_schedule(self, schedule):
        self.diff_schedule = schedule
    
    def forward(self, x, use_diffusion=True):
        hidden = self.encoder(x)
        diff_loss = torch.tensor(0.0).to(x.device)
        
        if use_diffusion and self.diff_schedule is not None:
            hidden_noisy, _ = self.diff_schedule.q_sample(hidden, self.num_diff_steps - 1)
            diff_loss = self.diff_schedule.compute_loss(self.diffusion_net, hidden)
            hidden_denoised = self.diff_schedule.p_sample(self.diffusion_net, hidden_noisy, self.num_diff_steps - 1)
        else:
            hidden_denoised = hidden
        
        outputs = self.uncertainty_head(hidden_denoised)
        outputs['diff_loss'] = diff_loss
        return outputs
    
    @torch.no_grad()
    def predict_with_uncertainty(self, x, n_samples=100):
        self.eval()
        pred_samples, aleatoric_samples, epistemic_samples = [], [], []
        
        for _ in range(n_samples):
            outputs = self.forward(x, use_diffusion=True)
            pred_samples.append((outputs['main_value'] + outputs['residual_mu']).cpu())
            aleatoric_samples.append(outputs['aleatoric'].cpu())
            epistemic_samples.append(outputs['epistemic'].cpu())
        
        pred_samples = torch.stack(pred_samples, dim=0)
        aleatoric_samples = torch.stack(aleatoric_samples, dim=0)
        epistemic_samples = torch.stack(epistemic_samples, dim=0)
        
        prediction = pred_samples.mean(dim=0)
        aleatoric_pred = aleatoric_samples.mean(dim=0)
        epistemic_pred = epistemic_samples.mean(dim=0)
        mc_std = torch.sqrt(pred_samples.var(dim=0))
        
        epistemic_total = torch.sqrt(epistemic_pred**2 + mc_std**2)
        total_std = torch.sqrt(aleatoric_pred**2 + epistemic_total**2)
        epistemic_ratio = epistemic_total**2 / (total_std**2 + 1e-8)
        
        return {
            'prediction': prediction,
            'aleatoric': aleatoric_pred,
            'epistemic_total': epistemic_total,
            'total_std': total_std,
            'epistemic_ratio': epistemic_ratio,
            'mc_samples': pred_samples.numpy(),
        }


# =========================
# Loss
# =========================
class DecomposedUncertaintyLoss(nn.Module):
    def __init__(self, lambda_diff=1.0, lambda_calib=0.1):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_calib = lambda_calib
    
    def forward(self, target, outputs):
        main_value = outputs['main_value']
        mu = outputs['residual_mu']
        aleatoric = outputs['aleatoric']
        epistemic = outputs['epistemic']
        diff_loss = outputs['diff_loss']
        
        prediction = main_value + mu
        residual = target - prediction
        
        total_var = aleatoric**2 + epistemic**2 + 1e-6
        nll = 0.5 * (torch.log(total_var) + residual**2 / total_var)
        nll_loss = nll.mean()
        
        abs_residual = torch.abs(residual)
        calib_loss = F.mse_loss(epistemic, (abs_residual * 0.5).detach())
        
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
def compute_metrics(test_orig, pred_orig, total_orig, epi_orig, ale_orig, mc_orig):
    metrics = {}
    
    # Point
    metrics['MAE'] = np.mean(np.abs(test_orig - pred_orig))
    
    # CRPS
    metrics['CRPS'] = np.mean(ps.crps_gaussian(test_orig, pred_orig, total_orig))
    
    # ECE
    confidence_levels = np.linspace(0.1, 0.95, 10)
    observed_coverage = []
    for conf in confidence_levels:
        z = norm.ppf((1 + conf) / 2)
        lower = pred_orig - z * total_orig
        upper = pred_orig + z * total_orig
        coverage = np.mean((test_orig >= lower) & (test_orig <= upper))
        observed_coverage.append(coverage)
    metrics['ECE'] = np.mean(np.abs(confidence_levels - np.array(observed_coverage)))
    
    # Corr
    errors = np.abs(test_orig - pred_orig)
    metrics['Corr'] = np.corrcoef(epi_orig.flatten(), errors.flatten())[0, 1]
    
    # R_epi
    total_var = epi_orig**2 + ale_orig**2
    metrics['R_epi'] = np.mean(epi_orig**2 / (total_var + 1e-8))
    
    return metrics


# =========================
# Train & Test
# =========================
def train_model(model, train_loader, val_loader, diff_schedule, epochs=300, lr=5e-4):
    model.set_diffusion_schedule(diff_schedule)
    model.to(device)
    
    loss_fn = DecomposedUncertaintyLoss(lambda_diff=1.0, lambda_calib=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_rmse = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.squeeze().to(device)
            outputs = model(data, use_diffusion=True)
            total_loss, _ = loss_fn(label, outputs)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_rmse_list = []
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.squeeze().to(device)
                outputs = model(data, use_diffusion=False)
                pred = outputs['main_value'] + outputs['residual_mu']
                rmse = torch.sqrt(torch.mean((label - pred) ** 2)).item()
                val_rmse_list.append(rmse)
        
        val_rmse = np.mean(val_rmse_list)
        scheduler.step(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= 15:
            break
    
    model.load_state_dict(best_state)
    return model


def test_model(model, test_data, test_labels, sd_label, diff_schedule, n_mc=100):
    model.set_diffusion_schedule(diff_schedule)
    model.to(device)
    model.eval()
    
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    results = model.predict_with_uncertainty(test_tensor, n_samples=n_mc)
    
    scale = sd_label.scale_[0]
    mean = sd_label.mean_[0]
    
    pred_orig = results['prediction'].numpy() * scale + mean
    test_orig = test_labels * scale + mean
    epi_orig = results['epistemic_total'].numpy() * scale
    ale_orig = results['aleatoric'].numpy() * scale
    total_orig = results['total_std'].numpy() * scale
    mc_orig = results['mc_samples'] * scale + mean
    
    metrics = compute_metrics(test_orig, pred_orig, total_orig, epi_orig, ale_orig, mc_orig)
    
    return metrics


# =========================
# Main
# =========================
def main():
    print(f"\n{'#'*70}")
    print(f"# Ablation Study: Effect of Learnable Scaling Factors")
    print(f"{'#'*70}\n")
    
    # 配置
    gamma_configs = [
        ('both_learnable', 'Learnable γ (default)'),
        ('both_fixed_zero', 'Fixed γ_ale = γ_epi = 0'),
        ('both_fixed_init', 'Fixed γ_ale = 0, γ_epi = -1'),
        ('ale_learnable_only', 'Learnable γ_ale only'),
        ('epi_learnable_only', 'Learnable γ_epi only'),
    ]
    
    setup_seed(42)
    
    # 加载数据
    print("Loading data...")
    train_data = np.load('../GEF_data/train_data.npy')
    train_label = np.load('../GEF_data/train_label.npy')
    val_data = np.load('../GEF_data/val_data.npy')
    val_label = np.load('../GEF_data/val_label.npy')
    test_data = np.load('../GEF_data/test_data.npy')
    test_label = np.load('../GEF_data/test_label.npy')
    
    seq_len = train_data.shape[1]
    n_var = train_data.shape[2]
    print(f"Data shape: {train_data.shape}")
    
    # 标准化
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
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    diff_schedule = DiffusionSchedule(5, device)
    
    all_results = []
    
    for gamma_config, config_name in gamma_configs:
        print(f"\n{'='*60}")
        print(f"Running: {config_name}")
        print(f"{'='*60}")
        
        setup_seed(42)
        
        model = DiffLoadGammaAblation(
            seq_len=seq_len, n_variates=n_var,
            d_model=64, n_heads=4, n_layers=2,
            num_diff_steps=5, dropout=0.1,
            gamma_config=gamma_config
        )
        
        model = train_model(model, train_loader, val_loader, diff_schedule, epochs=300)
        metrics = test_model(model, test_norm, test_label_norm, sd_label, diff_schedule)
        
        metrics['Configuration'] = config_name
        all_results.append(metrics)
        
        print(f"  MAE:   {metrics['MAE']:.2f}")
        print(f"  CRPS:  {metrics['CRPS']:.2f}")
        print(f"  ECE:   {metrics['ECE']:.4f}")
        print(f"  Corr:  {metrics['Corr']:.4f}")
        print(f"  R_epi: {metrics['R_epi']:.2f}")
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"SUMMARY: Effect of Learnable Scaling Factors")
    print(f"{'='*70}")
    print(f"{'Configuration':<35} {'MAE':>8} {'CRPS':>8} {'ECE':>8} {'Corr':>8} {'R_epi':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        print(f"{r['Configuration']:<35} {r['MAE']:>8.2f} {r['CRPS']:>8.2f} {r['ECE']:>8.4f} {r['Corr']:>8.4f} {r['R_epi']:>8.2f}")
    print(f"{'='*70}")
    
    # 保存
    df = pd.DataFrame(all_results)
    df.to_csv('./ablation_gamma_results.csv', index=False)
    print(f"\nResults saved to: ./ablation_gamma_results.csv")


if __name__ == "__main__":
    main()
