"""
Ablation Study Script: Effect of Explicit Uncertainty Head
===========================================================

运行方式：
    python ablation_explicit_head.py --variant mc_only
    python ablation_explicit_head.py --variant explicit_only  
    python ablation_explicit_head.py --variant no_calib

变体说明：
1. mc_only: 移除显式UQ头，仅用MC采样方差估计不确定性
2. explicit_only: 仅用显式预测，不用MC采样方差
3. no_calib: 有显式头但移除校准损失 (lambda_calib=0)

完整模型已经运行过，数据在 diffload_v3_enhanced/results/run_1/
"""

import os
import sys
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
from scipy.stats import norm, spearmanr
import properscoring as ps

import time
import argparse
import matplotlib.pyplot as plt

# -------------------------
# Device
# -------------------------
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# 从原文件复制的基础组件
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
    
    def q_sample(self, x_0: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_0)
        x_t = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        return x_t, noise
    
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: int, 
                 add_noise: bool = True) -> torch.Tensor:
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
    
    def compute_loss(self, model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_steps, size=(batch_size // 2,)).to(self.device)
        t = torch.cat([t, self.num_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)
        
        noise = torch.randn_like(x_0)
        x_noisy = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        
        predicted_noise = model(x_noisy, t.squeeze(-1))
        return F.mse_loss(predicted_noise, noise)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# =========================
# 原始显式UQ头 (用于 explicit_only 和 no_calib)
# =========================
class ExplicitUncertaintyHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        
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
        
        self.log_aleatoric_scale = nn.Parameter(torch.tensor(0.0))
        self.log_epistemic_scale = nn.Parameter(torch.tensor(-1.0))
        
        self.sigma_min = 0.01
        self.sigma_max = 3.0
    
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    
    def get_scales(self) -> Dict[str, float]:
        return {
            'aleatoric_scale': torch.exp(self.log_aleatoric_scale).item(),
            'epistemic_scale': torch.exp(self.log_epistemic_scale).item(),
        }


# =========================
# 变体1: MC Only Head (移除显式UQ分支)
# =========================
class MCOnlyHead(nn.Module):
    """
    只有预测头，没有显式的uncertainty分支。
    不确定性完全来自MC采样的方差。
    """
    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.main_head = nn.Linear(hidden_dim, output_dim)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_feat = self.shared(hidden)
        
        main_value = self.main_head(shared_feat).squeeze(-1)
        residual_mu = self.mu_head(shared_feat).squeeze(-1)
        
        # 没有显式的uncertainty输出，返回零占位
        batch_size = hidden.shape[0]
        zeros = torch.zeros(batch_size, device=hidden.device)
        
        return {
            'main_value': main_value,
            'residual_mu': residual_mu,
            'aleatoric': zeros,  # 占位
            'epistemic': zeros,  # 占位
        }
    
    def get_scales(self) -> Dict[str, float]:
        return {'aleatoric_scale': 0.0, 'epistemic_scale': 0.0}


# =========================
# 模型变体
# =========================
class DiffLoadAblation(nn.Module):
    """
    消融实验模型：支持不同的UQ头配置
    """
    
    def __init__(self, seq_len: int, n_variates: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, num_diff_steps: int = 5,
                 dropout: float = 0.1, variant: str = 'full'):
        super().__init__()
        
        self.d_model = d_model
        self.num_diff_steps = num_diff_steps
        self.variant = variant
        
        self.encoder = iTransformerEncoder(
            seq_len=seq_len,
            n_variates=n_variates,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        self.diffusion_net = MLPDiffusion(num_diff_steps, d_model)
        
        # 根据变体选择不同的UQ头
        if variant == 'mc_only':
            self.uncertainty_head = MCOnlyHead(d_model, output_dim=1)
        else:
            self.uncertainty_head = ExplicitUncertaintyHead(d_model, output_dim=1)
        
        self.diff_schedule: Optional[DiffusionSchedule] = None
    
    def set_diffusion_schedule(self, schedule: DiffusionSchedule):
        self.diff_schedule = schedule
    
    def forward(self, x: torch.Tensor, use_diffusion: bool = True
                ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        
        diff_loss = torch.tensor(0.0).to(x.device)
        
        if use_diffusion and self.diff_schedule is not None:
            hidden_noisy, _ = self.diff_schedule.q_sample(hidden, self.num_diff_steps - 1)
            diff_loss = self.diff_schedule.compute_loss(self.diffusion_net, hidden)
            hidden_denoised = self.diff_schedule.p_sample(
                self.diffusion_net, hidden_noisy, self.num_diff_steps - 1
            )
        else:
            hidden_denoised = hidden
        
        outputs = self.uncertainty_head(hidden_denoised)
        outputs['diff_loss'] = diff_loss
        outputs['hidden'] = hidden_denoised
        
        return outputs
    
    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100
                                  ) -> Dict[str, torch.Tensor]:
        self.eval()
        
        pred_samples = []
        aleatoric_samples = []
        epistemic_samples = []
        
        for _ in range(n_samples):
            outputs = self.forward(x, use_diffusion=True)
            pred_samples.append((outputs['main_value'] + outputs['residual_mu']).cpu())
            aleatoric_samples.append(outputs['aleatoric'].cpu())
            epistemic_samples.append(outputs['epistemic'].cpu())
        
        pred_samples = torch.stack(pred_samples, dim=0)
        aleatoric_samples = torch.stack(aleatoric_samples, dim=0)
        epistemic_samples = torch.stack(epistemic_samples, dim=0)
        
        prediction = pred_samples.mean(dim=0)
        mc_variance = pred_samples.var(dim=0)
        mc_std = torch.sqrt(mc_variance)
        
        if self.variant == 'mc_only':
            # MC Only: 不确定性完全来自MC方差
            # 假设 epistemic = mc_std, aleatoric = 固定小值
            epistemic_total = mc_std
            aleatoric_pred = torch.full_like(mc_std, 0.1)  # 小的固定值
        elif self.variant == 'explicit_only':
            # Explicit Only: 不使用MC方差
            epistemic_total = epistemic_samples.mean(dim=0)
            aleatoric_pred = aleatoric_samples.mean(dim=0)
        else:
            # Full 或 no_calib: 混合模式
            epistemic_pred = epistemic_samples.mean(dim=0)
            aleatoric_pred = aleatoric_samples.mean(dim=0)
            epistemic_total = torch.sqrt(epistemic_pred**2 + mc_std**2)
        
        total_std = torch.sqrt(aleatoric_pred**2 + epistemic_total**2)
        epistemic_ratio = epistemic_total / (total_std + 1e-8)
        
        return {
            'prediction': prediction,
            'aleatoric': aleatoric_pred,
            'epistemic_total': epistemic_total,
            'total_std': total_std,
            'epistemic_ratio': epistemic_ratio,
            'mc_samples': pred_samples.numpy(),
        }


# =========================
# 损失函数变体
# =========================
class AblationLoss(nn.Module):
    def __init__(self, lambda_diff: float = 1.0, lambda_calib: float = 0.1, 
                 variant: str = 'full'):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_calib = lambda_calib
        self.variant = variant
    
    def forward(self, target: torch.Tensor, outputs: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        main_value = outputs['main_value']
        mu = outputs['residual_mu']
        aleatoric = outputs['aleatoric']
        epistemic = outputs['epistemic']
        diff_loss = outputs['diff_loss']
        
        prediction = main_value + mu
        residual = target - prediction
        
        if self.variant == 'mc_only':
            # MC Only: 只用MSE损失 + diffusion损失
            mse_loss = F.mse_loss(prediction, target)
            total_loss = mse_loss + self.lambda_diff * diff_loss
            
            return total_loss, {
                'total_loss': total_loss.item(),
                'mse_loss': mse_loss.item(),
                'diff_loss': diff_loss.item(),
                'nll_loss': 0.0,
                'calib_loss': 0.0,
            }
        
        # 有显式UQ头的变体
        total_var = aleatoric**2 + epistemic**2 + 1e-6
        
        nll = 0.5 * (torch.log(total_var) + residual**2 / total_var)
        nll_loss = nll.mean()
        
        if self.variant == 'no_calib':
            # 无校准损失
            calib_loss = torch.tensor(0.0).to(target.device)
        else:
            # 有校准损失
            abs_residual = torch.abs(residual)
            calib_loss = F.mse_loss(epistemic, (abs_residual * 0.5).detach())
        
        total_loss = nll_loss + self.lambda_diff * diff_loss + self.lambda_calib * calib_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'nll_loss': nll_loss.item(),
            'diff_loss': diff_loss.item(),
            'calib_loss': calib_loss.item() if isinstance(calib_loss, torch.Tensor) else calib_loss,
        }


# =========================
# 评估指标
# =========================
class UncertaintyMetrics:
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        return np.mean(ps.crps_gaussian(y_true, mu, sigma))
    
    @staticmethod
    def ece_parametric(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                       n_bins: int = 10) -> float:
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
        return ece
    
    @staticmethod
    def picp(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, conf: float = 0.90) -> float:
        z = norm.ppf((1 + conf) / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        return np.mean((y_true >= lower) & (y_true <= upper))
    
    @staticmethod
    def corr_epi_error(epistemic: np.ndarray, errors: np.ndarray) -> float:
        abs_errors = np.abs(errors)
        return np.corrcoef(epistemic.flatten(), abs_errors.flatten())[0, 1]


# =========================
# 训练和测试
# =========================
def train_model(model, train_loader, val_loader, diff_schedule,
                save_path, epochs=300, lr=5e-4, lambda_diff=1.0, 
                lambda_calib=0.1, variant='full'):
    
    model.to(device)
    model.set_diffusion_schedule(diff_schedule)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = AblationLoss(lambda_diff=lambda_diff, lambda_calib=lambda_calib, variant=variant)
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    print(f"\nTraining variant: {variant}")
    print(f"lambda_diff={lambda_diff}, lambda_calib={lambda_calib}")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, use_diffusion=True)
            loss, loss_dict = loss_fn(batch_y, outputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss_dict['total_loss'])
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x, use_diffusion=True)
                _, loss_dict = loss_fn(batch_y, outputs)
                val_losses.append(loss_dict['total_loss'])
        
        val_loss = np.mean(val_losses)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, f"{save_path}/best_model.pth")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {np.mean(train_losses):.4f} | Val: {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    return best_val_loss


def test_model(model, test_data, test_labels, sd_data, sd_label,
               diff_schedule, save_path, n_mc=100, variant='full'):
    
    model.to(device)
    model.set_diffusion_schedule(diff_schedule)
    model.eval()
    
    test_tensor = torch.FloatTensor(test_data).to(device)
    
    print(f"\nRunning inference with {n_mc} MC samples...")
    results = model.predict_with_uncertainty(test_tensor, n_samples=n_mc)
    
    # 转换回原始尺度
    pred_orig = sd_label.inverse_transform(results['prediction'].numpy().reshape(-1, 1)).flatten()
    test_orig = sd_label.inverse_transform(test_labels.reshape(-1, 1)).flatten()
    
    scale = sd_label.scale_[0]
    epi_orig = results['epistemic_total'].numpy() * scale
    ale_orig = results['aleatoric'].numpy() * scale
    total_orig = results['total_std'].numpy() * scale
    
    mc_orig = results['mc_samples'] * scale + sd_label.mean_[0]
    
    # 计算指标
    metrics = {}
    metrics['MAE'] = UncertaintyMetrics.mae(test_orig, pred_orig)
    metrics['CRPS'] = UncertaintyMetrics.crps_gaussian(test_orig, pred_orig, total_orig)
    metrics['ECE'] = UncertaintyMetrics.ece_parametric(test_orig, pred_orig, total_orig)
    metrics['PICP_90'] = UncertaintyMetrics.picp(test_orig, pred_orig, total_orig, 0.90)
    
    errors = test_orig - pred_orig
    metrics['Corr'] = UncertaintyMetrics.corr_epi_error(epi_orig, errors)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"ABLATION RESULTS: {variant}")
    print(f"{'='*60}")
    print(f"  MAE:       {metrics['MAE']:.2f}")
    print(f"  CRPS:      {metrics['CRPS']:.2f}")
    print(f"  ECE:       {metrics['ECE']:.4f}")
    print(f"  Corr:      {metrics['Corr']:.4f}")
    print(f"  PICP@90%:  {metrics['PICP_90']*100:.1f}%")
    print(f"{'='*60}")
    
    # 保存结果
    pd.DataFrame([metrics]).to_csv(f"{save_path}/ablation_metrics_{variant}.csv", index=False)
    
    return metrics


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description='Ablation Study: Explicit Uncertainty Head')
    parser.add_argument("--variant", type=str, required=True, 
                        choices=['mc_only', 'explicit_only', 'no_calib'],
                        help="Ablation variant to run")
    parser.add_argument("--data_path", type=str, default="../GEF_data",
                        help="Path to data folder (GEF_data or COV_data)")
    parser.add_argument("--diff_steps", type=int, default=5)
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
    
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {args.variant}")
    print(f"{'='*60}")
    
    # 设置路径
    root_path = f"./ablation_{args.variant}"
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(f"{root_path}/models", exist_ok=True)
    os.makedirs(f"{root_path}/results", exist_ok=True)
    
    model_path = f"{root_path}/models"
    result_path = f"{root_path}/results"
    
    setup_seed(42)
    
    # 加载数据
    print(f"\nLoading data from {args.data_path}...")
    train_data = np.load(f'{args.data_path}/train_data.npy')
    train_label = np.load(f'{args.data_path}/train_label.npy')
    val_data = np.load(f'{args.data_path}/val_data.npy')
    val_label = np.load(f'{args.data_path}/val_label.npy')
    test_data = np.load(f'{args.data_path}/test_data.npy')
    test_label = np.load(f'{args.data_path}/test_label.npy')
    
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
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Diffusion schedule
    diff_schedule = DiffusionSchedule(args.diff_steps, device)
    
    # 创建模型
    model = DiffLoadAblation(
        seq_len=seq_len, n_variates=n_var,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        num_diff_steps=args.diff_steps, dropout=0.1,
        variant=args.variant
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 设置lambda_calib
    lambda_calib = 0.0 if args.variant == 'no_calib' else args.lambda_calib
    
    # 训练
    train_model(
        model, train_loader, val_loader, diff_schedule,
        model_path, epochs=args.epochs, lr=args.lr,
        lambda_diff=args.lambda_diff, lambda_calib=lambda_calib,
        variant=args.variant
    )
    
    # 测试
    ckpt = torch.load(f"{model_path}/best_model.pth")
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"\nLoaded model from epoch {ckpt['epoch']}")
    
    metrics = test_model(
        model, test_norm, test_label_norm,
        sd_data, sd_label, diff_schedule,
        result_path, n_mc=args.mc_samples, variant=args.variant
    )
    
    print(f"\nResults saved to: {result_path}")
    print("Done!")


if __name__ == "__main__":
    main()
