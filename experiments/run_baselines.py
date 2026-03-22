"""
Baseline Comparison Experiments
===============================

运行方式：
    python run_baselines.py

将自动运行3个基线方法并输出汇总结果：
1. iTransformer (No UQ) - 确定性基线
2. MC Dropout - 隐式UQ
3. Deep Ensemble - 集成方法

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
# 基础组件
# =========================

def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))
    
    def __len__(self):
        return len(self.data)


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


# =========================
# Baseline 1: iTransformer (No UQ)
# =========================
class iTransformerNoUQ(nn.Module):
    """确定性基线：只有点预测，无不确定性"""
    
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = iTransformerEncoder(seq_len, n_variates, d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        hidden = self.encoder(x)
        return self.head(hidden).squeeze(-1)


# =========================
# Baseline 2: MC Dropout
# =========================
class iTransformerMCDropout(nn.Module):
    """MC Dropout: 训练和推理时都使用dropout"""
    
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = iTransformerEncoder(seq_len, n_variates, d_model, n_heads, n_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        hidden = self.encoder(x)
        hidden = self.dropout(hidden)  # 推理时也保持dropout
        return self.head(hidden).squeeze(-1)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """MC采样获取不确定性"""
        self.train()  # 保持dropout激活
        
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred.cpu())
        
        samples = torch.stack(samples, dim=0)
        
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        
        return {
            'prediction': mean,
            'total_std': std,
            'mc_samples': samples.numpy()
        }


# =========================
# Baseline 3: Deep Ensemble
# =========================
class DeepEnsemble:
    """Deep Ensemble: 5个独立模型的集成"""
    
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4, n_layers=2, 
                 dropout=0.1, n_members=5):
        self.n_members = n_members
        self.models = [
            iTransformerNoUQ(seq_len, n_variates, d_model, n_heads, n_layers, dropout)
            for _ in range(n_members)
        ]
    
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
    
    def train_member(self, member_idx, train_loader, val_loader, epochs=300, lr=5e-4):
        """训练单个成员"""
        model = self.models[member_idx]
        model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            for data, label in train_loader:
                data, label = data.to(device), label.squeeze().to(device)
                pred = model(data)
                loss = F.mse_loss(pred, label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.to(device), label.squeeze().to(device)
                    pred = model(data)
                    val_losses.append(F.mse_loss(pred, label).item())
            
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                break
        
        model.load_state_dict(best_state)
        return model
    
    def train_all(self, train_loader, val_loader, epochs=300, lr=5e-4):
        """训练所有成员"""
        for i in range(self.n_members):
            print(f"  Training ensemble member {i+1}/{self.n_members}...")
            setup_seed(i + 1)  # 不同的随机种子
            self.train_member(i, train_loader, val_loader, epochs, lr)
    
    def predict_with_uncertainty(self, x):
        """集成预测"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(device))
                predictions.append(pred.cpu())
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return {
            'prediction': mean,
            'total_std': std,
            'mc_samples': predictions.numpy()
        }


# =========================
# 评估指标
# =========================
def compute_metrics(test_orig, pred_orig, total_orig=None, mc_samples=None):
    metrics = {}
    
    # Point metrics
    metrics['MAE'] = np.mean(np.abs(test_orig - pred_orig))
    metrics['RMSE'] = np.sqrt(np.mean((test_orig - pred_orig) ** 2))
    metrics['MAPE'] = np.mean(np.abs((test_orig - pred_orig) / (np.abs(test_orig) + 1e-8))) * 100
    
    # Probabilistic metrics (if uncertainty available)
    if total_orig is not None:
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
        
        # PICP@90%
        z90 = norm.ppf(0.95)
        lower90 = pred_orig - z90 * total_orig
        upper90 = pred_orig + z90 * total_orig
        metrics['PICP_90'] = np.mean((test_orig >= lower90) & (test_orig <= upper90))
        
        # Corr (uncertainty vs error)
        errors = np.abs(test_orig - pred_orig)
        if np.std(total_orig) > 1e-8:
            metrics['Corr'] = np.corrcoef(total_orig.flatten(), errors.flatten())[0, 1]
        else:
            metrics['Corr'] = 0.0
    else:
        metrics['CRPS'] = '-'
        metrics['ECE'] = '-'
        metrics['PICP_90'] = '-'
        metrics['Corr'] = '-'
    
    return metrics


# =========================
# 训练和测试函数
# =========================
def train_deterministic(model, train_loader, val_loader, epochs=300, lr=5e-4):
    """训练确定性模型"""
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.squeeze().to(device)
            pred = model(data)
            loss = F.mse_loss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.squeeze().to(device)
                pred = model(data)
                val_losses.append(F.mse_loss(pred, label).item())
        
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= 15:
            break
    
    model.load_state_dict(best_state)
    return model


def test_deterministic(model, test_data, test_labels, sd_label):
    """测试确定性模型"""
    model.to(device)
    model.eval()
    
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred = model(test_tensor).cpu().numpy()
    
    scale = sd_label.scale_[0]
    mean = sd_label.mean_[0]
    
    pred_orig = pred * scale + mean
    test_orig = test_labels * scale + mean
    
    return compute_metrics(test_orig, pred_orig)


def test_mc_dropout(model, test_data, test_labels, sd_label, n_mc=100):
    """测试MC Dropout模型"""
    model.to(device)
    
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    results = model.predict_with_uncertainty(test_tensor, n_samples=n_mc)
    
    scale = sd_label.scale_[0]
    mean = sd_label.mean_[0]
    
    pred_orig = results['prediction'].numpy() * scale + mean
    test_orig = test_labels * scale + mean
    total_orig = results['total_std'].numpy() * scale
    mc_orig = results['mc_samples'] * scale + mean
    
    return compute_metrics(test_orig, pred_orig, total_orig, mc_orig)


def test_ensemble(ensemble, test_data, test_labels, sd_label):
    """测试Deep Ensemble"""
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    results = ensemble.predict_with_uncertainty(test_tensor)
    
    scale = sd_label.scale_[0]
    mean = sd_label.mean_[0]
    
    pred_orig = results['prediction'].numpy() * scale + mean
    test_orig = test_labels * scale + mean
    total_orig = results['total_std'].numpy() * scale
    
    return compute_metrics(test_orig, pred_orig, total_orig)


# =========================
# Main
# =========================
def main():
    print(f"\n{'#'*70}")
    print(f"# Baseline Comparison Experiments")
    print(f"{'#'*70}\n")
    
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
    
    all_results = []
    
    # =====================
    # Baseline 1: iTransformer (No UQ)
    # =====================
    print(f"\n{'='*60}")
    print("Baseline 1: iTransformer (No UQ)")
    print(f"{'='*60}")
    
    setup_seed(42)
    model_no_uq = iTransformerNoUQ(seq_len, n_var, d_model=64, n_heads=4, n_layers=2)
    model_no_uq = train_deterministic(model_no_uq, train_loader, val_loader)
    metrics_no_uq = test_deterministic(model_no_uq, test_norm, test_label_norm, sd_label)
    metrics_no_uq['Method'] = 'iTransformer (No UQ)'
    all_results.append(metrics_no_uq)
    
    print(f"  MAE:  {metrics_no_uq['MAE']:.2f}")
    print(f"  RMSE: {metrics_no_uq['RMSE']:.2f}")
    print(f"  MAPE: {metrics_no_uq['MAPE']:.2f}%")
    
    # =====================
    # Baseline 2: MC Dropout
    # =====================
    print(f"\n{'='*60}")
    print("Baseline 2: MC Dropout")
    print(f"{'='*60}")
    
    setup_seed(42)
    model_mc = iTransformerMCDropout(seq_len, n_var, d_model=64, n_heads=4, n_layers=2, dropout=0.1)
    model_mc = train_deterministic(model_mc, train_loader, val_loader)
    metrics_mc = test_mc_dropout(model_mc, test_norm, test_label_norm, sd_label, n_mc=100)
    metrics_mc['Method'] = 'MC Dropout'
    all_results.append(metrics_mc)
    
    print(f"  MAE:     {metrics_mc['MAE']:.2f}")
    print(f"  RMSE:    {metrics_mc['RMSE']:.2f}")
    print(f"  CRPS:    {metrics_mc['CRPS']:.2f}")
    print(f"  ECE:     {metrics_mc['ECE']:.4f}")
    print(f"  PICP@90: {metrics_mc['PICP_90']*100:.1f}%")
    
    # =====================
    # Baseline 3: Deep Ensemble
    # =====================
    print(f"\n{'='*60}")
    print("Baseline 3: Deep Ensemble (5 members)")
    print(f"{'='*60}")
    
    ensemble = DeepEnsemble(seq_len, n_var, d_model=64, n_heads=4, n_layers=2, n_members=5)
    ensemble.train_all(train_loader, val_loader)
    metrics_ens = test_ensemble(ensemble, test_norm, test_label_norm, sd_label)
    metrics_ens['Method'] = 'Deep Ensemble'
    all_results.append(metrics_ens)
    
    print(f"  MAE:     {metrics_ens['MAE']:.2f}")
    print(f"  RMSE:    {metrics_ens['RMSE']:.2f}")
    print(f"  CRPS:    {metrics_ens['CRPS']:.2f}")
    print(f"  ECE:     {metrics_ens['ECE']:.4f}")
    print(f"  PICP@90: {metrics_ens['PICP_90']*100:.1f}%")
    
    # =====================
    # 汇总输出
    # =====================
    print(f"\n{'='*70}")
    print(f"SUMMARY: Baseline Comparison")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'CRPS':>8} {'ECE':>8} {'PICP@90':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for r in all_results:
        crps_str = f"{r['CRPS']:.2f}" if isinstance(r['CRPS'], float) else r['CRPS']
        ece_str = f"{r['ECE']:.4f}" if isinstance(r['ECE'], float) else r['ECE']
        picp_str = f"{r['PICP_90']*100:.1f}%" if isinstance(r['PICP_90'], float) else r['PICP_90']
        
        print(f"{r['Method']:<25} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {r['MAPE']:>8.2f} {crps_str:>8} {ece_str:>8} {picp_str:>8}")
    
    print(f"{'='*70}")
    
    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_csv('./baseline_results.csv', index=False)
    print(f"\nResults saved to: ./baseline_results.csv")
    
    print("\n" + "="*70)
    print("Note: Add DiffLoad-UQ results manually for comparison")
    print("DiffLoad-UQ: MAE=109.54, RMSE=150.37, CRPS=79.39, ECE=0.021, PICP@90=88.8%")
    print("="*70)


if __name__ == "__main__":
    main()
