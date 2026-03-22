"""
Gaussian Process (GP) Baseline for Load Forecasting
====================================================

运行方式：
    python run_gp_baseline.py

使用 scikit-learn 的 GaussianProcessRegressor
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from scipy.stats import norm
import properscoring as ps
import time

# =========================
# 指标函数
# =========================
def MAE(true, pred):
    return np.mean(np.abs(true - pred))

def MAPE(true, pred):
    return np.mean(np.abs((pred - true) / true)) * 100

def CRPS(true, mu, sigma):
    return np.mean(ps.crps_gaussian(true, mu, sigma))

def ECE(true, pred, sigma, n_bins=10):
    """Expected Calibration Error"""
    confidence_levels = np.linspace(0.1, 0.95, n_bins)
    observed_coverage = []
    
    for conf in confidence_levels:
        z = norm.ppf((1 + conf) / 2)
        lower = pred - z * sigma
        upper = pred + z * sigma
        coverage = np.mean((true >= lower) & (true <= upper))
        observed_coverage.append(coverage)
    
    return np.mean(np.abs(confidence_levels - np.array(observed_coverage)))

def PICP(true, pred, sigma, confidence=0.90):
    """Prediction Interval Coverage Probability"""
    z = norm.ppf((1 + confidence) / 2)
    lower = pred - z * sigma
    upper = pred + z * sigma
    return np.mean((true >= lower) & (true <= upper))


# =========================
# Main
# =========================
def main():
    print("="*60)
    print("Gaussian Process (GP) Baseline")
    print("="*60)
    
    # 加载数据
    print("\nLoading data...")
    train_data = np.load('../GEF_data/train_data.npy')
    train_label = np.load('../GEF_data/train_label.npy')
    test_data = np.load('../GEF_data/test_data.npy')
    test_label = np.load('../GEF_data/test_label.npy')
    
    n_samples, seq_len, n_features = train_data.shape
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    # 展平时序数据为特征向量
    train_data_flat = train_data.reshape(train_data.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)
    
    # 标准化
    sd_data = StandardScaler().fit(train_data_flat)
    sd_label = StandardScaler().fit(train_label.reshape(-1, 1))
    
    X_train = sd_data.transform(train_data_flat)
    X_test = sd_data.transform(test_data_flat)
    y_train = sd_label.transform(train_label.reshape(-1, 1)).flatten()
    y_test_norm = sd_label.transform(test_label.reshape(-1, 1)).flatten()
    
    # 由于GP计算复杂度高 O(n³)，我们需要降采样或使用稀疏GP
    # 这里使用子采样训练数据
    max_train_samples = 2000  # GP对大数据集计算量大
    if len(X_train) > max_train_samples:
        print(f"Subsampling training data: {len(X_train)} -> {max_train_samples}")
        indices = np.random.choice(len(X_train), max_train_samples, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train
    
    # 定义GP核函数
    # Matern核通常比RBF更适合时序数据
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    
    print("\nTraining Gaussian Process...")
    print(f"Kernel: {kernel}")
    
    start_time = time.time()
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    
    gp.fit(X_train_sub, y_train_sub)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    print(f"Optimized kernel: {gp.kernel_}")
    
    # 预测
    print("\nRunning inference...")
    start_time = time.time()
    
    # 分批预测以避免内存问题
    batch_size = 500
    y_pred_list = []
    y_std_list = []
    
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_pred_batch, y_std_batch = gp.predict(X_batch, return_std=True)
        y_pred_list.append(y_pred_batch)
        y_std_list.append(y_std_batch)
    
    y_pred_norm = np.concatenate(y_pred_list)
    y_std_norm = np.concatenate(y_std_list)
    
    infer_time = time.time() - start_time
    print(f"Inference completed in {infer_time:.2f}s")
    
    # 反归一化
    y_pred = sd_label.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    y_true = test_label.flatten()
    y_std = y_std_norm * sd_label.scale_[0]  # 标准差也需要缩放
    
    # 计算指标
    test_MAE = MAE(y_true, y_pred)
    test_MAPE = MAPE(y_true, y_pred)
    test_CRPS = CRPS(y_true, y_pred, y_std)
    test_ECE = ECE(y_true, y_pred, y_std)
    test_PICP90 = PICP(y_true, y_pred, y_std, confidence=0.90)
    
    # 输出结果
    print("\n" + "="*60)
    print("POINT PREDICTION METRICS")
    print("="*60)
    print(f"MAE:  {test_MAE:.2f}")
    print(f"MAPE: {test_MAPE:.2f}%")
    
    print("="*60)
    print("PROBABILISTIC METRICS")
    print("="*60)
    print(f"CRPS: {test_CRPS:.2f}")
    
    print("="*60)
    print("UNCERTAINTY CALIBRATION")
    print("="*60)
    print(f"ECE:      {test_ECE:.4f}")
    print(f"PICP@90%: {test_PICP90*100:.1f}%")
    
    print("\n" + "="*60)
    print("GP - FINAL SUMMARY")
    print("="*60)
    print(f"MAE:         {test_MAE:.2f}")
    print(f"MAPE:        {test_MAPE:.2f}%")
    print(f"CRPS:        {test_CRPS:.2f}")
    print(f"ECE:         {test_ECE:.4f}")
    print(f"PICP@90%:    {test_PICP90*100:.1f}%")
    print("="*60)
    
    # 保存结果
    results = {
        'MAE': test_MAE,
        'MAPE': test_MAPE,
        'CRPS': test_CRPS,
        'ECE': test_ECE,
        'PICP_90': test_PICP90
    }
    
    pd.DataFrame([results]).to_csv('./gp_baseline_results.csv', index=False)
    print("\nResults saved to: ./gp_baseline_results.csv")


if __name__ == "__main__":
    np.random.seed(42)
    main()
