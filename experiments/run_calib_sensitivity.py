"""
λ_calib 敏感性分析 - 自动批量运行
================================

运行方式：
    python run_calib_sensitivity.py

将自动运行 λ_calib = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0] 共6组实验，
并将结果汇总保存到 CSV 文件。
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd

# 要测试的 λ_calib 值
LAMBDA_CALIB_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

# 其他固定参数
FIXED_PARAMS = {
    'lambda_diff': 1.0,
    'epochs': 300,
    'batch_size': 256,
    'lr': 5e-4,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'diff_steps': 5,
    'mc_samples': 100,
}

def run_single_experiment(lambda_calib, result_dir):
    """运行单个实验"""
    
    # 构建命令
    cmd = [
        'python', 'diffload_uncertainty_v3_enhanced.py',
        '--mode', 'both',
        '--lambda_calib', str(lambda_calib),
        '--lambda_diff', str(FIXED_PARAMS['lambda_diff']),
        '--epochs', str(FIXED_PARAMS['epochs']),
        '--batch_size', str(FIXED_PARAMS['batch_size']),
        '--lr', str(FIXED_PARAMS['lr']),
        '--d_model', str(FIXED_PARAMS['d_model']),
        '--n_heads', str(FIXED_PARAMS['n_heads']),
        '--n_layers', str(FIXED_PARAMS['n_layers']),
        '--diff_steps', str(FIXED_PARAMS['diff_steps']),
        '--mc_samples', str(FIXED_PARAMS['mc_samples']),
    ]
    
    print(f"\n{'='*70}")
    print(f"Running experiment: λ_calib = {lambda_calib}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    # 运行
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error running experiment with λ_calib = {lambda_calib}")
        return None
    
    # 复制结果到专门的目录
    src_dir = './diffload_v3_enhanced/results/run_1'
    dst_dir = f'{result_dir}/calib_{lambda_calib}'
    
    os.makedirs(dst_dir, exist_ok=True)
    os.system(f'cp -r {src_dir}/* {dst_dir}/')
    
    # 读取metrics
    metrics_file = f'{dst_dir}/metrics.csv'
    if os.path.exists(metrics_file):
        metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
        metrics['lambda_calib'] = lambda_calib
        return metrics
    
    return None


def main():
    # 创建结果目录
    result_dir = './ablation_calib_sensitivity'
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# λ_calib Sensitivity Analysis")
    print(f"# Testing values: {LAMBDA_CALIB_VALUES}")
    print(f"# Results will be saved to: {result_dir}")
    print(f"{'#'*70}")
    
    all_results = []
    
    for lambda_calib in LAMBDA_CALIB_VALUES:
        metrics = run_single_experiment(lambda_calib, result_dir)
        if metrics:
            all_results.append(metrics)
            
            # 打印当前结果
            print(f"\n>>> Results for λ_calib = {lambda_calib}:")
            print(f"    MAE:      {metrics.get('MAE', 'N/A'):.2f}")
            print(f"    CRPS:     {metrics.get('CRPS_Gaussian', 'N/A'):.2f}")
            print(f"    ECE:      {metrics.get('ECE', 'N/A'):.4f}")
            print(f"    Corr:     {metrics.get('UQ_corr_epistemic_error', 'N/A'):.4f}")
            print(f"    PICP@90%: {metrics.get('PICP_90', 'N/A')*100:.1f}%")
    
    # 汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # 选择关键列
        key_cols = ['lambda_calib', 'MAE', 'CRPS_Gaussian', 'ECE', 
                    'UQ_corr_epistemic_error', 'PICP_90']
        available_cols = [c for c in key_cols if c in summary_df.columns]
        summary_df = summary_df[available_cols]
        
        # 重命名列
        summary_df.columns = ['λ_calib', 'MAE', 'CRPS', 'ECE', 'Corr', 'PICP@90%']
        
        # 保存
        summary_file = f'{result_dir}/summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY OF ALL EXPERIMENTS")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"\nResults saved to: {summary_file}")
        print(f"{'='*70}")
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
