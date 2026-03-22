"""
Advanced Visualization Module for LoadUQ-Former
===============================================

提供丰富多样的可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, gaussian_kde, pearsonr, spearmanr
from scipy.signal import periodogram, find_peaks
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

#plt.style.use('seaborn-v0_8-whitegrid')
# ---------- robust matplotlib/seaborn style ----------
def set_plot_style():
    candidates = [
        "seaborn-v0_8-whitegrid",  # new matplotlib
        "seaborn-whitegrid",       # old matplotlib
        "seaborn",                 # older alias
        "ggplot",                  # fallback
        "default",
    ]
    available = set(plt.style.available)
    for s in candidates:
        if s in available:
            plt.style.use(s)
            break

    # seaborn theme (optional)
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        pass

set_plot_style()




class PremiumVisualizer:
    """高级可视化器 - 提供丰富多样的图表展示"""
    
    # 颜色方案
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'epistemic': '#3498DB',
        'aleatoric': '#E74C3C',
        'prediction': '#2ECC71',
        'truth': '#9B59B6',
        'background': '#F5F5F5',
        'grid': '#E0E0E0'
    }
    
    def __init__(self, save_path: str, figsize_scale: float = 1.0):
        self.save_path = save_path
        self.figsize_scale = figsize_scale
        os.makedirs(save_path, exist_ok=True)
        
        # 设置默认字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.dpi'] = 300
    
    def _scale_figsize(self, width: float, height: float) -> Tuple[float, float]:
        return (width * self.figsize_scale, height * self.figsize_scale)
    
    def plot_dashboard_style_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epistemic_std: np.ndarray,
        aleatoric_std: np.ndarray,
        metrics: Dict[str, float],
        title: str = "LoadUQ-Former Performance Dashboard"
    ):
        """仪表板风格的综合结果展示"""
        
        fig = plt.figure(figsize=self._scale_figsize(24, 20))
        
        # 创建不规则网格布局
        gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.35)
        
        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)
        n_plot = min(300, len(y_true))
        x = np.arange(n_plot)
        
        # ===== 顶部标题区域 =====
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
        
        # ===== 1. 主预测图 (占据顶部大部分空间) =====
        ax1 = fig.add_subplot(gs[0, :4])
        ax1.plot(x, y_true[:n_plot], color=self.COLORS['truth'], 
                linewidth=2, label='Ground Truth', alpha=0.9)
        ax1.plot(x, y_pred[:n_plot], color=self.COLORS['prediction'], 
                linewidth=1.5, label='Prediction', linestyle='--', alpha=0.9)
        ax1.fill_between(x, 
                        y_pred[:n_plot] - 1.96 * total_std[:n_plot],
                        y_pred[:n_plot] + 1.96 * total_std[:n_plot],
                        alpha=0.25, color=self.COLORS['prediction'], label='95% CI')
        ax1.fill_between(x,
                        y_pred[:n_plot] - 1.0 * total_std[:n_plot],
                        y_pred[:n_plot] + 1.0 * total_std[:n_plot],
                        alpha=0.35, color=self.COLORS['prediction'], label='68% CI')
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Load Value', fontsize=12)
        ax1.set_title('📈 Load Forecasting with Confidence Intervals', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ===== 2. KPI指标卡片 =====
        ax2 = fig.add_subplot(gs[0, 4:])
        ax2.axis('off')
        
        # 创建KPI卡片
        kpi_data = [
            ('MAE', metrics.get('MAE', 0), '📊'),
            ('RMSE', metrics.get('RMSE', 0), '📉'),
            ('MAPE', metrics.get('MAPE', 0), '📈'),
            ('R²', metrics.get('R2', 0), '🎯'),
            ('CRPS', metrics.get('CRPS', 0), '📋'),
            ('Coverage', metrics.get('Coverage_95', 0), '✅')
        ]
        
        for i, (name, value, emoji) in enumerate(kpi_data):
            row = i // 2
            col = i % 2
            x_pos = 0.05 + col * 0.5
            y_pos = 0.85 - row * 0.35
            
            # 绘制KPI卡片
            rect = FancyBboxPatch((x_pos - 0.02, y_pos - 0.12), 0.44, 0.28,
                                  boxstyle="round,pad=0.02,rounding_size=0.02",
                                  facecolor='#F8F9FA', edgecolor='#DEE2E6',
                                  linewidth=2, transform=ax2.transAxes)
            ax2.add_patch(rect)
            
            ax2.text(x_pos + 0.2, y_pos + 0.08, f'{emoji} {name}',
                    transform=ax2.transAxes, fontsize=11, fontweight='bold',
                    ha='center', va='center', color='#495057')
            
            if name == 'Coverage':
                ax2.text(x_pos + 0.2, y_pos - 0.02, f'{value:.1f}%',
                        transform=ax2.transAxes, fontsize=18, fontweight='bold',
                        ha='center', va='center', color=self.COLORS['primary'])
            elif name == 'MAPE':
                ax2.text(x_pos + 0.2, y_pos - 0.02, f'{value:.2f}%',
                        transform=ax2.transAxes, fontsize=18, fontweight='bold',
                        ha='center', va='center', color=self.COLORS['primary'])
            else:
                ax2.text(x_pos + 0.2, y_pos - 0.02, f'{value:.4f}',
                        transform=ax2.transAxes, fontsize=18, fontweight='bold',
                        ha='center', va='center', color=self.COLORS['primary'])
        
        # ===== 3. 不确定性分解堆叠图 =====
        ax3 = fig.add_subplot(gs[1, :3])
        ax3.stackplot(x, 
                     epistemic_std[:n_plot]**2, 
                     aleatoric_std[:n_plot]**2,
                     labels=['Epistemic (Model)', 'Aleatoric (Data)'],
                     colors=[self.COLORS['epistemic'], self.COLORS['aleatoric']],
                     alpha=0.7)
        ax3.set_xlabel('Time Step', fontsize=12)
        ax3.set_ylabel('Variance', fontsize=12)
        ax3.set_title('🔍 Uncertainty Decomposition Over Time', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # ===== 4. 不确定性比例环形图 =====
        ax4 = fig.add_subplot(gs[1, 3])
        total_var = np.mean(epistemic_std**2 + aleatoric_std**2)
        epi_ratio = np.mean(epistemic_std**2) / total_var * 100
        ale_ratio = np.mean(aleatoric_std**2) / total_var * 100
        
        sizes = [epi_ratio, ale_ratio]
        colors_pie = [self.COLORS['epistemic'], self.COLORS['aleatoric']]
        
        # 外环
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=['', ''], autopct='',
            colors=colors_pie, startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white')
        )
        
        # 中心文字
        ax4.text(0, 0, f'{epi_ratio:.0f}%\n÷\n{ale_ratio:.0f}%',
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax4.set_title('🥧 Uncertainty Ratio', fontsize=14, fontweight='bold')
        
        # 图例
        legend_labels = [f'Epistemic: {epi_ratio:.1f}%', f'Aleatoric: {ale_ratio:.1f}%']
        ax4.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=9)
        
        # ===== 5. 散点图 + 密度 =====
        ax5 = fig.add_subplot(gs[1, 4:])
        
        # 散点图着色按不确定性
        scatter = ax5.scatter(y_true[:n_plot], y_pred[:n_plot], 
                             c=total_std[:n_plot], cmap='YlOrRd',
                             alpha=0.6, s=20, edgecolors='none')
        
        # 对角线
        min_val, max_val = y_true.min(), y_true.max()
        ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)
        
        ax5.set_xlabel('Ground Truth', fontsize=12)
        ax5.set_ylabel('Prediction', fontsize=12)
        r2 = metrics.get('R2', 0)
        ax5.set_title(f'📊 Prediction vs Truth (R² = {r2:.4f})', fontsize=14, fontweight='bold')
        
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label='Uncertainty')
        ax5.grid(True, alpha=0.3)
        
        # ===== 6. 残差分布 =====
        ax6 = fig.add_subplot(gs[2, :2])
        residuals = y_true - y_pred
        
        # 直方图 + KDE
        ax6.hist(residuals, bins=50, density=True, alpha=0.7, 
                color=self.COLORS['primary'], edgecolor='white', linewidth=0.5)
        
        # 拟合正态分布
        mu, std = norm.fit(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        ax6.plot(x_norm, norm.pdf(x_norm, mu, std), 
                color=self.COLORS['secondary'], linewidth=2, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
        
        ax6.set_xlabel('Residual', fontsize=12)
        ax6.set_ylabel('Density', fontsize=12)
        ax6.set_title('📉 Residual Distribution', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # ===== 7. 校准曲线 =====
        ax7 = fig.add_subplot(gs[2, 2:4])
        
        quantiles = np.linspace(0.1, 0.9, 9)
        observed_freq = []
        for q in quantiles:
            threshold = norm.ppf(q, y_pred, total_std)
            observed = np.mean(y_true <= threshold)
            observed_freq.append(observed)
        
        ax7.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
        ax7.plot(quantiles, observed_freq, 'o-', 
                color=self.COLORS['primary'], linewidth=2, markersize=8, label='Model')
        ax7.fill_between(quantiles, quantiles - 0.1, quantiles + 0.1, 
                        alpha=0.2, color='gray', label='±10% band')
        
        ax7.set_xlabel('Expected Probability', fontsize=12)
        ax7.set_ylabel('Observed Probability', fontsize=12)
        ax7.set_title('📐 Calibration Curve', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])
        ax7.grid(True, alpha=0.3)
        
        # ===== 8. 覆盖率柱状图 =====
        ax8 = fig.add_subplot(gs[2, 4:])
        
        coverage_levels = [50, 70, 80, 90, 95]
        expected = []
        actual = []
        
        for level in coverage_levels:
            z = norm.ppf(0.5 + level/200)
            lower = y_pred - z * total_std
            upper = y_pred + z * total_std
            coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
            expected.append(level)
            actual.append(coverage)
        
        x_pos = np.arange(len(coverage_levels))
        width = 0.35
        
        bars1 = ax8.bar(x_pos - width/2, expected, width, label='Expected',
                       color=self.COLORS['epistemic'], alpha=0.7)
        bars2 = ax8.bar(x_pos + width/2, actual, width, label='Actual',
                       color=self.COLORS['aleatoric'], alpha=0.7)
        
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels([f'{l}%' for l in coverage_levels])
        ax8.set_xlabel('Confidence Level', fontsize=12)
        ax8.set_ylabel('Coverage (%)', fontsize=12)
        ax8.set_title('✅ Coverage Analysis', fontsize=14, fontweight='bold')
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # ===== 9. 误差 vs 不确定性 =====
        ax9 = fig.add_subplot(gs[3, :2])
        
        errors = np.abs(y_true - y_pred)
        ax9.scatter(total_std, errors, alpha=0.3, s=10, c=self.COLORS['primary'])
        
        # 趋势线
        z = np.polyfit(total_std, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(total_std.min(), total_std.max(), 100)
        ax9.plot(x_line, p(x_line), color=self.COLORS['secondary'], linewidth=2)
        
        corr, pval = spearmanr(total_std, errors)
        
        ax9.set_xlabel('Total Uncertainty', fontsize=12)
        ax9.set_ylabel('Absolute Error', fontsize=12)
        ax9.set_title(f'🔗 Uncertainty-Error Correlation (ρ = {corr:.3f})', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # ===== 10. 时间段误差分布 =====
        ax10 = fig.add_subplot(gs[3, 2:4])
        
        # 分成若干时间段
        n_segments = 5
        segment_size = len(y_true) // n_segments
        segment_errors = []
        segment_labels = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else len(y_true)
            seg_error = np.abs(y_true[start:end] - y_pred[start:end])
            segment_errors.append(seg_error)
            segment_labels.append(f'Seg {i+1}')
        
        bp = ax10.boxplot(segment_errors, labels=segment_labels, patch_artist=True)
        
        colors_box = plt.cm.viridis(np.linspace(0.2, 0.8, n_segments))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax10.set_xlabel('Time Segment', fontsize=12)
        ax10.set_ylabel('Absolute Error', fontsize=12)
        ax10.set_title('📊 Error Distribution by Time Segment', fontsize=14, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')
        
        # ===== 11. 周期性分析 =====
        ax11 = fig.add_subplot(gs[3, 4:])
        
        # 计算频谱
        if len(y_true) > 100:
            freq, psd = periodogram(y_true[:min(1000, len(y_true))])
            
            # 只显示有意义的频率范围
            valid_idx = (freq > 0) & (freq < 0.5)
            ax11.semilogy(freq[valid_idx], psd[valid_idx], 
                         color=self.COLORS['primary'], linewidth=1.5, alpha=0.8)
            
            # 标记峰值
            peaks, _ = find_peaks(psd[valid_idx], height=np.percentile(psd[valid_idx], 90))
            if len(peaks) > 0:
                ax11.scatter(freq[valid_idx][peaks], psd[valid_idx][peaks], 
                           color=self.COLORS['secondary'], s=100, zorder=5, marker='*')
        
        ax11.set_xlabel('Frequency', fontsize=12)
        ax11.set_ylabel('Power Spectral Density', fontsize=12)
        ax11.set_title('🔄 Periodicity Analysis (PSD)', fontsize=14, fontweight='bold')
        ax11.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.save_path}/dashboard_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to {self.save_path}/dashboard_results.png")
    
    def plot_uncertainty_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epistemic_std: np.ndarray,
        aleatoric_std: np.ndarray,
        n_points: int = 200
    ):
        """不确定性热力图可视化"""
        
        fig, axes = plt.subplots(2, 2, figsize=self._scale_figsize(16, 12))
        
        n_points = min(n_points, len(y_true))
        
        # 1. 预测值 vs 不确定性热力图
        ax1 = axes[0, 0]
        total_std = np.sqrt(epistemic_std[:n_points]**2 + aleatoric_std[:n_points]**2)
        
        # 创建2D直方图
        h = ax1.hist2d(y_pred[:n_points], total_std, bins=50, cmap='YlOrRd')
        ax1.set_xlabel('Predicted Value', fontsize=12)
        ax1.set_ylabel('Total Uncertainty', fontsize=12)
        ax1.set_title('Prediction vs Uncertainty Distribution', fontsize=14, fontweight='bold')
        plt.colorbar(h[3], ax=ax1, label='Count')
        
        # 2. 认知不确定性 vs 偶然不确定性
        ax2 = axes[0, 1]
        h2 = ax2.hist2d(epistemic_std[:n_points], aleatoric_std[:n_points], bins=50, cmap='Blues')
        ax2.set_xlabel('Epistemic Uncertainty', fontsize=12)
        ax2.set_ylabel('Aleatoric Uncertainty', fontsize=12)
        ax2.set_title('Uncertainty Components Relationship', fontsize=14, fontweight='bold')
        
        # 添加对角线
        max_val = max(epistemic_std[:n_points].max(), aleatoric_std[:n_points].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5)
        plt.colorbar(h2[3], ax=ax2, label='Count')
        
        # 3. 时间-不确定性热力图
        ax3 = axes[1, 0]
        
        # 重塑为2D (假设有周期性)
        period = 24  # 假设24小时周期
        n_periods = n_points // period
        if n_periods > 1:
            total_std_reshaped = total_std[:n_periods * period].reshape(n_periods, period)
            im = ax3.imshow(total_std_reshaped, aspect='auto', cmap='viridis')
            ax3.set_xlabel('Hour of Day', fontsize=12)
            ax3.set_ylabel('Day Index', fontsize=12)
            ax3.set_title('Time-of-Day Uncertainty Pattern', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax3, label='Uncertainty')
        else:
            ax3.plot(np.arange(n_points), total_std)
            ax3.set_xlabel('Time Step', fontsize=12)
            ax3.set_ylabel('Uncertainty', fontsize=12)
            ax3.set_title('Uncertainty Over Time', fontsize=14, fontweight='bold')
        
        # 4. 误差-不确定性联合分布
        ax4 = axes[1, 1]
        errors = np.abs(y_true[:n_points] - y_pred[:n_points])
        
        # 使用seaborn的联合分布图
        from scipy.stats import gaussian_kde
        
        xy = np.vstack([total_std, errors])
        try:
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            scatter = ax4.scatter(total_std[idx], errors[idx], c=z[idx], s=20, cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, ax=ax4, label='Density')
        except:
            ax4.scatter(total_std, errors, alpha=0.3, s=10)
        
        ax4.set_xlabel('Total Uncertainty', fontsize=12)
        ax4.set_ylabel('Absolute Error', fontsize=12)
        ax4.set_title('Uncertainty-Error Joint Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/uncertainty_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Uncertainty heatmap saved to {self.save_path}/uncertainty_heatmap.png")
    
    def plot_model_comparison_radar(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Performance Comparison"
    ):
        """雷达图比较多个模型的性能"""
        
        # 选择用于比较的指标
        metric_names = ['MAE', 'RMSE', 'MAPE', 'R2', 'CRPS', 'Coverage_95']
        metric_labels = ['MAE ↓', 'RMSE ↓', 'MAPE ↓', 'R² ↑', 'CRPS ↓', 'Coverage ↑']
        
        # 获取所有模型名称
        model_names = list(metrics_dict.keys())
        n_models = len(model_names)
        
        # 提取数据并归一化
        data = []
        for model in model_names:
            model_data = []
            for metric in metric_names:
                value = metrics_dict[model].get(metric, 0)
                model_data.append(value)
            data.append(model_data)
        
        data = np.array(data)
        
        # 归一化到0-1 (对于需要最小化的指标取反)
        data_normalized = np.zeros_like(data)
        for i, metric in enumerate(metric_names):
            col = data[:, i]
            if metric in ['R2', 'Coverage_95']:  # 越大越好
                data_normalized[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
            else:  # 越小越好
                data_normalized[:, i] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-8)
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=self._scale_figsize(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))
        
        for i, (model, color) in enumerate(zip(model_names, colors)):
            values = data_normalized[i].tolist()
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/model_comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar chart saved to {self.save_path}/model_comparison_radar.png")
    
    def plot_prediction_intervals_animation_style(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        n_frames: int = 4,
        n_points: int = 100
    ):
        """多帧展示预测区间变化"""
        
        n_points = min(n_points, len(y_true))
        
        fig, axes = plt.subplots(2, 2, figsize=self._scale_figsize(16, 12))
        axes = axes.flatten()
        
        confidence_levels = [50, 80, 90, 95]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(confidence_levels)))
        
        for idx, (ax, ci) in enumerate(zip(axes, confidence_levels)):
            z = norm.ppf(0.5 + ci/200)
            
            x = np.arange(n_points)
            ax.plot(x, y_true[:n_points], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(x, y_pred[:n_points], 'r--', linewidth=1.5, label='Prediction', alpha=0.8)
            
            lower = y_pred[:n_points] - z * y_std[:n_points]
            upper = y_pred[:n_points] + z * y_std[:n_points]
            ax.fill_between(x, lower, upper, alpha=0.4, color=colors[idx], label=f'{ci}% CI')
            
            # 计算覆盖率
            coverage = np.mean((y_true[:n_points] >= lower) & (y_true[:n_points] <= upper)) * 100
            
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('Load Value', fontsize=11)
            ax.set_title(f'{ci}% Confidence Interval (Coverage: {coverage:.1f}%)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Prediction Intervals at Different Confidence Levels', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/prediction_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction intervals saved to {self.save_path}/prediction_intervals.png")
    
    def plot_error_analysis_comprehensive(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epistemic_std: np.ndarray,
        aleatoric_std: np.ndarray
    ):
        """综合误差分析图"""
        
        fig = plt.figure(figsize=self._scale_figsize(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        residuals = y_true - y_pred
        errors = np.abs(residuals)
        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)
        
        # 1. 残差时间序列
        ax1 = fig.add_subplot(gs[0, :])
        n_plot = min(300, len(residuals))
        x = np.arange(n_plot)
        ax1.fill_between(x, -1.96 * total_std[:n_plot], 1.96 * total_std[:n_plot],
                        alpha=0.3, color='gray', label='±1.96σ band')
        ax1.plot(x, residuals[:n_plot], 'b-', linewidth=0.8, alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Residual', fontsize=12)
        ax1.set_title('Residual Time Series with Uncertainty Band', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差分布 + QQ图
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(residuals, bins=50, density=True, alpha=0.7, color=self.COLORS['primary'],
                edgecolor='white', linewidth=0.5)
        mu, std = norm.fit(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x_norm, norm.pdf(x_norm, mu, std), 'r-', linewidth=2)
        ax2.set_xlabel('Residual', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'Residual Distribution (μ={mu:.2f}, σ={std:.2f})', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        ax3 = fig.add_subplot(gs[1, 1])
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=ax3)
        ax3.get_lines()[0].set_color(self.COLORS['primary'])
        ax3.get_lines()[0].set_markersize(3)
        ax3.get_lines()[1].set_color('red')
        ax3.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 残差自相关
        ax4 = fig.add_subplot(gs[1, 2])
        max_lag = min(50, len(residuals) // 4)
        acf = np.correlate(residuals - np.mean(residuals), 
                          residuals - np.mean(residuals), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        ax4.bar(np.arange(max_lag), acf[:max_lag], color=self.COLORS['primary'], alpha=0.7)
        ax4.axhline(y=1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=-1.96/np.sqrt(len(residuals)), color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Lag', fontsize=12)
        ax4.set_ylabel('Autocorrelation', fontsize=12)
        ax4.set_title('Residual Autocorrelation', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. 残差 vs 预测值
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(y_pred, residuals, alpha=0.3, s=10, c=self.COLORS['primary'])
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Predicted Value', fontsize=12)
        ax5.set_ylabel('Residual', fontsize=12)
        ax5.set_title('Residuals vs Predictions', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 绝对误差分布
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(errors, bins=50, density=True, alpha=0.7, color=self.COLORS['aleatoric'],
                edgecolor='white', linewidth=0.5)
        ax6.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(errors):.2f}')
        ax6.axvline(np.median(errors), color='g', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(errors):.2f}')
        ax6.set_xlabel('Absolute Error', fontsize=12)
        ax6.set_ylabel('Density', fontsize=12)
        ax6.set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # 7. 误差百分位数
        ax7 = fig.add_subplot(gs[2, 2])
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        error_percentiles = np.percentile(errors, percentiles)
        
        ax7.barh(np.arange(len(percentiles)), error_percentiles, 
                color=plt.cm.Reds(np.linspace(0.3, 0.9, len(percentiles))), alpha=0.8)
        ax7.set_yticks(np.arange(len(percentiles)))
        ax7.set_yticklabels([f'{p}th' for p in percentiles])
        ax7.set_xlabel('Absolute Error', fontsize=12)
        ax7.set_ylabel('Percentile', fontsize=12)
        ax7.set_title('Error Percentiles', fontsize=13, fontweight='bold')
        
        # 添加数值标签
        for i, (p, v) in enumerate(zip(percentiles, error_percentiles)):
            ax7.text(v + 0.01 * max(error_percentiles), i, f'{v:.2f}', 
                    va='center', fontsize=9)
        ax7.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Comprehensive Error Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.save_path}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Error analysis saved to {self.save_path}/error_analysis.png")
    
    def create_summary_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epistemic_std: np.ndarray,
        aleatoric_std: np.ndarray,
        metrics: Dict[str, float],
        model_name: str = "LoadUQ-Former"
    ):
        """生成综合报告"""
        
        # 调用所有可视化方法
        print("\n" + "="*60)
        print("Generating Comprehensive Visualization Report...")
        print("="*60 + "\n")
        
        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)
        
        # 1. 仪表板
        print("1/5 Creating dashboard...")
        self.plot_dashboard_style_results(
            y_true, y_pred, epistemic_std, aleatoric_std, metrics,
            title=f"{model_name} Performance Dashboard"
        )
        
        # 2. 不确定性热力图
        print("2/5 Creating uncertainty heatmap...")
        self.plot_uncertainty_heatmap(y_true, y_pred, epistemic_std, aleatoric_std)
        
        # 3. 预测区间
        print("3/5 Creating prediction intervals...")
        self.plot_prediction_intervals_animation_style(y_true, y_pred, total_std)
        
        # 4. 误差分析
        print("4/5 Creating error analysis...")
        self.plot_error_analysis_comprehensive(y_true, y_pred, epistemic_std, aleatoric_std)
        
        # 5. 如果有多个模型可以创建雷达图
        print("5/5 Report generation complete!")
        
        print(f"\n✅ All visualizations saved to: {self.save_path}")
        print("\nGenerated files:")
        print("  - dashboard_results.png")
        print("  - uncertainty_heatmap.png")
        print("  - prediction_intervals.png")
        print("  - error_analysis.png")


# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 500
    
    # 模拟真实数据
    t = np.linspace(0, 10, n_samples)
    y_true = 100 + 20 * np.sin(2 * np.pi * t / 2) + np.random.normal(0, 5, n_samples)
    
    # 模拟预测
    y_pred = y_true + np.random.normal(0, 3, n_samples)
    
    # 模拟不确定性
    epistemic_std = 2 + np.abs(np.sin(t)) * 2
    aleatoric_std = 3 + np.random.rand(n_samples) * 2
    
    # 模拟指标
    metrics = {
        'MAE': 2.5,
        'RMSE': 3.2,
        'MAPE': 2.8,
        'R2': 0.95,
        'CRPS': 1.8,
        'Coverage_95': 94.5
    }
    
    # 创建可视化器
    visualizer = PremiumVisualizer('./demo_output')
    
    # 生成报告
    visualizer.create_summary_report(
        y_true, y_pred, epistemic_std, aleatoric_std, metrics,
        model_name="Demo Model"
    )
