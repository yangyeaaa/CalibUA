import os
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from functools import cached_property

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
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[runtime] device = {_DEVICE}")


def fix_random_state(seed: int):
    """Reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═════════════════════════════════════════════
# §1  Cosine-Sigmoid Noise Schedule
# ═════════════════════════════════════════════
@dataclass
class NoiseScheduler:
    """
    Manages the forward (q) and reverse (p) noise process.

    Schedule: sigmoid-mapped linear grid in logit space,
    producing a smooth monotonic beta curve ∈ [β_min, β_max].
    """
    T: int
    device: torch.device = _DEVICE
    beta_min: float = 1e-5
    beta_max: float = 5e-3

    # populated by __post_init__
    beta: torch.Tensor = field(init=False, repr=False)
    alpha: torch.Tensor = field(init=False, repr=False)
    alpha_cum: torch.Tensor = field(init=False, repr=False)
    sqrt_alpha_cum: torch.Tensor = field(init=False, repr=False)
    sqrt_one_m_alpha_cum: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        # build schedule tensors once
        logits = torch.linspace(-6.0, 6.0, self.T, device=self.device)
        raw = torch.sigmoid(logits)
        self.beta = raw * (self.beta_max - self.beta_min) + self.beta_min

        self.alpha = 1.0 - self.beta
        self.alpha_cum = self.alpha.cumprod(dim=0)
        self.sqrt_alpha_cum = self.alpha_cum.sqrt()
        self.sqrt_one_m_alpha_cum = (1.0 - self.alpha_cum).sqrt()

    # ---- forward process ----
    def perturb(self, x0: torch.Tensor, step: int):
        """Return (x_t, ε) via reparameterisation trick."""
        eps = torch.randn_like(x0)
        xt = self.sqrt_alpha_cum[step] * x0 + self.sqrt_one_m_alpha_cum[step] * eps
        return xt, eps

    # ---- single reverse step ----
    def denoise_step(self, net: nn.Module, xt: torch.Tensor, step: int,
                     stochastic: bool = True) -> torch.Tensor:
        """One DDPM reverse step: x_{t} → x_{t-1}."""
        xt = xt.to(self.device)
        coeff = self.beta[step] / self.sqrt_one_m_alpha_cum[step]
        predicted_eps = net(xt, torch.tensor([step], device=self.device))
        mu = (xt - coeff * predicted_eps) / (1.0 - self.beta[step]).sqrt()

        if stochastic and step > 0:
            return mu + self.beta[step].sqrt() * torch.randn_like(xt)
        return mu

    # ---- full reverse chain ----
    def reverse_chain(self, net: nn.Module, xT: torch.Tensor,
                      from_step: Optional[int] = None) -> torch.Tensor:
        t0 = self.T - 1 if from_step is None else from_step
        x = xT
        for s in range(t0, -1, -1):
            x = self.denoise_step(net, x, s, stochastic=(s > 0))
        return x

    # ---- training objective ----
    def score_matching_loss(self, net: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """ε-prediction MSE with antithetic time sampling."""
        B = x0.shape[0]
        half = B // 2
        t_lo = torch.randint(0, self.T, (half,), device=self.device)
        t_all = torch.cat([t_lo, self.T - 1 - t_lo], dim=0).unsqueeze(-1)

        eps = torch.randn_like(x0)
        noisy = self.sqrt_alpha_cum[t_all] * x0 + self.sqrt_one_m_alpha_cum[t_all] * eps
        return F.mse_loss(net(noisy, t_all.squeeze(-1)), eps)


# ═════════════════════════════════════════════
# §2  Network Blocks
# ═════════════════════════════════════════════
class TimestepConditionedMLP(nn.Module):
    """
    Denoising MLP with sinusoidal-style learned step embeddings.

    Architecture: 3 residual-like blocks, each conditioned on t via
    additive embedding after the first linear.
    """
    def __init__(self, dim_in: int, dim_hidden: int, n_steps: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.t_embs = nn.ModuleList()
        for _ in range(3):
            self.blocks.append(nn.Sequential(
                nn.Linear(dim_in if not self.blocks else dim_hidden, dim_hidden),
                nn.SiLU(),
            ))
            self.t_embs.append(nn.Embedding(n_steps, dim_hidden))
            dim_in = dim_hidden           # after first block

        self.proj_out = nn.Linear(dim_hidden, dim_in)  # NOTE: dim_in is now dim_hidden
        # fix: output should match original hidden_dim
        # we'll set it properly in a factory or just accept dim_hidden as the contract

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = x
        for blk, emb in zip(self.blocks, self.t_embs):
            h = blk(h) + emb(t.to(h.device))
        return self.proj_out(h)

    @classmethod
    def build(cls, data_dim: int, hidden: int, n_steps: int):
        """Construct with correct output projection back to data_dim."""
        net = cls.__new__(cls)
        nn.Module.__init__(net)
        net.blocks = nn.ModuleList()
        net.t_embs = nn.ModuleList()

        in_d = data_dim
        for _ in range(3):
            net.blocks.append(nn.Sequential(
                nn.Linear(in_d, hidden),
                nn.SiLU(),
            ))
            net.t_embs.append(nn.Embedding(n_steps, hidden))
            in_d = hidden

        net.proj_out = nn.Linear(hidden, data_dim)
        return net


class VariateAttentionEncoder(nn.Module):
    """
    iTransformer-style encoder: treats each *variate* as a token
    whose feature is the full time-series of that variate.

    Reference: Liu et al., "iTransformer", ICLR 2024.
    """
    def __init__(self, seq_len: int, n_vars: int, d: int = 64,
                 heads: int = 4, depth: int = 2, drop: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(seq_len, d)
        self.pos = nn.Parameter(torch.zeros(1, n_vars, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            dropout=drop, batch_first=True, activation='gelu',
        )
        self.attn_stack = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_vars)  →  tokens: (B, n_vars, d)
        tokens = self.embed(x.permute(0, 2, 1)) + self.pos
        tokens = self.ln(self.attn_stack(tokens))
        tokens = tokens + self.ffn(tokens)
        return tokens[:, 0, :]                       # CLS = first variate


class DualUncertaintyHead(nn.Module):
    """
    Predicts point value **and** two uncertainty scalars
    (aleatoric σ_a, epistemic σ_e) from the latent vector.

    Both σ are bounded via sigmoid ∈ [σ_min, σ_max] and
    further scaled by learnable log-scale parameters.
    """
    LO, HI = 0.01, 3.0

    def __init__(self, d: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1))

        self.fc_main = nn.Linear(d, 1)
        self.fc_res  = nn.Linear(d, 1)

        self.fc_ale = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.fc_epi = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

        self.log_s_ale = nn.Parameter(torch.zeros(1))
        self.log_s_epi = nn.Parameter(torch.full((1,), -1.0))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        rng = self.HI - self.LO

        ale_raw = self.fc_ale(h).squeeze(-1)
        epi_raw = self.fc_epi(h).squeeze(-1)

        return {
            'main_value':   self.fc_main(h).squeeze(-1),
            'residual_mu':  self.fc_res(h).squeeze(-1),
            'aleatoric':    (torch.sigmoid(ale_raw) * rng + self.LO) * self.log_s_ale.exp(),
            'epistemic':    (torch.sigmoid(epi_raw) * rng + self.LO) * self.log_s_epi.exp(),
        }

    def current_scales(self) -> Dict[str, float]:
        return {
            'aleatoric_scale': self.log_s_ale.exp().item(),
            'epistemic_scale': self.log_s_epi.exp().item(),
        }


# ═════════════════════════════════════════════
# §3  Full Model
# ═════════════════════════════════════════════
class DiffLoadUDE(nn.Module):
    """
    Diffusion-enhanced Load forecaster with
    Uncertainty Decomposition and Estimation.
    """
    def __init__(self, seq_len: int, n_vars: int, d: int = 64,
                 heads: int = 4, depth: int = 2, diff_T: int = 5,
                 drop: float = 0.1):
        super().__init__()
        self.d = d
        self.diff_T = diff_T

        self.encoder = VariateAttentionEncoder(seq_len, n_vars, d, heads, depth, drop)
        self.denoiser = TimestepConditionedMLP.build(d, 128, diff_T)
        self.head = DualUncertaintyHead(d)
        self.scheduler: Optional[NoiseScheduler] = None

    def attach_scheduler(self, sched: NoiseScheduler):
        self.scheduler = sched

    def forward(self, x: torch.Tensor, diffuse: bool = True) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        diff_loss = torch.tensor(0.0, device=x.device)

        if diffuse and self.scheduler is not None:
            z_noisy, _ = self.scheduler.perturb(z, self.diff_T - 1)
            diff_loss = self.scheduler.score_matching_loss(self.denoiser, z)
            z = self.scheduler.denoise_step(self.denoiser, z_noisy, self.diff_T - 1)

        out = self.head(z)
        out['diff_loss'] = diff_loss
        out['hidden'] = z
        return out

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, n_mc: int = 100
                                 ) -> Dict[str, torch.Tensor]:
        self.eval()
        buf = {k: [] for k in ('pred', 'ale', 'epi')}

        for _ in range(n_mc):
            o = self.forward(x, diffuse=True)
            buf['pred'].append((o['main_value'] + o['residual_mu']).cpu())
            buf['ale'].append(o['aleatoric'].cpu())
            buf['epi'].append(o['epistemic'].cpu())

        stacked = {k: torch.stack(v) for k, v in buf.items()}
        mu_pred = stacked['pred'].mean(0)
        ale_avg = stacked['ale'].mean(0)
        epi_avg = stacked['epi'].mean(0)
        mc_std  = stacked['pred'].std(0)

        epi_combined = (epi_avg ** 2 + mc_std ** 2).sqrt()
        total = (ale_avg ** 2 + epi_combined ** 2).sqrt()

        return {
            'prediction':     mu_pred,
            'aleatoric':      ale_avg,
            'epistemic_pred': epi_avg,
            'epistemic_mc':   mc_std,
            'epistemic_total': epi_combined,
            'total_std':      total,
            'epistemic_ratio': epi_combined / (total + 1e-8),
            'mc_samples':     stacked['pred'].numpy(),
            'scales':         self.head.current_scales(),
        }


# ═════════════════════════════════════════════
# §4  Loss
# ═════════════════════════════════════════════
class JointUncertaintyLoss(nn.Module):
    """
    Gaussian NLL  +  diffusion score-matching  +  epistemic calibration.

    The calibration term nudges σ_epi ∝ |residual| so that the
    epistemic head *learns* to reflect actual prediction error.
    """
    def __init__(self, w_diff: float = 1.0, w_cal: float = 0.1):
        super().__init__()
        self.w_diff = w_diff
        self.w_cal  = w_cal

    def forward(self, y: torch.Tensor, out: Dict[str, torch.Tensor]):
        pred = out['main_value'] + out['residual_mu']
        res  = y - pred

        var_tot = out['aleatoric'] ** 2 + out['epistemic'] ** 2 + 1e-6
        nll = 0.5 * (var_tot.log() + res ** 2 / var_tot)

        cal = F.mse_loss(out['epistemic'], (res.abs() * 0.5).detach())

        total = nll.mean() + self.w_diff * out['diff_loss'] + self.w_cal * cal
        return total, {
            'total': total.item(),
            'nll':   nll.mean().item(),
            'diff':  out['diff_loss'].item(),
            'cal':   cal.item(),
        }


# ═════════════════════════════════════════════
# §5  Evaluation Toolkit
# ═════════════════════════════════════════════
class Evaluator:
    """Stateless metrics container."""

    @staticmethod
    def mae(y, p):  return float(np.mean(np.abs(y - p)))

    @staticmethod
    def mape(y, p): return float(np.mean(np.abs((y - p) / (np.abs(y) + 1e-8)))) * 100

    @staticmethod
    def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))

    @staticmethod
    def crps_gauss(y, mu, sig): return float(np.mean(ps.crps_gaussian(y, mu, sig)))

    @staticmethod
    def crps_mc(y, samples):
        return float(np.mean([ps.crps_ensemble(y[i], samples[:, i])
                              for i in range(len(y))]))

    @staticmethod
    def pinball(y, mu, sig, quantiles):
        loss = 0.0
        n = len(y)
        for i in range(n):
            qvals = norm.ppf(quantiles, loc=mu[i], scale=sig[i])
            for q, qv in zip(quantiles, qvals):
                e = y[i] - qv
                loss += max((q - 1) * e, q * e)
        return loss / (n * len(quantiles))

    @staticmethod
    def coverage(y, mu, sig):
        info = {}
        for cl in (0.50, 0.75, 0.90, 0.95):
            z = norm.ppf((1 + cl) / 2)
            lo, hi = mu - z * sig, mu + z * sig
            tag = int(cl * 100)
            picp = float(np.mean((y >= lo) & (y <= hi)))
            info[f'PICP_{tag}'] = picp
            info[f'MPIW_{tag}'] = float(np.mean(hi - lo))
            info[f'Gap_{tag}']  = picp - cl
        return info

    @staticmethod
    def ece(y, mu, sig, n_bins=10):
        levels = np.linspace(0.1, 0.95, n_bins)
        obs = np.array([
            np.mean((y >= mu - norm.ppf((1 + c) / 2) * sig) &
                     (y <= mu + norm.ppf((1 + c) / 2) * sig))
            for c in levels
        ])
        return float(np.mean(np.abs(levels - obs))), levels, obs

    @staticmethod
    def decomp_quality(epi, ale, err):
        from scipy.stats import spearmanr
        tot = np.sqrt(epi ** 2 + ale ** 2)
        ae = np.abs(err)
        sr_tot, _ = spearmanr(tot, ae)
        sr_epi, _ = spearmanr(epi, ae)
        return {
            'corr_total_error':     float(np.corrcoef(tot, ae)[0, 1]),
            'corr_epistemic_error': float(np.corrcoef(epi, ae)[0, 1]),
            'corr_aleatoric_error': float(np.corrcoef(ale, ae)[0, 1]),
            'spearman_total_error': float(sr_tot),
            'spearman_epistemic_error': float(sr_epi),
            'epistemic_ratio': float(np.mean(epi / (tot + 1e-8))),
            'mean_epistemic':  float(np.mean(epi)),
            'mean_aleatoric':  float(np.mean(ale)),
            'mean_total':      float(np.mean(tot)),
            'std_epistemic':   float(np.std(epi)),
            'std_aleatoric':   float(np.std(ale)),
        }


# ═════════════════════════════════════════════
# §6  Dataset Wrapper
# ═════════════════════════════════════════════
class SlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):  return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ═════════════════════════════════════════════
# §7  Data Pipeline
# ═════════════════════════════════════════════
def load_and_normalise(base: str = '../GEF_data'):
    """Load .npy splits, fit scaler on train, return normalised arrays + scalers."""
    splits = {}
    for part in ('train', 'val', 'test'):
        splits[f'{part}_x'] = np.load(f'{base}/{part}_data.npy')
        splits[f'{part}_y'] = np.load(f'{base}/{part}_label.npy')

    n_seq, seq_len, n_var = splits['train_x'].shape

    sc_x = StandardScaler().fit(splits['train_x'].reshape(-1, n_var))
    sc_y = StandardScaler().fit(splits['train_y'].reshape(-1, 1))

    for part in ('train', 'val', 'test'):
        raw = splits[f'{part}_x']
        splits[f'{part}_x'] = sc_x.transform(raw.reshape(-1, n_var)).reshape(-1, seq_len, n_var)
        splits[f'{part}_y'] = sc_y.transform(splits[f'{part}_y'].reshape(-1, 1)).ravel()

    return splits, sc_x, sc_y, seq_len, n_var


# ═════════════════════════════════════════════
# §8  Visualisation (20-panel)
# ═════════════════════════════════════════════
def render_dashboard(y_true, res, met, path):
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)
    N = min(300, len(y_true))
    t = np.arange(N)

    pr = res['prediction'][:N]
    ep = res['epistemic'][:N]
    al = res['aleatoric'][:N]
    to = res['total_std'][:N]
    er = y_true[:N] - pr
    ae = np.abs(er)

    # --- row 1 ---
    ax = fig.add_subplot(gs[0, :3])
    ax.plot(t, y_true[:N], 'b-', lw=2, alpha=.8, label='Truth')
    ax.plot(t, pr, 'r--', lw=1.5, alpha=.8, label='Pred')
    ax.fill_between(t, pr - 1.96*to, pr + 1.96*to, alpha=.2, color='gray', label='95 % CI')
    ax.fill_between(t, pr - to, pr + to, alpha=.15, color='blue', label='68 % CI')
    ax.set(xlabel='Step', ylabel='Load'); ax.set_title('Forecast + CI', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax2 = fig.add_subplot(gs[0, 3]); ax2.axis('off')
    txt  = f"MAE  {met['MAE']:.2f}\nMAPE {met['MAPE']:.2f}%\nRMSE {met['RMSE']:.2f}\n"
    txt += f"CRPS(G) {met['CRPS_Gaussian']:.2f}\nCRPS(E) {met['CRPS_Empirical']:.2f}\n"
    txt += f"PB75 {met['Pinball_75']:.2f}\nPB50 {met['Pinball_50']:.2f}\nPB25 {met['Pinball_25']:.2f}\n"
    txt += f"ECE  {met['ECE']:.4f}"
    ax2.text(.05, .95, txt, transform=ax2.transAxes, fontsize=10, va='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', fc='#f0f0f0', ec='gray'))

    # --- row 2 ---
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.stackplot(t, al, ep, labels=['Aleatoric','Epistemic'], colors=['#ff7f0e','#2ca02c'], alpha=.7)
    ax3.set(xlabel='Step', ylabel='σ'); ax3.set_title('Stacked Uncertainty', fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(alpha=.3)

    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(t, ep, 'g-', lw=1.5, label=f'Epi μ={ep.mean():.1f}')
    ax4.plot(t, al, color='orange', lw=1.5, label=f'Ale μ={al.mean():.1f}')
    ax4.plot(t, to, 'b--', lw=1.5, alpha=.6, label=f'Tot μ={to.mean():.1f}')
    ax4.set(xlabel='Step', ylabel='σ'); ax4.set_title('Components', fontweight='bold')
    ax4.legend(fontsize=9); ax4.grid(alpha=.3)

    # --- row 3 ---
    ax5 = fig.add_subplot(gs[2, 0])
    ratio = res['epistemic_ratio'][:N]
    ax5.plot(t, ratio, 'purple', lw=1.5); ax5.fill_between(t, 0, ratio, alpha=.3, color='purple')
    ax5.axhline(.5, ls='--', color='gray', alpha=.5); ax5.set_ylim(0, 1)
    ax5.set(xlabel='Step', ylabel='Ratio'); ax5.set_title('Epi / Total', fontweight='bold')
    ax5.grid(alpha=.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(er, bins=50, density=True, alpha=.7, color='steelblue', edgecolor='white')
    xf = np.linspace(er.min(), er.max(), 100)
    ax6.plot(xf, norm.pdf(xf, er.mean(), er.std()), 'r-', lw=2)
    ax6.axvline(0, ls='--', color='green', lw=2); ax6.set_title('Error Dist', fontweight='bold')
    ax6.grid(alpha=.3)

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(al, bins=30, density=True, alpha=.6, color='orange', label='Ale')
    ax7.hist(ep, bins=30, density=True, alpha=.6, color='green', label='Epi')
    ax7.set_title('Unc Dist', fontweight='bold'); ax7.legend(fontsize=9); ax7.grid(alpha=.3)

    ax8 = fig.add_subplot(gs[2, 3])
    ax8.scatter(y_true[:N], pr, s=15, alpha=.4, edgecolors='none')
    lm = [min(y_true[:N].min(), pr.min()), max(y_true[:N].max(), pr.max())]
    ax8.plot(lm, lm, 'r--', lw=2)
    r2 = np.corrcoef(y_true[:N], pr)[0, 1] ** 2
    ax8.set_title(f'True v Pred R²={r2:.4f}', fontweight='bold'); ax8.grid(alpha=.3)

    # --- row 4 ---
    for idx_col, (unc, clr, lab) in enumerate([(to, 'blue', 'Total'), (ep, 'green', 'Epi')]):
        ax = fig.add_subplot(gs[3, idx_col])
        ax.scatter(unc, ae, s=15, alpha=.4, c=clr, edgecolors='none')
        z_ = np.polyfit(unc, ae, 1); p_ = np.poly1d(z_)
        xl = np.linspace(unc.min(), unc.max(), 100)
        r_ = np.corrcoef(unc, ae)[0, 1]
        ax.plot(xl, p_(xl), 'r-', lw=2, label=f'r={r_:.3f}')
        ax.set_title(f'{lab} σ vs |Err|', fontweight='bold'); ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax11 = fig.add_subplot(gs[3, 2])
    ex, ob = met['calib_expected'], met['calib_observed']
    ax11.plot([0,1],[0,1],'k--',lw=2); ax11.plot(ex, ob, 'bo-', lw=2, ms=8, label=f'ECE={met["ECE"]:.4f}')
    ax11.fill_between(ex, ex, ob, alpha=.3); ax11.set_aspect('equal')
    ax11.set_title('Calibration', fontweight='bold'); ax11.legend(fontsize=9); ax11.grid(alpha=.3)

    ax12 = fig.add_subplot(gs[3, 3])
    lvls = [50,75,90,95]; xp = np.arange(len(lvls)); w = .35
    ax12.bar(xp - w/2, [c/100 for c in lvls], w, label='Expect', color='lightblue')
    ax12.bar(xp + w/2, [met[f'PICP_{c}'] for c in lvls], w, label='Obs', color='steelblue')
    ax12.set_xticks(xp); ax12.set_xticklabels([f'{c}%' for c in lvls])
    ax12.set_title('Coverage', fontweight='bold'); ax12.legend(fontsize=9); ax12.grid(alpha=.3, axis='y')

    # --- row 5 ---
    ax13 = fig.add_subplot(gs[4, :2])
    ax13.plot(t, er, 'g-', alpha=.7, lw=1)
    ax13.fill_between(t, -2*to, 2*to, alpha=.15, color='red', label='±2σ')
    ax13.fill_between(t, -to, to, alpha=.25, color='blue', label='±1σ')
    ax13.axhline(0, color='k'); ax13.set_title('Residuals', fontweight='bold')
    ax13.legend(fontsize=9); ax13.grid(alpha=.3)

    ax14 = fig.add_subplot(gs[4, 2])
    pb = [met['Pinball_75'], met['Pinball_50'], met['Pinball_25']]
    ax14.bar(['75%','50%','25%'], pb, color=['#2ecc71','#3498db','#9b59b6'], edgecolor='k')
    ax14.set_title('Pinball', fontweight='bold'); ax14.grid(alpha=.3, axis='y')

    ax15 = fig.add_subplot(gs[4, 3]); ax15.axis('off')
    s  = f"Epi  μ={met['UQ_mean_epistemic']:.2f}  σ={met['UQ_std_epistemic']:.2f}\n"
    s += f"Ale  μ={met['UQ_mean_aleatoric']:.2f}  σ={met['UQ_std_aleatoric']:.2f}\n"
    s += f"EpiRatio={met['UQ_epistemic_ratio']:.4f}\n"
    s += f"r(Tot,|E|)={met['UQ_corr_total_error']:+.4f}\n"
    s += f"r(Epi,|E|)={met['UQ_corr_epistemic_error']:+.4f}\n"
    s += f"ρ(Tot,|E|)={met['UQ_spearman_total_error']:+.4f}"
    ax15.text(.05, .95, s, transform=ax15.transAxes, fontsize=9, va='top',
              fontfamily='monospace', bbox=dict(boxstyle='round', fc='#f8f8f8', ec='gray'))

    fig.suptitle('DiffLoad-UDE  Uncertainty Decomposition Dashboard',
                 fontsize=16, fontweight='bold', y=.98)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[viz] saved → {path}")


# ═════════════════════════════════════════════
# §9  Train / Test Routines
# ═════════════════════════════════════════════
def run_training(model, tr_dl, va_dl, sched, save_dir, *,
                 epochs=300, patience=15, lr=5e-4,
                 w_diff=1.0, w_cal=0.1):
    model.attach_scheduler(sched)
    model.to(_DEVICE)

    criterion = JointUncertaintyLoss(w_diff, w_cal)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=.5, min_lr=1e-6)

    best_rmse, wait = float('inf'), 0
    hist = {k: [] for k in ('loss', 'val_rmse', 'nll', 'diff', 'cal')}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_m = {k: [] for k in ('tot', 'nll', 'diff', 'cal')}

        for xb, yb in tr_dl:
            xb, yb = xb.to(_DEVICE), yb.squeeze().to(_DEVICE)
            loss, ld = criterion(yb, model(xb, diffuse=True))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in zip(('tot','nll','diff','cal'),
                            (ld['total'], ld['nll'], ld['diff'], ld['cal'])):
                ep_m[k].append(v)

        model.eval()
        vr = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(_DEVICE), yb.squeeze().to(_DEVICE)
                o = model(xb, diffuse=False)
                vr.append(((yb - o['main_value'] - o['residual_mu'])**2).mean().sqrt().item())

        avg_l = np.mean(ep_m['tot']); avg_v = np.mean(vr)
        hist['loss'].append(avg_l); hist['val_rmse'].append(avg_v)
        hist['nll'].append(np.mean(ep_m['nll']))
        hist['diff'].append(np.mean(ep_m['diff']))
        hist['cal'].append(np.mean(ep_m['cal']))
        lr_sched.step(avg_v)

        sc = model.head.current_scales()
        if avg_v < best_rmse:
            best_rmse, wait = avg_v, 0
            torch.save({'epoch': ep, 'state': model.state_dict(),
                        'val_rmse': avg_v, 'scales': sc},
                       f"{save_dir}/best.pt")
            print(f"  ep {ep:3d}  L={avg_l:.4f}  V={avg_v:.4f}  "
                  f"s_a={sc['aleatoric_scale']:.3f} s_e={sc['epistemic_scale']:.3f} ✓")
        else:
            wait += 1
            if ep % 10 == 0:
                print(f"  ep {ep:3d}  L={avg_l:.4f}  V={avg_v:.4f}  wait={wait}/{patience}")
            if wait >= patience:
                print(f"  → early stop @ {ep}"); break

    print(f"  training {time.time()-t0:.1f}s   best_val_rmse={best_rmse:.4f}")

    # history plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, k, c, tit in zip(axes.flat,
            ['loss','val_rmse','nll','diff'],
            ['tab:blue','tab:orange','tab:blue','tab:red'],
            ['Total Loss','Val RMSE','NLL','Diffusion Loss']):
        ax.plot(hist[k], color=c); ax.set_title(tit); ax.grid(alpha=.3)
    if len(hist['cal']):
        axes[1, 0].plot(hist['cal'], color='tab:green', label='Cal')
        axes[1, 0].legend()
    fig.tight_layout(); fig.savefig(f"{save_dir}/history.png", dpi=150); plt.close()
    return hist


def run_evaluation(model, test_x, test_y, sc_x, sc_y, sched, save_dir, n_mc=100):
    model.attach_scheduler(sched); model.to(_DEVICE); model.eval()

    xt = torch.from_numpy(test_x).float().to(_DEVICE)
    t0 = time.time()
    raw = model.predict_with_uncertainty(xt, n_mc)
    print(f"  inference {time.time()-t0:.1f}s   scales={raw['scales']}")

    s, m = sc_y.scale_[0], sc_y.mean_[0]
    pred   = raw['prediction'].numpy() * s + m
    truth  = test_y * s + m
    epi    = raw['epistemic_total'].numpy() * s
    ale    = raw['aleatoric'].numpy() * s
    total  = raw['total_std'].numpy() * s
    mc_o   = raw['mc_samples'] * s + m
    eratio = raw['epistemic_ratio'].numpy()

    E = Evaluator
    met = {}
    met['MAE']  = E.mae(truth, pred)
    met['MAPE'] = E.mape(truth, pred)
    met['RMSE'] = E.rmse(truth, pred)
    met['CRPS_Gaussian']  = E.crps_gauss(truth, pred, total)
    met['CRPS_Empirical'] = E.crps_mc(truth, mc_o)

    met['Pinball_75'] = E.pinball(truth, pred, total, [0.125, 0.875]) / 0.25
    met['Pinball_50'] = E.pinball(truth, pred, total, [0.25,  0.75])  / 0.5
    met['Pinball_25'] = E.pinball(truth, pred, total, [0.375, 0.625]) / 0.75

    met.update(E.coverage(truth, pred, total))

    ece_val, exp_arr, obs_arr = E.ece(truth, pred, total)
    met['ECE'] = ece_val; met['calib_expected'] = exp_arr; met['calib_observed'] = obs_arr

    err = truth - pred
    uq = E.decomp_quality(epi, ale, err)
    for k, v in uq.items():
        met[f'UQ_{k}'] = v

    # ── console report ──
    sep = '─' * 70
    print(f"\n{'═'*70}\n{'RESULTS':^70}\n{'═'*70}")
    for section, rows in [
        ('POINT', [('MAE', met['MAE']), ('MAPE %', met['MAPE']), ('RMSE', met['RMSE'])]),
        ('CRPS',  [('Gaussian', met['CRPS_Gaussian']), ('Empirical', met['CRPS_Empirical'])]),
        ('PINBALL',[('75% PI', met['Pinball_75']),('50% PI', met['Pinball_50']),('25% PI', met['Pinball_25'])]),
    ]:
        print(f"\n{sep}\n{section:^70}\n{sep}")
        for lbl, val in rows:
            print(f"  {lbl:<35} {val:>15.4f}")

    print(f"\n{sep}\n{'COVERAGE':^70}\n{sep}")
    for c in (50, 75, 90, 95):
        picp = met[f'PICP_{c}']; gap = met[f'Gap_{c}']; mpiw = met[f'MPIW_{c}']
        flag = '✓' if abs(gap) < .05 else ('↑' if gap > 0 else '↓')
        print(f"  {c}%  expect={c/100:.2f}  obs={picp:.4f}  gap={gap:+.4f}  width={mpiw:.2f}  {flag}")

    print(f"\n{sep}\n{'ECE = '+str(round(met['ECE'],4)):^70}\n{sep}")
    print(f"\n{sep}\n{'DECOMPOSITION':^70}\n{sep}")
    print(f"  Epi  μ={met['UQ_mean_epistemic']:.2f}  σ={met['UQ_std_epistemic']:.2f}  "
          f"ratio={met['UQ_epistemic_ratio']:.4f}")
    print(f"  Ale  μ={met['UQ_mean_aleatoric']:.2f}  σ={met['UQ_std_aleatoric']:.2f}")
    print(f"  r(Tot,|E|)={met['UQ_corr_total_error']:+.4f}  "
          f"r(Epi,|E|)={met['UQ_corr_epistemic_error']:+.4f}  "
          f"ρ(Tot)={met['UQ_spearman_total_error']:+.4f}")
    print(f"{'═'*70}\n")

    # save
    np.savez(f"{save_dir}/pred.npz", pred=pred, truth=truth,
             epi=epi, ale=ale, total=total)
    scalar_met = {k: v for k, v in met.items() if not isinstance(v, np.ndarray)}
    pd.DataFrame([scalar_met]).to_csv(f"{save_dir}/metrics.csv", index=False)

    res_out = dict(prediction=pred, epistemic=epi, aleatoric=ale,
                   total_std=total, epistemic_ratio=eratio, mc_samples=mc_o)
    render_dashboard(truth, res_out, met, f"{save_dir}/dashboard.png")
    return met, res_out


# ═════════════════════════════════════════════
# §10  Entry Point
# ═════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='DiffLoad-UDE')
    ap.add_argument('--mode',        default='both', choices=['train','test','both'])
    ap.add_argument('--diff_steps',  type=int, default=5)
    ap.add_argument('--runs',        type=int, default=1)
    ap.add_argument('--mc_samples',  type=int, default=100)
    ap.add_argument('--epochs',      type=int, default=300)
    ap.add_argument('--batch_size',  type=int, default=256)
    ap.add_argument('--lr',          type=float, default=5e-4)
    ap.add_argument('--d_model',     type=int, default=64)
    ap.add_argument('--n_heads',     type=int, default=4)
    ap.add_argument('--n_layers',    type=int, default=2)
    ap.add_argument('--w_diff',      type=float, default=1.0)
    ap.add_argument('--w_cal',       type=float, default=0.1)
    cfg = ap.parse_args()

    print(f"\n{'='*60}\n DiffLoad-UDE  d={cfg.d_model} h={cfg.n_heads} L={cfg.n_layers}"
          f"  w_d={cfg.w_diff} w_c={cfg.w_cal}\n{'='*60}")

    sched = NoiseScheduler(T=cfg.diff_steps, device=_DEVICE)
    root = './diffload_ude'

    data, sc_x, sc_y, seq_len, n_var = load_and_normalise()

    for rid in range(1, cfg.runs + 1):
        print(f"\n{'#'*60}\n# Run {rid}/{cfg.runs}\n{'#'*60}")
        fix_random_state(rid)

        os.makedirs(f"{root}/models/run_{rid}", exist_ok=True)
        os.makedirs(f"{root}/results/run_{rid}", exist_ok=True)
        mdir = f"{root}/models/run_{rid}"
        rdir = f"{root}/results/run_{rid}"

        model = DiffLoadUDE(seq_len, n_var, cfg.d_model, cfg.n_heads,
                            cfg.n_layers, cfg.diff_steps, drop=0.1)
        print(f"  params = {sum(p.numel() for p in model.parameters()):,}")

        if cfg.mode in ('train', 'both'):
            tr_dl = DataLoader(SlidingWindowDataset(data['train_x'], data['train_y']),
                               batch_size=cfg.batch_size, shuffle=True)
            va_dl = DataLoader(SlidingWindowDataset(data['val_x'], data['val_y']),
                               batch_size=cfg.batch_size, shuffle=False)
            run_training(model, tr_dl, va_dl, sched, mdir,
                         epochs=cfg.epochs, lr=cfg.lr,
                         w_diff=cfg.w_diff, w_cal=cfg.w_cal)

        if cfg.mode in ('test', 'both'):
            ck = torch.load(f"{mdir}/best.pt", map_location=_DEVICE)
            model.load_state_dict(ck['state'])
            print(f"  loaded epoch {ck['epoch']}")
            run_evaluation(model, data['test_x'], data['test_y'],
                           sc_x, sc_y, sched, rdir, n_mc=cfg.mc_samples)

    print(f"\n{'='*60}\n Done.\n{'='*60}")


if __name__ == '__main__':
    main()
