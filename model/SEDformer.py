import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional


class _STHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_minus_th, alpha: float = 5.0):
        ctx.save_for_backward(u_minus_th)
        ctx.alpha = alpha
        return (u_minus_th >= 0.).to(u_minus_th.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (u_minus_th,) = ctx.saved_tensors
        alpha = ctx.alpha
        s = torch.sigmoid(alpha * u_minus_th)
        return grad_output * alpha * s * (1 - s), None


def spike_fn(u_minus_th, alpha: float = 5.0):
    return _STHeaviside.apply(u_minus_th, alpha)


class VarStepLIF1D(nn.Module):
    def __init__(self, channels: int, tau_init: float = 2.0,
                 learnable_tau: bool = True, v_reset: float = 0.0,
                 surrogate_alpha: float = 5.0):
        super().__init__()
        self.channels = channels
        self.v_reset = v_reset
        self.surr_alpha = surrogate_alpha
        log_tau0 = torch.log(torch.ones(channels) * float(tau_init))
        self.log_tau = nn.Parameter(log_tau0, requires_grad=learnable_tau)

    def forward(self, I: torch.Tensor, dt: torch.Tensor, v_th: torch.Tensor):
        B, C, D, L = I.shape
        assert C == self.channels
        tau = torch.exp(self.log_tau).view(1, C, 1, 1)
        dt_ = dt.view(B, 1, 1, L)
        alpha_k = torch.exp(-dt_ / (tau + 1e-6))
        kappa_k = 1.0 - alpha_k

        v = torch.zeros(B, C, D, device=I.device, dtype=I.dtype)
        spikes = []
        vth = v_th.view(1, C, 1)

        for k in range(L):
            v = alpha_k[..., k] * v + kappa_k[..., k] * I[..., k]
            s = spike_fn(v - vth, alpha=self.surr_alpha)
            v = torch.where(s.bool(), v - vth, v)
            spikes.append(s.unsqueeze(-1))
        return torch.cat(spikes, dim=-1)


class IMTSConvSpikeEncoder(nn.Module):
    def __init__(self,
                 out_channels_per_var: int = 8,
                 kernel_size: int = 3,
                 spike_temp: float = 5.0,
                 lif_tau_init: float = 2.0,
                 learnable_tau: bool = True,
                 detach_feats: bool = True):
        super().__init__()
        self.c = out_channels_per_var
        self.spike_temp = spike_temp
        self.detach_feats = detach_feats

        self.convbn = nn.Sequential(
            nn.Conv2d(1, self.c, kernel_size=(1, kernel_size),
                      stride=1, padding=(0, kernel_size // 2)),
            nn.BatchNorm2d(self.c),
        )
        self.dt_scale = nn.Parameter(torch.tensor(1.0))
        self.dt_gate  = nn.Sequential(nn.Linear(1, 1, bias=True), nn.Sigmoid())
        self.theta = nn.Parameter(torch.zeros(self.c))
        self.var_lif = VarStepLIF1D(self.c, tau_init=lif_tau_init,
                                    learnable_tau=learnable_tau,
                                    surrogate_alpha=5.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, m: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        if m is None:
            m = torch.ones_like(x)

        xin = (x * m).permute(0, 2, 1).unsqueeze(1)
        feats = self.convbn(xin)

        dt = torch.cat([t[:, :1] * 0, t[:, 1:] - t[:, :-1]], dim=1)
        dt_norm = (dt.clamp_min(0)) / (self.dt_scale.abs() + 1e-6)
        dt_feat = torch.log1p(dt_norm).unsqueeze(-1)
        s = self.dt_gate(dt_feat).transpose(1, 2).unsqueeze(1)

        gated = feats * s
        I = self.spike_temp * (gated - self.theta.view(1, -1, 1, 1))

        spikes = self.var_lif(I, dt, self.theta)

        feats_branch = feats.detach() if self.detach_feats else feats
        mix = torch.cat([spikes, feats_branch], dim=1)
        return mix, feats


class SinCosPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 16384):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        L = x.size(0)
        return x + self.pe[:L].unsqueeze(1).to(x.dtype)


def nonneg_rff(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    BS, h, L, dh = x.shape
    R = W.size(-1)
    proj = torch.einsum('bhld,hdr->bhlr', x, W) + b.view(1, h, 1, R)
    phi  = F.elu(proj, alpha=1.0) + 1.0 + 1e-6
    return phi / math.sqrt(R)


class MembraneIntegrator(nn.Module):
    def __init__(self, dim: int, tau_init: float = 2.0, learnable_tau: bool = True):
        super().__init__()
        init = math.log(math.expm1(max(tau_init - 1.0, 0.0)) + 1e-6)
        self.w_tau = nn.Parameter(torch.full((dim,), float(init)),
                                  requires_grad=learnable_tau)

    def forward(self, x: torch.Tensor):
        L, BS, D = x.shape
        tau = F.softplus(self.w_tau) + 1.0
        beta = 1.0 - 1.0 / tau
        beta = beta.clamp(0.0, 0.9999).view(1, 1, D)
        m = torch.zeros(BS, D, device=x.device, dtype=x.dtype)
        out = []
        for t in range(L):
            xt = x[t]
            m = beta[0, 0] * m + (1.0 - beta[0, 0]) * xt
            out.append(m.unsqueeze(0))
        return torch.cat(out, dim=0)


class SpikingLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, rff_features_num: int = 48,
                 lif_tau: float = 2.0, detach_reset: bool = True, qkv_bias: bool = False):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.R = rff_features_num

        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=True)
        self.Wo = nn.Linear(dim, dim, bias=True)

        self.bn_q = nn.BatchNorm1d(dim)
        self.bn_k = nn.BatchNorm1d(dim)
        self.bn_v = nn.BatchNorm1d(dim)

        self.mem_q = MembraneIntegrator(dim, tau_init=lif_tau, learnable_tau=True)
        self.mem_k = MembraneIntegrator(dim, tau_init=lif_tau, learnable_tau=True)

        scale = 1.0 / math.sqrt(self.dh)
        self.W = nn.Parameter(torch.randn(self.heads, self.dh, self.R) * scale)
        self.b = nn.Parameter(2 * math.pi * torch.rand(self.heads, self.R))

    @staticmethod
    def _bn_timebatch(x: torch.Tensor, bn: nn.BatchNorm1d):
        L, BS, D = x.shape
        return bn(x.reshape(L * BS, D)).view(L, BS, D)

    def _split_heads(self, t: torch.Tensor):
        L, BS, D = t.shape
        return t.view(L, BS, self.heads, self.dh).permute(1, 2, 0, 3).contiguous()

    def _merge_heads(self, t: torch.Tensor):
        BS, h, L, dh = t.shape
        return t.permute(2, 0, 1, 3).contiguous().view(L, BS, h * dh)

    def forward(self, x: torch.Tensor):
        q_in = self._bn_timebatch(self.Wq(x), self.bn_q)
        k_in = self._bn_timebatch(self.Wk(x), self.bn_k)
        v    = self._bn_timebatch(self.Wv(x), self.bn_v)

        Q = self.mem_q(q_in)
        K = self.mem_k(k_in)

        q = self._split_heads(Q)
        k = self._split_heads(K)
        v = self._split_heads(v)

        phi_q = nonneg_rff(q, self.W, self.b)
        phi_k = nonneg_rff(k, self.W, self.b)

        K_sum  = torch.clamp(phi_k.sum(dim=2), min=1e-6)
        KV_sum = torch.einsum('bhlr,bhld->bhrd', phi_k, v)
        out    = torch.einsum('bhlr,bhrd->bhld', phi_q, KV_sum) / \
                 (torch.einsum('bhlr,bhr->bhl', phi_q, K_sum)[..., None] + 1e-6)

        out = self._merge_heads(out)
        return self.Wo(out)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _bn(self, x, bn):
        L, BS, D = x.shape
        return bn(x.reshape(L * BS, D)).view(L, BS, D)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x); x = self._bn(x, self.bn1); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self._bn(x, self.bn2)
        return x


class SpikingTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 lif_tau: float = 2.0, detach_reset: bool = True,
                 rff_features_num: int = 48, dropout: float = 0.0):
        super().__init__()
        self.bn1  = nn.BatchNorm1d(dim)
        self.attn = SpikingLinearAttention(dim, heads=heads,
                                           rff_features_num=rff_features_num,
                                           lif_tau=lif_tau, detach_reset=detach_reset)
        self.bn2  = nn.BatchNorm1d(dim)
        self.ffn  = MLPBlock(dim, hidden=int(dim * mlp_ratio), dropout=dropout)

    def _bn(self, x, bn):
        L, BS, D = x.shape
        return bn(x.reshape(L * BS, D)).view(L, BS, D)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self._bn(x, self.bn1))
        x = x + self.ffn(self._bn(x, self.bn2))
        return x


class SpikingTransformerBackbone(nn.Module):
    def __init__(self, c_in: int, dim: int, depth: int = 1, heads: int = 4,
                 lif_tau: float = 2.0, detach_reset: bool = True,
                 rff_features_num: int = 48, te_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(c_in, dim)
        self.pe = SinCosPE(dim)
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(dim, heads=heads, mlp_ratio=4.0,
                                    lif_tau=lif_tau, detach_reset=detach_reset,
                                    rff_features_num=rff_features_num, dropout=0.0)
            for _ in range(depth)
        ])
        self.te_scale   = nn.Linear(1, 1)
        self.te_per_sin = nn.Linear(1, (te_dim - 1) // 2)
        self.te_per_cos = nn.Linear(1, te_dim - 1 - ((te_dim - 1) // 2))
        self.te_proj    = nn.Linear(te_dim, dim)

    def _TE(self, t):
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        return torch.cat([
            self.te_scale(t),
            torch.sin(self.te_per_sin(t)),
            torch.cos(self.te_per_cos(t))
        ], dim=-1)

    def forward(self, mix: torch.Tensor, t_ds: torch.Tensor):
        B, C2, D, Lp = mix.shape
        x = mix.permute(3, 0, 2, 1).contiguous().view(Lp, B * D, C2)
        x = self.embed(x)

        te = self._TE(t_ds)
        te = self.te_proj(te)
        te = te.view(Lp, B, 1, self.dim).repeat(1, 1, D, 1).view(Lp, B * D, self.dim)
        x  = x + te

        x = self.pe(x)

        for blk in self.blocks:
            x = blk(x)

        x = x.view(Lp, B, D, self.dim).permute(1, 3, 2, 0).contiguous()
        return x


class SEDformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 enc_channels_per_var: int = 8,
                 model_dim: int = 64,
                 te_dim: int = 32,
                 spike_temp: float = 5.0,
                 lif_tau: float = 2.0,
                 detach_reset: bool = True,
                 trans_layers: int = 1,
                 trans_heads: int = 4,
                 rff_features_num: int = 48,
                 pool_stride: int = 8,
                 detach_feats: bool = True):
        super().__init__()
        self.D = input_dim
        self.hid = model_dim
        self.te_dim = te_dim
        self.pool_stride = max(1, int(pool_stride))

        self.encoder = IMTSConvSpikeEncoder(
            out_channels_per_var=enc_channels_per_var,
            kernel_size=3,
            spike_temp=spike_temp,
            lif_tau_init=lif_tau,
            learnable_tau=True,
            detach_feats=detach_feats
        )

        if self.pool_stride > 1:
            self.temporal_pool = nn.MaxPool2d(kernel_size=(1, self.pool_stride),
                                              stride=(1, self.pool_stride),
                                              ceil_mode=False)
        else:
            self.temporal_pool = nn.Identity()

        self.backbone = SpikingTransformerBackbone(
            c_in=enc_channels_per_var * 2,
            dim=model_dim,
            depth=trans_layers,
            heads=trans_heads,
            lif_tau=lif_tau,
            detach_reset=detach_reset,
            rff_features_num=rff_features_num,
            te_dim=te_dim
        )

        self.te_scale   = nn.Linear(1, 1)
        self.te_per_sin = nn.Linear(1, (self.te_dim - 1) // 2)
        self.te_per_cos = nn.Linear(1, self.te_dim - 1 - ((self.te_dim - 1) // 2))
        self.decoder = nn.Sequential(
            nn.Linear(self.hid + self.te_dim, self.hid), nn.ReLU(True),
            nn.Linear(self.hid, self.hid),               nn.ReLU(True),
            nn.Linear(self.hid, 1)
        )

    def _TE(self, t):
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        return torch.cat([
            self.te_scale(t),
            torch.sin(self.te_per_sin(t)),
            torch.cos(self.te_per_cos(t))
        ], dim=-1)

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
        if stride <= 1:
            return mask.permute(0, 2, 1).contiguous()
        B, L, D = mask.shape
        x = mask.permute(0, 2, 1).contiguous().view(B * D, 1, L)
        pooled = F.max_pool1d(x, kernel_size=stride, stride=stride, ceil_mode=False)
        return (pooled > 0).float().view(B, D, -1)

    @staticmethod
    def _downsample_time(t: torch.Tensor, stride: int) -> torch.Tensor:
        if stride <= 1:
            return t
        B, L = t.shape
        idx = torch.arange(stride - 1, L, stride, device=t.device)
        return t[:, idx]

    def forecasting(self, tp_pred, X, tp_true, mask=None):
        B, L, N = X.shape
        mask = torch.ones_like(X) if mask is None else mask

        mix, _ = self.encoder(X, tp_true, mask)

        mix_ds = self.temporal_pool(mix)
        m_ds = self._downsample_mask(mask, self.pool_stride)
        t_ds = self._downsample_time(tp_true, self.pool_stride)

        hist = self.backbone(mix_ds, t_ds)

        hist_bdlh = hist.permute(0, 2, 3, 1).contiguous()
        m_sum = m_ds.sum(dim=2, keepdim=True).clamp_min(1e-6)
        z = (hist_bdlh * m_ds.unsqueeze(-1)).sum(dim=2) / m_sum

        te_p = self._TE(tp_pred)
        te_p = te_p.unsqueeze(1).repeat(1, N, 1, 1)
        Lp = tp_pred.size(1)
        h = z.unsqueeze(2).expand(B, N, Lp, self.hid)
        y = self.decoder(torch.cat([h, te_p], dim=-1)).squeeze(-1)

        return y.unsqueeze(0).permute(0, 1, 3, 2)

    def forward(self, tp_pred, X, tp_true, mask=None):
        return self.forecasting(tp_pred, X, tp_true, mask)

    def forward_irregular(self, batch_list: List[Dict[str, torch.Tensor]]):
        predictions = []
        for sample in batch_list:
            X  = sample["observed_data"].unsqueeze(0)
            tt = sample["observed_tp"].unsqueeze(0)
            m  = sample["observed_mask"].unsqueeze(0)
            tp = sample["tp_to_predict"].unsqueeze(0)
            pred = self.forecasting(tp, X, tt, m)
            predictions.append(pred.squeeze(0))
        return predictions
