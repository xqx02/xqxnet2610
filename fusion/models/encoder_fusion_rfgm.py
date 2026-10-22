import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class EncoderFusionRFGMRefine(nn.Module):
    """
    Encoder-fusion-prior guided decoder refinement.

    current: decoder feature after r3/r2 projection.
    prior:   same-level encoder fusion feature f3/f2.

    The module uses frequency and edge gating ideas, but the
    prior is projected and normalized before interacting with decoder features.
    The residual projection is zero-initialized, so the initial behavior is
    close to identity for warm-start experiments.
    """
    def __init__(
        self,
        current_channels,
        prior_channels,
        alpha_init=0.01,
        alpha_max=0.05,
        top_k=4,
        topk_temperature=0.25,
        amp_scale_init=0.015,
        amp_scale_max=0.05,
        phase_scale_init=0.004,
        phase_scale_max=0.015,
        beta_init=0.08,
        beta_max=0.20,
        edge_mask_floor=0.25,
        edge_mask_sharpness=6.0,
        phase_delta_clip=3.0,
    ):
        super().__init__()
        self.channels = int(current_channels)
        self.top_k = int(top_k)
        self.topk_temperature = float(topk_temperature)
        self.alpha_max = float(alpha_max)
        self.amp_scale_max = float(amp_scale_max)
        self.phase_scale_max = float(phase_scale_max)
        self.beta_max = float(beta_max)
        self.edge_mask_floor = float(edge_mask_floor)
        self.edge_mask_sharpness = float(edge_mask_sharpness)
        self.phase_delta_clip = float(phase_delta_clip)

        self.prior_proj = nn.Sequential(
            nn.Conv2d(prior_channels, current_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(current_channels, current_channels, 1, bias=True),
        )
        self.cur_norm = LayerNorm2d(current_channels)
        self.prior_norm = LayerNorm2d(current_channels)
        self.amp_guide = nn.Sequential(
            nn.Conv2d(1, current_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.phase_delta = nn.Sequential(
            nn.Conv2d(6 * current_channels, current_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(current_channels, current_channels, 1, bias=True),
        )
        self.spatial_delta = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, 3, 1, 1, groups=current_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(current_channels, current_channels, 1, bias=True),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(5 * current_channels, current_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(current_channels, current_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.res_proj = nn.Conv2d(current_channels, current_channels, 1, bias=True)
        nn.init.zeros_(self.res_proj.weight)
        nn.init.zeros_(self.res_proj.bias)

        self.alpha_logit = nn.Parameter(self._logit_from_ratio(alpha_init, self.alpha_max))
        self.amp_scale_logit = nn.Parameter(self._logit_from_ratio(amp_scale_init, self.amp_scale_max))
        self.phase_scale_logit = nn.Parameter(self._logit_from_ratio(phase_scale_init, self.phase_scale_max))
        self.beta_logit = nn.Parameter(self._logit_from_ratio(beta_init, self.beta_max))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.last_stats = {}

    @staticmethod
    def _logit_from_ratio(value, max_value):
        ratio = torch.tensor(float(value) / max(float(max_value), 1e-8))
        ratio = torch.clamp(ratio, 1e-4, 1.0 - 1e-4)
        return torch.logit(ratio)

    def _sobel_grad(self, x):
        b, c, _, _ = x.shape
        weight_x = self.sobel_x.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        weight_y = self.sobel_y.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        gx = F.conv2d(x.float(), weight_x.float(), padding=1, groups=c)
        gy = F.conv2d(x.float(), weight_y.float(), padding=1, groups=c)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        grad = grad / (grad.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return torch.clamp(grad, 0.0, 3.0)

    @staticmethod
    def _radial_frequency_weight(freq_tensor):
        _, _, h, w = freq_tensor.shape
        device = freq_tensor.device
        dtype = freq_tensor.dtype
        orig_w = max(1, (w - 1) * 2)
        fy = torch.fft.fftfreq(h, d=1.0, device=device).abs().to(dtype=dtype)
        fx = torch.fft.rfftfreq(orig_w, d=1.0, device=device).abs().to(dtype=dtype)
        radial = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
        radial = radial / radial.max().clamp_min(1e-6)
        # Favor low/mid frequencies and avoid the DC-only solution.
        weight = torch.exp(-0.5 * ((radial - 0.25) / 0.22).pow(2))
        return weight.view(1, 1, h, w)

    def _topk_prior_amplitude(self, prior_amp, cur_amp):
        b, c, h, w = prior_amp.shape
        freq_weight = self._radial_frequency_weight(prior_amp)
        missing_amp = F.relu(prior_amp - cur_amp)
        scores = (missing_amp * freq_weight).mean(dim=(2, 3))
        top_k = max(1, min(self.top_k, c))
        top_scores, top_idx = torch.topk(scores, k=top_k, dim=1)
        weights = torch.softmax(top_scores / max(self.topk_temperature, 1e-6), dim=1)
        gather_idx = top_idx.view(b, top_k, 1, 1).expand(-1, -1, h, w)
        prior_topk = missing_amp.gather(1, gather_idx)
        prior_weighted = torch.sum(prior_topk * weights.view(b, top_k, 1, 1), dim=1, keepdim=True)
        return prior_weighted, top_idx, weights, scores

    def forward(self, current, prior):
        if prior is None:
            return current
        dtype = current.dtype
        prior = F.interpolate(prior, size=current.shape[-2:], mode="bilinear", align_corners=False)
        prior = self.prior_proj(prior.to(dtype=dtype))

        cur_n = self.cur_norm(current)
        prior_n = self.prior_norm(prior)

        with torch.amp.autocast(device_type=current.device.type, enabled=False):
            cur_f = torch.fft.rfft2(cur_n.float(), norm="ortho")
            prior_f = torch.fft.rfft2(prior_n.float(), norm="ortho")
            cur_amp = torch.abs(cur_f)
            prior_amp = torch.abs(prior_f)
            cur_phase = torch.angle(cur_f)
            prior_phase = torch.angle(prior_f)

            topk_amp, top_idx, top_weights, scores = self._topk_prior_amplitude(prior_amp, cur_amp)
            amp_delta = torch.log1p(prior_amp) - torch.log1p(cur_amp)
            amp_gate = self.amp_guide(topk_amp.float()).float()
            amp_scale = self.amp_scale_max * torch.sigmoid(self.amp_scale_logit).float()
            amp_hat = cur_amp + amp_scale * amp_gate * amp_delta
            amp_hat = torch.clamp(amp_hat, min=0.0)

            phase_diff = prior_phase - cur_phase
            phase_ctx = torch.cat(
                [
                    torch.sin(prior_phase),
                    torch.cos(prior_phase),
                    torch.sin(cur_phase),
                    torch.cos(cur_phase),
                    torch.sin(phase_diff),
                    torch.cos(phase_diff),
                ],
                dim=1,
            )
            phase_delta = self.phase_delta(phase_ctx.float()).float()
            phase_delta = phase_delta - phase_delta.mean(dim=(2, 3), keepdim=True)
            phase_norm = phase_delta.abs().mean(dim=(2, 3), keepdim=True)
            phase_delta = torch.clamp(
                phase_delta / (phase_norm + 1e-6),
                min=-self.phase_delta_clip,
                max=self.phase_delta_clip,
            )
            phase_scale = self.phase_scale_max * torch.sigmoid(self.phase_scale_logit).float()
            phase_hat = cur_phase + phase_scale * phase_delta
            refined_f = torch.complex(amp_hat * torch.cos(phase_hat), amp_hat * torch.sin(phase_hat))
            freq_delta = torch.fft.irfft2(refined_f, s=current.shape[-2:], norm="ortho").to(dtype) - cur_n

        prior_hp = prior_n - F.avg_pool2d(prior_n, kernel_size=3, stride=1, padding=1)
        cur_hp = cur_n - F.avg_pool2d(cur_n, kernel_size=3, stride=1, padding=1)
        spatial_delta = self.spatial_delta(prior_hp - cur_hp)
        cur_grad = self._sobel_grad(cur_n)
        prior_grad = self._sobel_grad(prior_n)
        missing_edge = F.relu(prior_grad.mean(dim=1, keepdim=True) - 0.8 * cur_grad.mean(dim=1, keepdim=True))
        missing_edge = missing_edge / (missing_edge.mean(dim=(2, 3), keepdim=True) + 1e-6)
        missing_edge = torch.clamp(missing_edge, 0.0, 2.0)
        edge_mask = torch.sigmoid(self.edge_mask_sharpness * (missing_edge - missing_edge.mean(dim=(2, 3), keepdim=True)))
        edge_mask_ch = self.edge_mask_floor + (1.0 - self.edge_mask_floor) * edge_mask.expand(-1, self.channels, -1, -1)

        gate_ctx = torch.cat(
            [
                cur_n.abs(),
                prior_n.abs(),
                (prior_n - cur_n).abs(),
                cur_grad.to(dtype),
                prior_grad.to(dtype),
            ],
            dim=1,
        )
        gate = self.detail_gate(gate_ctx)
        beta = self.beta_max * torch.sigmoid(self.beta_logit)
        delta = spatial_delta + beta.to(dtype) * freq_delta
        injected = gate * edge_mask_ch.to(dtype) * self.res_proj(delta)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit)
        out = current + alpha.to(dtype) * injected

        with torch.no_grad():
            self.last_stats = {
                "enc_prior_rfgm_alpha": float(alpha.detach().cpu()),
                "enc_prior_rfgm_amp_scale": float(amp_scale.detach().cpu()),
                "enc_prior_rfgm_phase_scale": float(phase_scale.detach().cpu()),
                "enc_prior_rfgm_beta": float(beta.detach().cpu()),
                "enc_prior_rfgm_amp_gate_mean": float(amp_gate.mean().detach().cpu()),
                "enc_prior_rfgm_gate_mean": float(gate.mean().detach().cpu()),
                "enc_prior_rfgm_edge_mask_mean": float(edge_mask.mean().detach().cpu()),
                "enc_prior_rfgm_missing_edge_mean": float(missing_edge.mean().detach().cpu()),
                "enc_prior_rfgm_freq_delta_abs_mean": float(freq_delta.abs().mean().detach().cpu()),
                "enc_prior_rfgm_spatial_delta_abs_mean": float(spatial_delta.abs().mean().detach().cpu()),
                "enc_prior_rfgm_injected_abs_mean": float((alpha.to(dtype) * injected).abs().mean().detach().cpu()),
                "enc_prior_rfgm_topk_mean": float(top_idx.float().mean().detach().cpu()),
                "enc_prior_rfgm_topk_weight_max": float(top_weights.max(dim=-1).values.mean().detach().cpu()),
                "enc_prior_rfgm_score_mean": float(scores.mean().detach().cpu()),
            }
        return out
