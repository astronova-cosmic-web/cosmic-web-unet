"""Compact U-Net mapping noisy, masked shear [γ₁, γ₂, mask] to convergence κ.

Encoder: ``depth`` levels of (3×3 conv → norm → ReLU) × 2 followed by 2×2 max
pooling, doubling the channels each level. Decoder: bilinear ×2 upsampling, a
1×1 conv to halve the channels, concatenation with the encoder features at the
same scale, then the same double conv. A final 1×1 conv gives a **linear**
output (no activation), since κ is a signed field.

Bilinear upsampling is used instead of transposed convolutions, which tend to
leave checkerboard artifacts: those would add spurious small-scale power, which
is exactly what the power-spectrum loss and metrics measure.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UNet", "NORMS", "PADDING_MODES"]

NORMS = ("batch", "group", "none")
PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def _norm(kind: str, channels: int, groups: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(math.gcd(groups, channels), channels)
    return nn.Identity()


class ConvBlock(nn.Module):
    """(3×3 conv → norm → ReLU) × 2, keeping the spatial size."""

    def __init__(self, in_ch: int, out_ch: int, norm: str, padding_mode: str, groups: int):
        super().__init__()
        bias = norm == "none"  # a following norm layer makes the conv bias redundant
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode, bias=bias),
            _norm(norm, out_ch, groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=padding_mode, bias=bias),
            _norm(norm, out_ch, groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear ×2 upsampling, 1×1 conv to ``out_ch``, concat skip, double conv."""

    def __init__(self, in_ch: int, out_ch: int, norm: str, padding_mode: str, groups: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.block = ConvBlock(2 * out_ch, out_ch, norm, padding_mode, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, self.reduce(x)], dim=1))


class UNet(nn.Module):
    """U-Net for κ reconstruction.

    Args:
        in_channels: input channels; 3 for [γ₁, γ₂, mask].
        out_channels: output channels; 1 for κ.
        base_channels: channels at full resolution; doubled at each level.
        depth: number of 2× downsamplings. Input height and width must be
            divisible by ``2**depth``.
        norm: ``"batch"``, ``"group"`` (better for very small batches) or ``"none"``.
        padding_mode: convolution padding; ``"zeros"`` suits crops and masked edges.
        groups: number of groups for ``norm="group"``.

    Input (B, in_channels, H, W) → output (B, out_channels, H, W). The default
    (base_channels=32, depth=4) has about 7.2 M parameters; at 256² the
    bottleneck is 16 × 16.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32, depth: int = 4,
                 norm: str = "batch", padding_mode: str = "zeros", groups: int = 8):
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        if padding_mode not in PADDING_MODES:
            raise ValueError(f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}")
        if min(in_channels, out_channels, base_channels, depth, groups) < 1:
            raise ValueError("in_channels, out_channels, base_channels, depth and groups must be >= 1")
        self.in_channels, self.out_channels, self.depth = in_channels, out_channels, depth
        widths = [base_channels * 2**i for i in range(depth + 1)]
        args = (norm, padding_mode, groups)

        self.stem = ConvBlock(in_channels, widths[0], *args)
        self.down = nn.ModuleList(ConvBlock(widths[i], widths[i + 1], *args) for i in range(depth))
        self.up = nn.ModuleList(Up(widths[i + 1], widths[i], *args) for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], out_channels, 1)  # linear output: no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected input (B, {self.in_channels}, H, W), got {tuple(x.shape)}")
        factor = 2**self.depth
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f"input size {tuple(x.shape[-2:])} must be divisible by 2**depth = {factor}")
        h = self.stem(x)
        skips = []
        for down in self.down:
            skips.append(h)
            h = down(F.max_pool2d(h, 2))
        for up in self.up:
            h = up(h, skips.pop())
        return self.head(h)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
