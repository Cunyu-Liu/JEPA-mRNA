import torch
from torch import nn


class LocalContextMixer(nn.Module):
    def __init__(self, d_z, kernel_size=3):
        super().__init__()
        if d_z <= 0 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("d_z must be positive and kernel_size must be odd and at least 3")
        self.norm = nn.LayerNorm(d_z)
        self.depthwise = nn.Conv2d(d_z, d_z, kernel_size, padding=kernel_size // 2,
                                   groups=d_z, bias=False)
        self.pointwise = nn.Conv2d(d_z, d_z, 1, bias=False)
        nn.init.zeros_(self.pointwise.weight)

    def forward(self, context):
        if context.ndim != 4 or context.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("context must have shape [batch, length, length, d_z]")
        residual = context
        mixed = self.norm(context).permute(0, 3, 1, 2)
        mixed = self.pointwise(self.depthwise(mixed)).permute(0, 2, 3, 1)
        return residual + mixed
