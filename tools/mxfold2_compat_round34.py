#!/usr/bin/env python3
"""MXfold2 在新版 PyTorch 下的非连续张量兼容补丁。"""
import torch


def patch_paired_layer_reshape() -> None:
    from mxfold2.fold.layers import PairedLayer

    def forward(self, x):
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x_u = torch.triu(x.reshape(B * C, N, N), diagonal=diag).view(B, C, N, N)
        x_l = torch.tril(x.reshape(B * C, N, N), diagonal=-1).view(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).view(B * 2, C, N, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        x_u, x_l = torch.split(x, B, dim=0)
        x_u = torch.triu(x_u.view(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_u.view(B, -1, N, N), diagonal=-1)
        x = x_u + x_l
        x = x.permute(0, 2, 3, 1).reshape(B * N * N, -1)
        x = self.fc(x)
        return x.view(B, N, N, -1)

    PairedLayer.forward = forward
