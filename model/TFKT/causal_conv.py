import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)
