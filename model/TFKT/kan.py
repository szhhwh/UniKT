import torch
import torch.nn as nn


class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.kaiming_uniform_(self.cheby_coeffs, mode="fan_in")
        self.register_buffer("arange", torch.arange(degree + 1))

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, orig_shape[-1])

        x = torch.tanh(x)
        x = torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6)
        x = torch.acos(x)
        x = x.unsqueeze(-1) * self.arange
        x = torch.cos(x)

        y = torch.einsum("...id,iod->...o", x, self.cheby_coeffs)
        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], -1)
        return y
