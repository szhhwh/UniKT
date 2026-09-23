import torch
import torch.nn as nn
import torch.nn.functional as F


class KTBehaviorExtractor(nn.Module):
    def __init__(self, Fs=1, window_size=64, low=1 / 15, high=1 / 5, scale=20.0):
        super().__init__()
        self.N = window_size
        self.Fs = Fs
        self.scale = scale
        self.min_freq = 2 * Fs / window_size
        max_freq = Fs / 2

        low = max(low, self.min_freq)
        high = min(high, max_freq)

        range_freq = max_freq - self.min_freq
        init_low = (low - self.min_freq) / range_freq
        init_high = (high - self.min_freq) / range_freq

        self.raw_low = nn.Parameter(
            torch.logit(torch.tensor([init_low], dtype=torch.float32))
        )
        self.raw_high = nn.Parameter(
            torch.logit(torch.tensor([init_high], dtype=torch.float32))
        )

    def get_freq_bounds(self):
        """Map trainable parameters to ordered frequency bounds."""
        max_freq = self.Fs / 2
        range_freq = max_freq - self.min_freq

        f_low = torch.sigmoid(self.raw_low) * range_freq + self.min_freq
        f_high = torch.sigmoid(self.raw_high) * range_freq + self.min_freq

        return torch.min(f_low, f_high), torch.max(f_low, f_high)

    def forward(self, x):
        """Compute the three band means without materializing window FFTs.

        The mean of an inverse FFT is its zero-frequency coefficient divided
        by the window length. Therefore each returned band mean is the causal
        window mean multiplied by that band's mask at frequency zero.
        """
        f_low, f_high = self.get_freq_bounds()
        mask_low = torch.sigmoid(self.scale * f_low)
        mask_high = torch.sigmoid(-self.scale * f_high)
        mask_mid = (1 - mask_low) * (1 - mask_high)

        padded = F.pad(x.transpose(1, 2), (self.N - 1, 0))
        window_mean = F.avg_pool1d(padded, kernel_size=self.N, stride=1).transpose(1, 2)
        return (
            window_mean * mask_low,
            window_mean * mask_mid,
            window_mean * mask_high,
        )
