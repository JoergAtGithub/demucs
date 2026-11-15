"""Numerical equivalence of the factorized STFT/ISTFT against torch."""
import torch

from demucs.istft import ISTFT_Process
from demucs.stft import STFT_Process


def test_stft_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 20000)
    z = STFT_Process()(x)

    ref = torch.stft(
        x[:, 0], 4096, 1024, window=torch.hann_window(4096), normalized=True,
        center=True, return_complex=True, pad_mode="reflect",
    )
    ref = torch.view_as_real(ref)
    assert z.shape == ref.shape
    assert (z - ref).abs().max() < 1e-5


def test_istft_matches_torch():
    torch.manual_seed(0)
    spec = torch.randn(8, 2049, 40, 2)
    y = ISTFT_Process()(spec[..., 0], spec[..., 1], length=30000)

    z = torch.view_as_complex(spec.contiguous())
    ref = torch.istft(
        z, 4096, 1024, window=torch.hann_window(4096), normalized=True,
        length=30000, center=True,
    )
    assert (y - ref).abs().max() < 1e-4
    assert y.shape == ref.shape
