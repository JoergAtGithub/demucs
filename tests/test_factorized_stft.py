"""Numerical equivalence of the factorized STFT/ISTFT against the dense matrix version and torch."""
import torch

from demucs.istft import ISTFT_Process
from demucs.stft import STFT_Process


def test_stft_matches_dense_and_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 20000)
    fact = STFT_Process(factorized=True)
    dense = STFT_Process(factorized=False)
    zf, zd = fact(x), dense(x)
    assert zf.shape == zd.shape

    ref = torch.stft(
        x[:, 0], 4096, 1024, window=torch.hann_window(4096), normalized=True,
        center=True, return_complex=True, pad_mode="reflect",
    )
    ref = torch.view_as_real(ref)
    err_fact = (zf - ref).abs().max()
    err_dense = (zd - ref).abs().max()
    assert err_fact < 1e-5
    assert err_fact <= err_dense


def test_istft_matches_dense_and_torch():
    torch.manual_seed(0)
    spec = torch.randn(8, 2049, 40, 2)
    fact = ISTFT_Process(factorized=True)
    dense = ISTFT_Process(factorized=False)
    yf = fact(spec[..., 0], spec[..., 1], length=30000)
    yd = dense(spec[..., 0], spec[..., 1], length=30000)
    assert (yf - yd).abs().max() < 1e-4

    z = torch.view_as_complex(spec.contiguous())
    ref = torch.istft(
        z, 4096, 1024, window=torch.hann_window(4096), normalized=True,
        length=30000, center=True,
    )
    assert (yf - ref).abs().max() < 1e-4
    assert yf.shape == ref.shape
