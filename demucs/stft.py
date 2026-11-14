# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
This module implements a custom STFT process that is compatible with ONNX export.
It uses PyTorch's convolution operations to compute the STFT, avoiding the use of complex
numbers directly, which can be problematic for ONNX export.
"""
import math
import torch
import enum


# Constants set for Demucs
NFFT = 4096  # Number of FFT components for the STFT process
HOP_LENGTH = 1024  # Number of samples between successive frames in the STFT
WINDOW_TYPE = "hann"  # Type of window function used in the STFT
WINDOW_LENGTH = NFFT  # Length of the window function
NORMALIZED = True  # Whether to normalize the window function
CENTER = True  # Whether to center the input signal before STFT
PAD_MODE = "reflect"  # Select reflect or constant padding mode for the STFT process.


# Enum for window types
class WindowType(enum.StrEnum):
    BARTLETT = "bartlett"
    BLACKMAN = "blackman"
    HAMMING = "hamming"
    HANN = "hann"
    KAISER = "kaiser"

    def __call__(self, window_length):
        match self:
            case WindowType.BARTLETT:
                return torch.bartlett_window(window_length)
            case WindowType.BLACKMAN:
                return torch.blackman_window(window_length)
            case WindowType.HAMMING:
                return torch.hamming_window(window_length)
            case WindowType.HANN:
                return torch.hann_window(window_length)
            case WindowType.KAISER:
                return torch.kaiser_window(window_length, periodic=True, beta=12.0)
            case _:
                raise NotImplementedError(
                    f"Window type {self} doesn't yet have a function."
                )


class STFT_Process(torch.nn.Module):
    def __init__(
        self,
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        window_type=WINDOW_TYPE,
        window_length=WINDOW_LENGTH,
        normalized=NORMALIZED,
        center=CENTER,
        pad_mode=PAD_MODE,
    ):
        super(STFT_Process, self).__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window_type = window_type
        self.window_length = window_length
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.half_n_fft = n_fft // 2  # Precompute once

        # Get window function and compute window once
        if self.window_length != self.n_fft:
            raise NotImplementedError(
                "The case of window length not equal to n_fft is not implemented "
                f"in {self.__class__.__name__}."
            )
        window = WindowType(window_type)(self.window_length).float()

        # STFT forward pass preparation
        time_steps = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(0)
        frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(
            1
        )

        # Calculate omega matrix once
        omega = 2 * torch.pi * frequencies * time_steps / n_fft

        # Calculate window function
        cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

        # Normalize window if needed
        if normalized:
            cos_kernel = cos_kernel / torch.sqrt(
                torch.tensor([n_fft], dtype=torch.float32)
            )
            sin_kernel = sin_kernel / torch.sqrt(
                torch.tensor([n_fft], dtype=torch.float32)
            )

        # Register the DFT matrix [n_fft, 2 * n_freqs] as buffer (cos part, then sin part)
        dft_matrix = torch.cat([cos_kernel.squeeze(1).T, sin_kernel.squeeze(1).T], dim=1)
        self.register_buffer("dft_matrix", dft_matrix)
        self._frame_idx_cache = {}

    def _frame_idx(self, padded_length, device):
        key = (padded_length, device)
        if key not in self._frame_idx_cache:
            num_frames = (padded_length - self.n_fft) // self.hop_len + 1
            frame_indices = (
                torch.arange(self.n_fft).unsqueeze(0)
                + torch.arange(num_frames).unsqueeze(1) * self.hop_len
            )
            self._frame_idx_cache[key] = (
                frame_indices.reshape(-1).to(device),
                num_frames,
            )
        return self._frame_idx_cache[key]

    def forward(self, x):
        batch_size, num_channels, _ = x.shape
        if self.center:
            x_padded = torch.nn.functional.pad(
                x, (self.half_n_fft, self.half_n_fft), mode=self.pad_mode
            )
        else:
            x_padded = x

        # Gather + reshape instead of unfold()
        frame_indices, num_frames = self._frame_idx(x_padded.shape[-1], x.device)
        frames = torch.index_select(x_padded, dim=-1, index=frame_indices)
        frames = frames.reshape(batch_size, num_channels, num_frames, self.n_fft)

        # Perform matrix multiplication with the DFT matrix
        spec = torch.matmul(frames, self.dft_matrix)
        n_freqs = self.half_n_fft + 1
        real_part = spec[..., :n_freqs].transpose(-1, -2).squeeze(1)
        image_part = spec[..., n_freqs:].transpose(-1, -2).squeeze(1)

        # return real_part, image_part
        return torch.stack((real_part, image_part), dim=-1)


demucs_stft = STFT_Process(
    n_fft=NFFT,
    hop_len=HOP_LENGTH,
    window_type=WINDOW_TYPE,
    window_length=WINDOW_LENGTH,
    normalized=NORMALIZED,
    center=CENTER,
    pad_mode=PAD_MODE,
)
