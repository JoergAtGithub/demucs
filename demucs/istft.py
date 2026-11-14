# Copyright (C) 2025 Mixxx Development Team
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Anmol Mishra.
"""
This module implements a custom ISTFT process that is compatible with ONNX export.
It uses PyTorch's matrix multiplication to compute the inverse STFT, avoiding the use of
complex numbers directly, which can be problematic for ONNX export.
"""
import torch
import enum


# Constants set for Demucs
NFFT = 4096  # Number of FFT components for the STFT process
HOP_LENGTH = 1024  # Number of samples between successive frames in the STFT
WINDOW_TYPE = "hann"  # Type of window function used in the STFT
WINDOW_LENGTH = NFFT  # Length of the window function
NORMALIZED = True  # Whether to normalize the window function
MAX_SIGNAL_LENGTH = int(
    44100 * 8
)  # Maximum length of the audio signal WITH padding (8 seconds at 44100 Hz)
MAX_FRAMES = (
    MAX_SIGNAL_LENGTH // HOP_LENGTH + 1
)  # Maximum number of frames for the audio length after STFT processed.
CENTER = True  # Whether to center the input signal before STFT


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


class ISTFT_Process(torch.nn.Module):
    def __init__(
        self,
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        window_type=WINDOW_TYPE,
        window_length=WINDOW_LENGTH,
        normalized=NORMALIZED,
        max_frames=MAX_FRAMES,
        center=CENTER,
    ):
        super(ISTFT_Process, self).__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window_type = window_type
        self.window_length = window_length
        self.normalized = normalized
        self.max_frames = max_frames
        self.center = center
        self.half_n_fft = n_fft // 2  # Precompute once
        self.overlap = n_fft // hop_len  # Number of frames overlapping each sample

        # Get window function and compute window once
        if self.window_length != self.n_fft:
            raise NotImplementedError(
                "The case of window length not equal to n_fft is not implemented "
                f"in {self.__class__.__name__}."
            )
        if self.n_fft % self.hop_len != 0:
            raise NotImplementedError(
                "The case of n_fft not being a multiple of hop_len is not implemented "
                f"in {self.__class__.__name__}."
            )
        window = WindowType(window_type)(self.window_length).float()

        # Check if center is false
        if not self.center:
            raise NotImplementedError(
                "No centering is not supported in this implementation."
            )

        # ISTFT forward pass preparation
        time_steps = torch.arange(n_fft, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(
            0
        )
        omega = 2 * torch.pi * ((time_steps * frequencies) % n_fft) / n_fft

        # Bins between DC and Nyquist appear twice in the full spectrum
        scale = torch.ones(self.half_n_fft + 1)
        scale[1:-1] = 2.0
        scale = scale / n_fft

        # Create inverse basis (window applied) of shape [2 * n_freqs, n_fft]
        cos_inverse = (torch.cos(omega) * scale.unsqueeze(0) * window.unsqueeze(1)).T
        sin_inverse = (-torch.sin(omega) * scale.unsqueeze(0) * window.unsqueeze(1)).T
        inverse_basis = torch.cat([cos_inverse, sin_inverse], dim=0)

        # Calculate window sum for overlap-add
        n = n_fft + hop_len * (max_frames - 1)
        window_sum = torch.zeros(n, dtype=torch.float32)
        win_sq = window**2

        # Calculate overlap-add weights
        for i in range(max_frames):
            sample = i * hop_len
            window_sum[sample: min(n, sample + n_fft)] += win_sq[
                : max(0, min(n_fft, n - sample))
            ]

        # Normalize window if needed
        if normalized:
            inverse_basis = inverse_basis * torch.sqrt(
                torch.tensor([n_fft], dtype=torch.float32)
            )

        # Register buffers
        self.register_buffer("inverse_basis", inverse_basis)
        self.register_buffer(
            "window_sum_inv", 1.0 / torch.clamp(window_sum, min=1e-8)
        )  # Clamp to avoid division by zero

    def forward(self, real, imag, length=None):
        # Prepare input for matrix multiplication: [batch, frames, 2 * n_freqs]
        complex_input = torch.cat((real, imag), dim=1).transpose(-1, -2)

        # Perform matrix multiplication: [batch, frames, n_fft]
        batch_size = complex_input.shape[0]
        inverse_frames = torch.matmul(complex_input, self.inverse_basis)

        # Overlap-add without fold: add `overlap` shifted, non-overlapping reshape passes
        num_frames = inverse_frames.shape[1]
        output_len = (num_frames - 1) * self.hop_len + self.n_fft
        inverse_transform = torch.zeros(
            (batch_size, output_len),
            dtype=inverse_frames.dtype,
            device=inverse_frames.device,
        )
        for offset in range(self.overlap):
            sub_frames = inverse_frames[:, offset:: self.overlap, :]
            num_sub_frames = sub_frames.shape[1]
            if num_sub_frames > 0:
                flat = sub_frames.reshape(batch_size, num_sub_frames * self.n_fft)
                start = offset * self.hop_len
                end = start + num_sub_frames * self.n_fft
                inverse_transform[:, start:end] = inverse_transform[:, start:end] + flat

        # Apply window correction
        start_idx = self.half_n_fft
        end_idx = output_len

        output = (
            inverse_transform[:, start_idx:end_idx]
            * self.window_sum_inv[start_idx:end_idx]
        )

        # If length is specified, trim the output to the desired length
        if length:
            pad_len = torch.clamp(torch.tensor(length) - output.size(-1), min=0)

            # Create a zero pad tensor regardless of need
            pad = torch.zeros(
                output.size(0),
                pad_len,
                dtype=output.dtype,
                device=output.device,
            )

            # Always cat, pad_len will be 0 if not needed
            output = torch.cat([output, pad], dim=-1)

            # Crop in all cases to enforce exact length
            output = output[..., :length]

        return output


demucs_istft = ISTFT_Process(
    n_fft=NFFT,
    hop_len=HOP_LENGTH,
    window_type=WINDOW_TYPE,
    window_length=WINDOW_LENGTH,
    normalized=NORMALIZED,
    max_frames=MAX_FRAMES,
    center=CENTER,
)
