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
import math
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
        self.radix = math.isqrt(n_fft)
        if self.radix * self.radix != n_fft:
            raise NotImplementedError(
                "ISTFT requires n_fft to be a perfect square."
            )

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

        self.register_buffer(
            "window_sum_inv", 1.0 / torch.clamp(window_sum, min=1e-8)
        )  # Clamp to avoid division by zero

        self._init_factorized(window)

    def _init_factorized(self, window):
        """Two-stage inverse DFT, n_fft = R * R, n = R*n1 + n2, k = k1 + R*k2.

        y[n1,n2] = Re sum_k1 W_R^(-k1*n1) * W_N^(-k1*n2) * sum_k2 Z[k1,k2] * W_R^(-k2*n2)
        Equivalent to a dense inverse DFT basis, with ~16x fewer MACs and R x R sized
        constant matrices.
        """
        R = self.radix
        N = self.n_fft
        n_freqs = self.half_n_fft + 1
        k2_count = -(-n_freqs // R)
        self.k2_count = k2_count
        self.n_freqs = n_freqs
        two_pi = 2.0 * math.pi
        r = torch.arange(R, dtype=torch.float64)
        k2 = torch.arange(k2_count, dtype=torch.float64)

        # Interior bins appear twice in the full spectrum; padding bins are zeroed
        scale = torch.zeros(k2_count * R, dtype=torch.float64)
        scale[:n_freqs] = 2.0
        scale[0] = 1.0
        scale[n_freqs - 1] = 1.0
        self.register_buffer("bin_scale", (scale / N).float())

        # Stage A: input [Zr | Zi] over k2, output [Br | Bi] over n2
        angA = two_pi * torch.outer(k2, r) / R
        cA, sA = torch.cos(angA), torch.sin(angA)
        stageA = torch.cat(
            [torch.cat([cA, sA], dim=1), torch.cat([-sA, cA], dim=1)], dim=0
        )
        self.register_buffer("stageA", stageA.float())

        # Twiddle W_N^(-k1*n2), layout [k1, n2]
        angT = two_pi * torch.outer(r, r) / N
        self.register_buffer("twiddle_re", torch.cos(angT).float())
        self.register_buffer("twiddle_im", torch.sin(angT).float())

        # Stage C: input [Cr | Ci] over k1, output real signal over n1
        angC = two_pi * torch.outer(r, r) / R
        self.register_buffer(
            "stageC", torch.cat([torch.cos(angC), -torch.sin(angC)], dim=0).float()
        )

        # Synthesis window (and normalization) in [n1, n2] layout
        scale_w = math.sqrt(N) if self.normalized else 1.0
        self.register_buffer(
            "window_scaled", (window.double() * scale_w).float().reshape(R, R)
        )

    def _inverse_frames_factorized(self, real, imag):
        R = self.radix
        freqs_in = real.shape[1]
        k2n = -(-freqs_in // R)
        pad = k2n * R - freqs_in
        zr = real.transpose(-1, -2)
        zi = imag.transpose(-1, -2)
        if pad:
            zr = torch.nn.functional.pad(zr, (0, pad))
            zi = torch.nn.functional.pad(zi, (0, pad))
        bin_scale = self.bin_scale[: k2n * R]
        zr = zr * bin_scale
        zi = zi * bin_scale
        lead = zr.shape[:-1]
        # bin k = k1 + R*k2 -> [.., k1, k2]
        # one flat batch dim keeps every MatMul at rank 3 (DirectML supports at most rank 4)
        zr = zr.reshape(-1, k2n, R).transpose(-1, -2)
        zi = zi.reshape(-1, k2n, R).transpose(-1, -2)
        if k2n == self.k2_count:
            stage_a = self.stageA
        else:
            kc = self.k2_count
            stage_a = torch.cat([self.stageA[:k2n], self.stageA[kc: kc + k2n]], dim=0)
        b = torch.matmul(torch.cat([zr, zi], dim=-1), stage_a)  # [.., k1, 2*n2]
        br, bi = b[..., :R], b[..., R:]
        cr = br * self.twiddle_re - bi * self.twiddle_im
        ci = br * self.twiddle_im + bi * self.twiddle_re
        c = torch.cat([cr.transpose(-1, -2), ci.transpose(-1, -2)], dim=-1)
        y = torch.matmul(c, self.stageC)  # [.., n2, n1]
        y = y.transpose(-1, -2) * self.window_scaled  # [.., n1, n2]
        return y.reshape(*lead, self.n_fft)

    def forward(self, real, imag, length=None, zero_frames=0):
        """real/imag: [batch, freqs, frames]. Missing high bins are treated as zero and
        `zero_frames` all-zero frames are implicitly present on each side."""
        batch_size = real.shape[0]
        # [batch, frames, n_fft]
        inverse_frames = self._inverse_frames_factorized(real, imag)

        # Overlap-add as one Conv2d: chunk j of the output is the sum over o of
        # chunk o of frame (j - o). The shifts are conv padding (no Pad op).
        num_frames = inverse_frames.shape[1]
        ov, hop = self.overlap, self.hop_len
        x = inverse_frames.reshape(batch_size, num_frames, ov, hop).permute(0, 2, 1, 3)
        weight = torch.zeros(1, ov, ov, 1, dtype=x.dtype, device=x.device)
        for o in range(ov):
            weight[0, o, ov - 1 - o, 0] = 1.0
        core = torch.nn.functional.conv2d(x, weight, padding=(ov - 1, 0))
        core = core.reshape(batch_size, (num_frames + ov - 1) * hop)
        core_len = core.shape[-1]

        # Coordinates of the (virtual) zero-padded frame sequence
        virtual_len = (num_frames + 2 * zero_frames - 1) * hop + self.n_fft
        half = self.half_n_fft
        out_len = virtual_len - half
        target = out_len if not length else min(out_len, int(length))

        shift = zero_frames * hop
        start = half - shift
        if start < 0:
            core = torch.nn.functional.pad(core, (-start, 0))
            start = 0
        if start + target > core.shape[-1]:
            core = torch.nn.functional.pad(core, (0, start + target - core.shape[-1]))
        output = core[:, start: start + target] * self.window_sum_inv[half: half + target]

        # If length is specified, pad up to the desired length
        if length:
            pad_len = max(int(length) - int(output.size(-1)), 0)
            if pad_len > 0:
                output = torch.nn.functional.pad(output, (0, pad_len))

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
