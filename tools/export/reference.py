# SPDX-License-Identifier: MIT
# Network and warp adapted from hzwer/Practical-RIFE, Copyright (c) 2021 hzwer.
# See models/licenses/practical-rife/LICENSE and THIRD_PARTY.md.
"""Unoptimized inference network with the official parameter and sampling layout.

Only inference is retained. Training helpers and the global warp-grid cache are
removed. The grid is constructed from tensor dimensions for dynamic ONNX export.
No beta folding, channel permutation, warp fusion, or dead-channel pass is used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def warp(image: Tensor, flow: Tensor) -> Tensor:
    """Sample an image using the official pixel-displacement convention.

    Args:
        image: NCHW source tensor with height and width greater than one.
        flow: N2HW displacement in source pixels, ordered as horizontal/vertical.
    """
    batch, _, height, width = flow.shape
    horizontal = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype)
    vertical = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype)
    horizontal = horizontal.view(1, 1, 1, width).expand(batch, 1, height, width)
    vertical = vertical.view(1, 1, height, 1).expand(batch, 1, height, width)
    grid = torch.cat((horizontal, vertical), 1)
    displacement = torch.cat(
        (flow[:, :1] / ((width - 1.0) / 2.0),
         flow[:, 1:2] / ((height - 1.0) / 2.0)), 1
    )
    return F.grid_sample(
        image, (grid + displacement).permute(0, 2, 3, 1),
        mode="bilinear", padding_mode="border", align_corners=True,
    )


def conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Create an official stride-two convolution followed by LeakyReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Head(nn.Module):
    """Official shared single-frame encoder; no cached or reordered features."""

    def __init__(self, encoding_channels: int) -> None:
        """Args:
            encoding_channels: Four for standard/lite, sixteen for large.
        """
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, encoding_channels, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        """Args:
            image: Padded NCHW RGB tensor, without range clamping.
        """
        features = self.relu(self.cnn0(image))
        features = self.relu(self.cnn1(features))
        features = self.relu(self.cnn2(features))
        return self.cnn3(features)


class ResConv(nn.Module):
    """Official residual convolution retaining its explicit learned beta."""

    def __init__(self, channels: int) -> None:
        """Args:
            channels: Number of input and output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        """Args:
            features: Stage-local NCHW features at their original resolution.
        """
        return self.relu(self.conv(features) * self.beta + features)


class IFBlock(nn.Module):
    """Official full-width stage, including all thirteen PixelShuffle channels."""

    def __init__(self, in_channels: int, channels: int) -> None:
        """Args:
            in_channels: Input channels including incoming flow where applicable.
            channels: Internal stage width from the official architecture.
        """
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_channels, channels // 2), conv(channels // 2, channels))
        self.convblock = nn.Sequential(*(ResConv(channels) for _ in range(8)))
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(channels, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, features: Tensor, flow: Tensor | None, scale: int) -> tuple[Tensor, Tensor, Tensor]:
        """Args:
            features: Officially ordered stage input channels at full resolution.
            flow: Accumulated full-resolution flow, or None in the first stage.
            scale: Official stage scale denominator for scale=1 inference.
        """
        features = F.interpolate(features, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            resized_flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
            features = torch.cat((features, resized_flow * 1.0 / scale), 1)
        result = self.lastconv(self.convblock(self.conv0(features)))
        result = F.interpolate(result, scale_factor=scale, mode="bilinear", align_corners=False)
        return result[:, :4] * scale, result[:, 4:5], result[:, 5:]


class ReferenceNet(nn.Module):
    """Five-stage official inference with runtime scalar time and padded RGB IO."""

    def __init__(self, encoding_channels: int, last_channels: int, scales: list[int]) -> None:
        """Args:
            encoding_channels: Shared encoder width from the model profile.
            last_channels: Final stage width; twenty-four for lite, otherwise 32.
            scales: Five official scale denominators, used without substitution.
        """
        super().__init__()
        self.scales = tuple(scales)
        self.block0 = IFBlock(7 + 2 * encoding_channels, 192)
        self.block1 = IFBlock(8 + 4 + 8 + 2 * encoding_channels, 128)
        self.block2 = IFBlock(8 + 4 + 8 + 2 * encoding_channels, 96)
        self.block3 = IFBlock(8 + 4 + 8 + 2 * encoding_channels, 64)
        self.block4 = IFBlock(8 + 4 + 8 + 2 * encoding_channels, last_channels)
        self.encode = Head(encoding_channels)

    def forward(self, image0: Tensor, image1: Tensor, timestep: Tensor) -> Tensor:
        """Args:
            image0: Float32 [1, 3, H, W] RGB, padded right/bottom by the caller.
            image1: Second RGB frame with exactly the same shape and dtype.
            timestep: Float32 [1, 1, 1, 1] interpolation time, normally in [0, 1].
        """
        # Retain the official concatenation/slice strides, including on CUDA.
        images = torch.cat((image0, image1), 1)
        image0, image1 = images[:, :3], images[:, 3:]
        time_map = timestep.repeat(1, 1, image0.shape[2], image0.shape[3])
        feature0, feature1 = self.encode(image0), self.encode(image1)
        warped0, warped1 = image0, image1
        flow, mask, features = None, None, None
        blocks = (self.block0, self.block1, self.block2, self.block3, self.block4)
        for block, scale in zip(blocks, self.scales):
            if flow is None:
                flow, mask, features = block(
                    torch.cat((image0, image1, feature0, feature1, time_map), 1), None, scale
                )
            else:
                warped_feature0 = warp(feature0, flow[:, :2])
                warped_feature1 = warp(feature1, flow[:, 2:4])
                delta, mask, features = block(
                    torch.cat((warped0, warped1, warped_feature0, warped_feature1, time_map, mask, features), 1),
                    flow, scale,
                )
                flow = flow + delta
            warped0, warped1 = warp(image0, flow[:, :2]), warp(image1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)


def pad_pair(image0: Tensor, image1: Tensor, alignment: int = 128) -> tuple[Tensor, Tensor]:
    """Validate reference inputs and apply right/bottom constant-zero padding.

    Args:
        image0: Float32 [1, 3, H, W] RGB with positive spatial dimensions.
        image1: Matching second frame on the same device.
        alignment: Official reference uses 128; candidates require separate checks.
    """
    if alignment not in (64, 128):
        raise ValueError("Reference alignment must be 64 or 128.")
    if image0.ndim != 4 or image0.shape[:2] != (1, 3) or min(image0.shape[-2:]) < 1:
        raise ValueError("Expected [1, 3, H, W] RGB with positive height and width.")
    if image0.shape != image1.shape or image0.dtype != torch.float32 or image1.dtype != torch.float32:
        raise ValueError("Both frames must have the same shape and float32 dtype.")
    if image0.device != image1.device:
        raise ValueError("Both frames must use the same device.")
    height, width = image0.shape[-2:]
    padding = (0, (-width) % alignment, 0, (-height) % alignment)
    return F.pad(image0, padding), F.pad(image1, padding)
