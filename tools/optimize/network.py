# SPDX-License-Identifier: MIT
# Network parameters and stage semantics: hzwer/Practical-RIFE (MIT).
"""Release-time inference transforms over the strictly loaded M1 reference.

No workspace graph names or community ONNX assets are used. Geometry stays FP32.
Sparse sampling evaluates the four *integer output pixels* of warp before their
half-pixel downsample. Sampling once at the averaged coordinate is not equivalent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Passes:
    """Args:
        sparse: Evaluate only warp pixels consumed by the following Resize.
        lowres: Bypass redundant full-resolution mask/feature intermediates.
        repack: Replace k4/s2/p1 deconvolution by exact coefficient rearrangement.
        dead: Remove the final eight unused state channels before computation.
    """

    sparse: bool = True
    lowres: bool = True
    repack: bool = True
    dead: bool = True


def resize(value: Tensor, factor: float) -> Tensor:
    """Args:
        value: NCHW tensor with aligned spatial dimensions.
        factor: Positive, power-of-two spatial scale.
    """
    if factor == 1:
        return value
    return F.interpolate(value, scale_factor=factor, mode="bilinear", align_corners=False)


def repack_deconv(layer: nn.ConvTranspose2d) -> nn.Sequential:
    """Rearrange coefficients using output coordinate y=2*i+k-1.

    Args:
        layer: Original group-one k4/s2/p1 deconvolution, including its bias.
    """
    assert layer.kernel_size == (4, 4) and layer.stride == (2, 2)
    assert layer.padding == (1, 1) and layer.groups == 1
    ci, co, _, _ = layer.weight.shape
    kernel = layer.weight.new_zeros(co * 4, ci, 3, 3)
    for py in range(2):
        for px in range(2):
            for ky in range(4):
                for kx in range(4):
                    if (py + 1 - ky) % 2 or (px + 1 - kx) % 2:
                        continue
                    y, x = 1 + (py + 1 - ky) // 2, 1 + (px + 1 - kx) // 2
                    kernel[py * 2 + px::4, :, y, x] = layer.weight[:, :, ky, kx].T
    conv = nn.Conv2d(ci, co * 4, 3, padding=1)
    conv.weight = nn.Parameter(kernel, requires_grad=False)
    conv.bias = nn.Parameter(layer.bias.repeat_interleave(4), requires_grad=False)
    return nn.Sequential(conv, nn.PixelShuffle(2))


def sample_pair(source: Tensor, flow: Tensor, scale: int) -> Tensor:
    """Warp two sources, optionally evaluating a sparse output lattice.

    Args:
        source: [2,C,H,W] paired RGB plus encoder features in original order.
        flow: [2,2,H,W] accumulated full-resolution pixel displacements.
        scale: Following half-pixel Resize denominator; one means full warp.

    The normalized base is built by the same linspace as the official warp.
    H/W remain dynamic in ONNX; no resolution-dependent initializer is stored.
    """
    _, _, height, width = flow.shape
    horizontal = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype)
    vertical = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype)
    base = torch.stack(torch.meshgrid(vertical, horizontal, indexing="ij")[::-1], -1)
    displacement = torch.cat((flow[:, :1] / ((width - 1.0) / 2.0),
                              flow[:, 1:] / ((height - 1.0) / 2.0)), 1).permute(0, 2, 3, 1)
    grid = displacement + base.unsqueeze(0)
    if scale == 1:
        return F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # For even scale, the half-pixel output lies halfway between these pixels.
    offset = scale // 2 - 1
    corners = [grid[:, offset + dy::scale, offset + dx::scale]
               for dy in (0, 1) for dx in (0, 1)]
    sparse = torch.stack(corners, -2)
    sampled = F.grid_sample(source, sparse.flatten(2, 3), mode="bilinear",
                            padding_mode="border", align_corners=True)
    return sampled.reshape(2, source.shape[1], height // scale, width // scale, 4).mean(-1)


class Synthesis(nn.Module):
    """Optimized synthesis with explicit shared single-frame encoder inputs."""

    def __init__(self, reference: nn.Module, passes: Passes, mixed: bool = False) -> None:
        """Args:
            reference: M1 network loaded with strict original checkpoint checks.
            passes: Independently selectable transforms for incremental validation.
            mixed: FP16 stage arithmetic with FP32 state, geometry and sampling.
        """
        super().__init__()
        self.scales, self.passes, self.mixed = reference.scales, passes, mixed
        self.blocks = nn.ModuleList(copy.deepcopy(getattr(reference, f"block{i}")) for i in range(5))
        if passes.dead:
            final = self.blocks[-1].lastconv[0]
            final.out_channels = 20
            final.weight = nn.Parameter(final.weight[:, :20].clone(), requires_grad=False)
            final.bias = nn.Parameter(final.bias[:20].clone(), requires_grad=False)
        if passes.repack:
            for block in self.blocks:
                block.lastconv[0] = repack_deconv(block.lastconv[0])
        if mixed:
            # Preserve learned beta and residual accumulation in FP32. Only the
            # neural convolution weights/activations cross half boundaries.
            for module in self.blocks.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    module.half()

    def forward(self, image0: Tensor, image1: Tensor, feature0: Tensor,
                feature1: Tensor, timestep: Tensor) -> Tensor:
        """Args:
            image0: First padded [1,3,H,W] RGB frame.
            image1: Second matching RGB frame.
            feature0: First frame's [1,C,H,W] encoder output, C=4 or 16.
            feature1: Second frame's encoder output in the same channel order.
            timestep: Runtime FP32 [1,1,1,1] scalar in [0,1].
        """
        source = torch.cat((torch.cat((image0, feature0), 1), torch.cat((image1, feature1), 1)), 0)
        flow, mask, state, previous = None, None, None, 1
        for index, (block, scale) in enumerate(zip(self.blocks, self.scales)):
            if flow is None:
                current = [resize(v, 1.0 / scale) for v in (image0, image1, feature0, feature1)]
            else:
                paired_flow = torch.cat((flow[:, :2], flow[:, 2:]), 0)
                warped = sample_pair(source, paired_flow, scale if self.passes.sparse else 1)
                if not self.passes.sparse:
                    warped = resize(warped, 1.0 / scale)
                current = [warped[:1, :3], warped[1:, :3], warped[:1, 3:], warped[1:, 3:]]
            time_map = timestep.expand_as(current[0][:, :1])
            current.append(time_map)
            if flow is not None:
                if self.passes.lowres and scale >= 2 and previous // scale >= 2:
                    current.extend((resize(mask, previous / scale), resize(state, previous / scale)))
                else:
                    current.extend((resize(resize(mask, previous), 1.0 / scale),
                                    resize(resize(state, previous), 1.0 / scale)))
                current.append(resize(flow, 1.0 / scale) / scale)
            stage_input = torch.cat(current, 1)
            if self.mixed:
                stage_input = stage_input.half()
            if self.mixed:
                hidden = block.conv0(stage_input).float()
                for residual in block.convblock:
                    delta = residual.conv(hidden.half()).float()
                    hidden = residual.relu(delta * residual.beta + hidden)
                raw = block.lastconv(hidden.half()).float()
            else:
                raw = block.lastconv(block.convblock(block.conv0(stage_input)))
            delta = resize(raw[:, :4], scale) * scale
            flow = delta if flow is None else flow + delta
            mask, state, previous = raw[:, 4:5], raw[:, 5:], scale
        warped = sample_pair(torch.cat((image0, image1), 0),
                             torch.cat((flow[:, :2], flow[:, 2:]), 0), 1)
        blend = torch.sigmoid(mask)
        return warped[:1] * blend + warped[1:] * (1 - blend)


class Optimized(nn.Module):
    """Full graph and independently exportable encoder/synthesis modules."""

    def __init__(self, reference: nn.Module, passes: Passes = Passes()) -> None:
        """Args:
            reference: Strict, unmodified M1 inference network.
            passes: Selected release-time transforms.
        """
        super().__init__()
        self.encode = copy.deepcopy(reference.encode)
        if passes.repack:
            self.encode.cnn3 = repack_deconv(self.encode.cnn3)
        self.synthesis = Synthesis(reference, passes)

    def forward(self, image0: Tensor, image1: Tensor, timestep: Tensor) -> Tensor:
        """Args:
            image0: First padded FP32 RGB frame.
            image1: Second matching FP32 RGB frame.
            timestep: Runtime scalar, never specialized to the trace sample.
        """
        return self.synthesis(image0, image1, self.encode(image0), self.encode(image1), timestep)


class MixedEncoder(nn.Module):
    """Explicit half neural arithmetic with float storage at the cache boundary."""

    def __init__(self, encoder: nn.Module) -> None:
        """Args:
            encoder: Strictly derived, already optimized single-frame encoder.
        """
        super().__init__()
        self.network = copy.deepcopy(encoder).half()

    def forward(self, image: Tensor) -> Tensor:
        """Args:
            image: Padded FP32 RGB, regardless of the external VS sample size.
        """
        return self.network(image.half()).float()


class MixedOptimized(nn.Module):
    """FP16 convolutions with FP32 residual accumulation/geometry/blend/linear IO."""

    def __init__(self, reference: nn.Module) -> None:
        """Args:
            reference: M1 network; this conversion never mutates the source.
        """
        super().__init__()
        optimized = Optimized(reference)
        self.encode = MixedEncoder(optimized.encode)
        self.synthesis = Synthesis(reference, Passes(), mixed=True)

    def forward(self, image0: Tensor, image1: Tensor, timestep: Tensor) -> Tensor:
        """Args:
            image0: First padded FP32 RGB frame.
            image1: Second matching frame.
            timestep: Runtime FP32 scalar; never specialized to one phase.
        """
        return self.synthesis(image0, image1, self.encode(image0), self.encode(image1), timestep)
