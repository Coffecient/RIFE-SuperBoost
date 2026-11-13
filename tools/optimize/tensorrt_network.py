# SPDX-License-Identifier: MIT
# Network parameters and semantics derive from hzwer/Practical-RIFE (MIT).
"""Prepared TensorRT fusion graphs with eager, independent PyTorch oracles.

Custom nodes replace only sampling, input packing, flow updates and blending.
The official stage scales and FP32 flow/coordinate contract remain explicit.
"""

import copy
import torch
from torch import nn
from torch.onnx.symbolic_helper import parse_args

from network import Optimized, Synthesis, Passes, resize, sample_pair


def tensor_type(value, dtype, channels):
    """Args:
        value: ONNX symbolic image whose dynamic spatial rank is retained.
        dtype: Explicit output scalar type.
        channels: Static logical channel count before physical HWC8 padding.
    """
    return value.type().with_dtype(dtype).with_sizes([1, channels, None, None])


class StagePack(torch.autograd.Function):
    """Downsample and pack the first stage, without full-size concatenations."""

    @staticmethod
    def forward(ctx, a, b, fa, fb, time, scale, features, half):
        """Args:
            ctx: Autograd export context; inference never uses backward.
            a, b: Padded source RGB tensors, in selected storage precision.
            fa, fb: Padded encoder features, in neural precision.
            time: Runtime scalar FP32 interpolation time.
            scale: Official first-stage denominator.
            features: Audited feature width, four or sixteen.
            half: True selects FP16 neural input; otherwise FP32.
        """
        values = [resize(v.float(), 1 / scale) for v in (a, b, fa, fb)]
        result = torch.cat([*values, time.expand_as(values[0][:, :1])], 1)
        return result.half() if half else result.float()

    @staticmethod
    @parse_args("v", "v", "v", "v", "v", "i", "i", "i")
    def symbolic(g, a, b, fa, fb, time, scale, features, half):
        """Args:
            g: ONNX export graph; other arguments follow forward's contract.
        """
        result = g.op("rsb.trt::RsbStagePack", a, b, fa, fb, time, scale_i=scale,
                      features_i=features, half_i=half, plugin_namespace_s="rife_superboost", plugin_version_s="1")
        return result.setType(tensor_type(a, torch.float16 if half else torch.float32, 7 + 2 * features))


class FlowWarpPack(torch.autograd.Function):
    """Fuse flow accumulation, sparse warp, state resizing and HWC8 packing."""

    @staticmethod
    def forward(ctx, a, b, fa, fb, previous, raw, time, scale, previous_scale, first, features, half):
        """Args:
            ctx: Inference export context.
            a, b, fa, fb: Padded RGB and cached features.
            previous: Prior accumulated FP32 flow; scalar placeholder on first update.
            raw: Previous stage's low-resolution FP32 flow/mask/state output.
            time: Runtime FP32 scalar.
            scale, previous_scale: Official stage denominators.
            first: First update omits the placeholder previous flow.
            features: Four or sixteen encoder channels.
            half: Select the convolution input precision.
        """
        flow = resize(raw[:, :4], previous_scale) * previous_scale
        if not first:
            flow = previous + flow
        source = torch.cat((torch.cat((a.float(), fa.float()), 1), torch.cat((b.float(), fb.float()), 1)), 0)
        warped = sample_pair(source, torch.cat((flow[:, :2], flow[:, 2:]), 0), scale)
        values = [warped[:1, :3], warped[1:, :3], warped[:1, 3:], warped[1:, 3:], time.expand_as(warped[:1, :1])]
        for value in (raw[:, 4:5], raw[:, 5:]):
            if scale >= 2 and previous_scale // scale >= 2:
                values.append(resize(value, previous_scale / scale))
            else:
                values.append(resize(resize(value, previous_scale), 1 / scale))
        values.append(resize(flow, 1 / scale) / scale)
        result = torch.cat(values, 1)
        return (result.half() if half else result.float()), flow

    @staticmethod
    @parse_args("v", "v", "v", "v", "v", "v", "v", "i", "i", "i", "i", "i")
    def symbolic(g, a, b, fa, fb, previous, raw, time, scale, previous_scale, first, features, half):
        """Args:
            g: ONNX export graph; remaining arguments follow forward's contract.
        """
        packed, flow = g.op("rsb.trt::RsbFlowWarpPack", a, b, fa, fb, previous, raw, time,
                            scale_i=scale, previous_scale_i=previous_scale, first_i=first,
                            features_i=features, half_i=half, plugin_namespace_s="rife_superboost", plugin_version_s="1", outputs=2)
        packed.setType(tensor_type(a, torch.float16 if half else torch.float32, 20 + 2 * features))
        flow.setType(tensor_type(a, torch.float32, 4))
        return packed, flow


class WarpBlend(torch.autograd.Function):
    """Fuse the final full-resolution flow update, two RGB warps and blending."""

    @staticmethod
    def forward(ctx, a, b, flow, raw, half):
        """Args:
            ctx: Inference export context.
            a, b: Padded source RGB images.
            flow: FP32 accumulated flow before the final scale-one stage.
            raw: Final stage's five live FP32 output channels.
            half: True rounds only the final storage result to FP16.
        """
        flow = flow + raw[:, :4]
        warped = sample_pair(torch.cat((a.float(), b.float()), 0), torch.cat((flow[:, :2], flow[:, 2:]), 0), 1)
        mask = torch.sigmoid(raw[:, 4:5])
        result = warped[:1] * mask + warped[1:] * (1 - mask)
        return result.half() if half else result.float()

    @staticmethod
    @parse_args("v", "v", "v", "v", "i")
    def symbolic(g, a, b, flow, raw, half):
        """Args:
            g: ONNX export graph; remaining arguments follow forward's contract.
        """
        return g.op("rsb.trt::RsbWarpBlend", a, b, flow, raw, half_i=half,
                    plugin_namespace_s="rife_superboost", plugin_version_s="1").setType(tensor_type(a, torch.float16 if half else torch.float32, 3))


class FusedSynthesis(Synthesis):
    """Keep neural arithmetic explicit while exporting the prepared custom nodes."""

    def __init__(self, reference, precision, storage):
        """Args:
            reference: Strictly checked official FP32 network.
            precision: fp16 neural arithmetic with FP32 residuals, or fp32.
            storage: RGB input/output storage, independent of feature precision.
        """
        super().__init__(reference, Passes(), mixed=precision == "fp16")
        self.features = reference.encode.cnn3.out_channels
        self.storage_half = storage == "fp16"

    def forward(self, a, b, fa, fb, time):
        """Args:
            a, b: Padded source RGB images in declared storage precision.
            fa, fb: Cached encoder outputs in declared neural precision.
            time: Runtime FP32 scalar; never specialized at export time.
        """
        flow, raw, previous = a.new_zeros(1, 4, 1, 1, dtype=torch.float32), None, 1
        for index, (block, scale) in enumerate(zip(self.blocks, self.scales)):
            if index == 0:
                packed = StagePack.apply(a, b, fa, fb, time, scale, self.features, int(self.mixed))
            else:
                packed, flow = FlowWarpPack.apply(a, b, fa, fb, flow, raw, time, scale,
                                                  previous, int(index == 1), self.features, int(self.mixed))
            if self.mixed:
                hidden = block.conv0(packed).float()
                for residual in block.convblock:
                    hidden = residual.relu(residual.conv(hidden.half()).float() * residual.beta + hidden)
                raw = block.lastconv(hidden.half()).float()
            else:
                raw = block.lastconv(block.convblock(block.conv0(packed)))
            previous = scale
        return WarpBlend.apply(a, b, flow, raw, int(self.storage_half))


class FusedEncoder(nn.Module):
    """Store features in their actual arithmetic precision, avoiding float expansion."""

    def __init__(self, reference, precision):
        """Args:
            reference: Official FP32 network before any project transforms.
            precision: fp16 or fp32 encoder arithmetic and output storage.
        """
        super().__init__()
        self.network = copy.deepcopy(Optimized(reference).encode)
        self.half_neural = precision == "fp16"
        if self.half_neural:
            self.network.half()

    def forward(self, image):
        """Args:
            image: Source storage tensor; RGBS stays FP32 until the neural boundary.
        """
        return self.network(image.half() if self.half_neural else image.float())
