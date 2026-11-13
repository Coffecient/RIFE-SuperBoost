// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime_api.h>
// Fields are serialized in the prepared ONNX, never inferred from node names.
struct RsbFusion {
    int features = 4, scale = 1, previous_scale = 1, first = 0, half = 1;
};
// kind: first-stage pack, flow/warp pack, or final warp/blend (0/1/2).
// inputs/outputs: validated device bindings; dimensions: padded source size;
// image_bytes: declared RGB storage; stream: caller-owned execution stream.
cudaError_t rsb_fusion(int kind, RsbFusion fields, const void* const* inputs,
    void* const* outputs, int height, int width, int image_bytes, cudaStream_t stream) noexcept;
// Register process-retained IPluginV3 creators before parsing/deserializing.
bool rsb_register_fusions() noexcept;
