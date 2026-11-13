// SPDX-License-Identifier: MIT
#pragma once
#include <cuda_runtime_api.h>
// source/target: packed NCHW buffers; width/height: original dimensions;
// padded_width/padded_height: engine dimensions; bytes: external sample size;
// storage_bytes: prepared model IO size, independently two (half) or four (float).
cudaError_t rsb_upload(const void* source, void* target, int width, int height,
    int padded_width, int padded_height, int bytes, int storage_bytes, cudaStream_t stream);
cudaError_t rsb_download(const void* source, void* target, int width, int height,
    int padded_width, int padded_height, int bytes, int storage_bytes, cudaStream_t stream);
