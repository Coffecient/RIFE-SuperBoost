// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#define RSB_API extern "C" __declspec(dllexport)
// All functions return one on success and zero on failure unless documented.
// Error messages are UTF-8, thread-local, and valid until the next API call.
RSB_API int rsbAbi() noexcept;
RSB_API const char* rsbError() noexcept;
// device: CUDA ordinal; json/capacity: caller-owned UTF-8 device-info output.
RSB_API int rsbDevice(int device, char* json, size_t capacity) noexcept;
// engine: absolute UTF-16 path; width/height: unpadded source dimensions;
// device: CUDA ordinal; handle: receives an opaque, per-node worker.
// Null encoder/synthesis selects full mode; alignment/channels: model profile;
// budget: cache bytes; max_slots: hard cap (2..32); graph: fixed-binding capture.
RSB_API int rsbCreateV2(const wchar_t* full, const wchar_t* encoder, const wchar_t* synthesis,
    int width, int height, int alignment, int channels, int device, uint64_t budget,
    int max_slots, int graph, void** handle) noexcept;
// handle: live worker; timestep: scalar interpolation time in [0,1]; bytes: 2/4;
// left/right/output: three plane pointers; strides: positive byte row strides.
// The caller keeps every plane alive and output writable until this returns.
RSB_API int rsbRun(void* handle, float timestep, int bytes,
    const void* const* left, const int64_t* left_strides,
    const void* const* right, const int64_t* right_strides,
    void* const* output, const int64_t* output_strides) noexcept;
// handle: worker to destroy; the caller serializes destruction against requests.
RSB_API int rsbDestroy(void* handle) noexcept;
// IDs: immutable frame identities within this node; use_ids=0 disables reuse.
// enabled=0 repeats the left frame on CPU, preserving cached RGB/features.
RSB_API int rsbRunFrames(void* handle, uint64_t left_id, uint64_t right_id, int use_ids, int enabled,
    float timestep, int bytes, const void* const* left, const int64_t* left_strides,
    const void* const* right, const int64_t* right_strides,
    void* const* output, const int64_t* output_strides) noexcept;
// Invalidate content IDs after outstanding work finishes; retain allocations.
RSB_API int rsbReset(void* handle) noexcept;
// json/capacity: caller-owned UTF-8 counters and actual cache allocation.
RSB_API int rsbStats(void* handle, char* json, size_t capacity) noexcept;
