// SPDX-License-Identifier: MIT
#include "transfer.h"
#include <cuda_fp16.h>
#include <cstdint>
namespace {
template<class T, class U> __global__ void upload(const T* source, U* target, int w, int h, int pw, int ph) {
    // source: packed RGB; target: padded FP32 RGB; w/h and pw/ph: dimensions.
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t plane = size_t(pw) * ph;
    if (i >= 3 * plane) return;
    int c = int(i / plane), y = int((i % plane) / pw), x = int(i % pw);
    target[i] = U((x < w && y < h) ? float(source[(size_t(c) * h + y) * w + x]) : 0.0f);
}
template<class T, class U> __global__ void download(const U* source, T* target, int w, int h, int pw, int ph) {
    // source: padded FP32 RGB; target: cropped packed RGB, without clamping.
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t plane = size_t(w) * h;
    if (i >= 3 * plane) return;
    int c = int(i / plane), y = int((i % plane) / w), x = int(i % w);
    target[i] = T(float(source[(size_t(c) * ph + y) * pw + x]));
}
}
cudaError_t rsb_upload(const void* source, void* target, int w, int h, int pw, int ph, int bytes, int storage_bytes, cudaStream_t stream) {
    unsigned blocks = unsigned((size_t(pw) * ph * 3 + 255) / 256);
    if (storage_bytes == 2) {
        if(bytes == 2) upload<<<blocks,256,0,stream>>>(static_cast<const __half*>(source),static_cast<__half*>(target),w,h,pw,ph);
        else upload<<<blocks,256,0,stream>>>(static_cast<const float*>(source),static_cast<__half*>(target),w,h,pw,ph);
    } else {
        if(bytes == 2) upload<<<blocks,256,0,stream>>>(static_cast<const __half*>(source),static_cast<float*>(target),w,h,pw,ph);
        else upload<<<blocks,256,0,stream>>>(static_cast<const float*>(source),static_cast<float*>(target),w,h,pw,ph);
    }
    return cudaGetLastError();
}
cudaError_t rsb_download(const void* source, void* target, int w, int h, int pw, int ph, int bytes, int storage_bytes, cudaStream_t stream) {
    unsigned blocks = unsigned((size_t(w) * h * 3 + 255) / 256);
    if(storage_bytes==2) {
        if(bytes==2)download<<<blocks,256,0,stream>>>(static_cast<const __half*>(source),static_cast<__half*>(target),w,h,pw,ph);
        else download<<<blocks,256,0,stream>>>(static_cast<const __half*>(source),static_cast<float*>(target),w,h,pw,ph);
    } else {
        if(bytes==2)download<<<blocks,256,0,stream>>>(static_cast<const float*>(source),static_cast<__half*>(target),w,h,pw,ph);
        else download<<<blocks,256,0,stream>>>(static_cast<const float*>(source),static_cast<float*>(target),w,h,pw,ph);
    }
    return cudaGetLastError();
}
