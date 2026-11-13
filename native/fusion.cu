// SPDX-License-Identifier: MIT
// Sampling/packing design reviewed against our independent native-lab kernels.
// Official FP32 geometry is retained; no Half normalized-grid boundary is added.
#include "fusion.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace {
struct Inputs { const void* p[7]; };
struct Sample { int64_t p[4]; float weight[4]; };
__device__ float load(const void* p, int bytes, int64_t i) {
    return bytes == 2 ? __half2float(static_cast<const __half*>(p)[i]) : static_cast<const float*>(p)[i];
}
__device__ Sample position(float x, float y, int h, int w) {
    x = fminf(fmaxf(x, 0.f), float(w - 1)); y = fminf(fmaxf(y, 0.f), float(h - 1));
    int x0 = int(x), y0 = int(y), x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float dx = x - x0, dy = y - y0;
    return {{int64_t(y0) * w + x0, int64_t(y0) * w + x1, int64_t(y1) * w + x0, int64_t(y1) * w + x1},
            {(1.f-dx)*(1.f-dy), dx*(1.f-dy), (1.f-dx)*dy, dx*dy}};
}
__device__ float sample(const void* p, int bytes, int64_t base, const Sample& s) {
    float value = __fmul_rn(load(p, bytes, base + s.p[0]), s.weight[0]);
    value = __fmaf_rn(load(p, bytes, base + s.p[1]), s.weight[1], value);
    value = __fmaf_rn(load(p, bytes, base + s.p[2]), s.weight[2], value);
    return __fmaf_rn(load(p, bytes, base + s.p[3]), s.weight[3], value);
}
__device__ Sample displaced(int x, int y, float fx, float fy, int h, int w) {
    // Match ONNX linspace -> flow normalization -> align_corners unnormalization.
    float gx = __fadd_rn(-1.f, __fmul_rn(float(x), 2.f / float(w - 1)));
    float gy = __fadd_rn(-1.f, __fmul_rn(float(y), 2.f / float(h - 1)));
    gx = __fadd_rn(gx, fx / (float(w - 1) / 2.f));
    gy = __fadd_rn(gy, fy / (float(h - 1) / 2.f));
    return position(__fmul_rn(__fmul_rn(__fadd_rn(gx, 1.f), .5f), float(w-1)),
                    __fmul_rn(__fmul_rn(__fadd_rn(gy, 1.f), .5f), float(h-1)), h, w);
}
__global__ void update_flow(Inputs in, float* output, int h, int w, RsbFusion f) {
    int64_t pixel = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, pixels = int64_t(h) * w;
    if (pixel >= pixels) return;
    int x = int(pixel % w), y = int(pixel / w), rh = h / f.previous_scale, rw = w / f.previous_scale;
    Sample s = position((x + .5f) / f.previous_scale - .5f, (y + .5f) / f.previous_scale - .5f, rh, rw);
    for (int c = 0; c < 4; ++c) {
        float value = __fmul_rn(sample(in.p[5], 4, int64_t(c) * rh * rw, s), float(f.previous_scale));
        if (!f.first) value = __fadd_rn(load(in.p[4], 4, c * pixels + pixel), value);
        output[c * pixels + pixel] = value;
    }
}
__device__ float resized_state(const void* raw, int c, int ox, int oy, int h, int w, RsbFusion f) {
    int rh = h / f.previous_scale, rw = w / f.previous_scale;
    int64_t base = int64_t(c) * rh * rw;
    if (f.scale == 1 || (f.scale >= 2 && f.previous_scale / f.scale >= 2)) {
        return sample(raw, 4, base, position((ox+.5f)*f.scale/f.previous_scale-.5f,
                                            (oy+.5f)*f.scale/f.previous_scale-.5f, rh, rw));
    }
    // Equal/small scale ratios retain the original upsample-then-downsample.
    Sample full = position((ox+.5f)*f.scale-.5f, (oy+.5f)*f.scale-.5f, h, w);
    float result = 0;
    for (int k = 0; k < 4; ++k) {
        int x = int(full.p[k] % w), y = int(full.p[k] / w);
        float value = sample(raw, 4, base, position((x+.5f)/f.previous_scale-.5f, (y+.5f)/f.previous_scale-.5f, rh, rw));
        result = __fmaf_rn(value, full.weight[k], result);
    }
    return result;
}
__global__ void pack_stage(Inputs in, const float* flow, void* output, int h, int w, int image_bytes, int kind, RsbFusion f) {
    __shared__ float tile[56][33];
    __shared__ Sample samples[2][4][32];
    int x = threadIdx.x, lane = threadIdx.y;
    int oh = h / f.scale, ow = w / f.scale;
    int64_t pixels = int64_t(h)*w, out_pixels = int64_t(oh)*ow, base = int64_t(blockIdx.x)*32, p = base+x;
    int channels = (kind == 0 ? 7 : 20) + 2*f.features;
    int physical = f.half ? ((channels+7)/8)*8 : channels;
    int corners = f.scale == 1 ? 1 : 4;
    if (kind && lane < 2 && p < out_pixels) {
        int ox = int(p % ow), oy = int(p / ow), offset = f.scale == 1 ? 0 : f.scale/2-1;
        for (int k = 0; k < corners; ++k) {
            int sx = ox*f.scale+offset+(k%2), sy = oy*f.scale+offset+(k/2);
            int64_t q = int64_t(sy)*w+sx;
            samples[lane][k][x] = displaced(sx, sy, flow[(lane*2)*pixels+q], flow[(lane*2+1)*pixels+q], h, w);
        }
    }
    __syncthreads();
    for (int c = lane; c < physical; c += 8) {
        float value = 0;
        if (p < out_pixels && c < channels) {
            int ox = int(p % ow), oy = int(p / ow);
            if (c < 6 + 2*f.features) {
                bool rgb = c < 6;
                int frame = rgb ? c/3 : (c-6)/f.features;
                int component = rgb ? c%3 : (c-6)%f.features;
                const void* source = in.p[rgb ? frame : 2+frame];
                int bytes = rgb ? image_bytes : f.half ? 2 : 4;
                int64_t start = int64_t(component)*pixels;
                if (kind == 0) value = sample(source, bytes, start, position((ox+.5f)*f.scale-.5f, (oy+.5f)*f.scale-.5f, h, w));
                else {
                    for (int k = 0; k < corners; ++k) value = __fadd_rn(value, sample(source, bytes, start, samples[frame][k][x]));
                    if (corners == 4) value = __fmul_rn(value, .25f);
                }
            } else if (c == 6+2*f.features) value = load(in.p[kind ? 6 : 4], 4, 0);
            else if (c < 16+2*f.features) value = resized_state(in.p[5], c-(7+2*f.features)+4, ox, oy, h, w, f);
            else value = sample(flow, 4, int64_t(c-(16+2*f.features))*pixels,
                                  position((ox+.5f)*f.scale-.5f, (oy+.5f)*f.scale-.5f, h, w)) / f.scale;
        }
        tile[c][x] = value;
        if (!f.half && p < out_pixels) static_cast<float*>(output)[int64_t(c)*out_pixels+p] = value;
    }
    __syncthreads();
    if (f.half) {
        for (int py = lane; py < 32; py += 8) if (base+py < out_pixels)
            for (int c = x; c < physical; c += 32)
                static_cast<__half*>(output)[(base+py)*physical+c] = __float2half_rn(tile[c][py]);
    }
}
__global__ void blend(Inputs in, void* output, int h, int w, int image_bytes, RsbFusion f) {
    int64_t p = int64_t(blockIdx.x)*blockDim.x+threadIdx.x, pixels = int64_t(h)*w;
    if (p >= pixels) return;
    int x = int(p%w), y = int(p/w);
    Sample samples[2];
    for (int frame=0; frame<2; ++frame) {
        float fx=__fadd_rn(load(in.p[2],4,(frame*2)*pixels+p),load(in.p[3],4,(frame*2)*pixels+p));
        float fy=__fadd_rn(load(in.p[2],4,(frame*2+1)*pixels+p),load(in.p[3],4,(frame*2+1)*pixels+p));
        samples[frame]=displaced(x,y,fx,fy,h,w);
    }
    float mask=1.f/(1.f+expf(-load(in.p[3],4,4*pixels+p)));
    for (int c=0;c<3;++c) {
        float a=sample(in.p[0],image_bytes,c*pixels,samples[0]), b=sample(in.p[1],image_bytes,c*pixels,samples[1]);
        float value=__fadd_rn(__fmul_rn(a,mask),__fmul_rn(b,1.f-mask));
        if(f.half) static_cast<__half*>(output)[c*pixels+p]=__float2half_rn(value);
        else static_cast<float*>(output)[c*pixels+p]=value;
    }
}
}
cudaError_t rsb_fusion(int kind, RsbFusion fields, const void* const* inputs, void* const* outputs,
                       int height, int width, int image_bytes, cudaStream_t stream) noexcept {
    if (!inputs || !outputs || height<1 || width<1 || (image_bytes!=2 && image_bytes!=4)) return cudaErrorInvalidValue;
    Inputs in{};
    int count=kind==0?5:kind==1?7:4;
    for(int i=0;i<count;++i) { if(!inputs[i])return cudaErrorInvalidValue; in.p[i]=inputs[i]; }
    if(kind==2) blend<<<unsigned((int64_t(height)*width+255)/256),256,0,stream>>>(in,outputs[0],height,width,image_bytes,fields);
    else {
        float* flow=kind==1?static_cast<float*>(outputs[1]):nullptr;
        if(kind==1) {
            update_flow<<<unsigned((int64_t(height)*width+255)/256),256,0,stream>>>(in,flow,height,width,fields);
            auto status=cudaGetLastError(); if(status!=cudaSuccess)return status;
        }
        int64_t pixels=int64_t(height/fields.scale)*(width/fields.scale);
        pack_stage<<<unsigned((pixels+31)/32),dim3(32,8),0,stream>>>(in,flow,outputs[0],height,width,image_bytes,kind,fields);
    }
    return cudaGetLastError();
}
