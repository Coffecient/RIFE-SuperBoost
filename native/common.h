// SPDX-License-Identifier: MIT
// Windows/NVIDIA utility patterns reviewed from our independent native-lab.
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsb {
namespace fs = std::filesystem;
inline void require(bool condition, const std::string& message) {
    // condition: required invariant; message: diagnostic on failure.
    if (!condition) throw std::runtime_error(message);
}
inline void check(cudaError_t status) {
    // status: CUDA API result, converted to a stable native error message.
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}
inline std::string utf8(const std::wstring& value) {
    // value: UTF-16 path or argument from Windows.
    if (value.empty()) return {};
    int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), int(value.size()), nullptr, 0, nullptr, nullptr);
    require(size > 0, "Invalid UTF-16 path");
    std::string text(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), text.data(), size, nullptr, nullptr);
    return text;
}
inline std::string quote(const std::string& value) {
    // value: UTF-8 string to encode as a JSON string literal.
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}
inline std::string module_path(const wchar_t* name) {
    // name: Already loaded DLL basename; an empty result means not loaded.
    HMODULE module = GetModuleHandleW(name);
    if (!module) return {};
    std::array<wchar_t, 32768> path{};
    DWORD length = GetModuleFileNameW(module, path.data(), DWORD(path.size()));
    require(length && length < path.size(), "Cannot resolve loaded DLL path");
    return utf8(std::wstring(path.data(), length));
}
inline std::vector<char> read_file(const fs::path& path) {
    // path: Existing engine, ONNX, or timing-cache file.
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(in.good(), "Cannot read " + utf8(path.wstring()));
    auto size = in.tellg();
    require(size > 0, "Empty input file");
    std::vector<char> data(static_cast<size_t>(size));
    in.seekg(0);
    require(bool(in.read(data.data(), size)), "Incomplete input read");
    return data;
}
inline void write_file(const fs::path& path, const void* data, size_t size) {
    // path: Caller-owned temporary output; data/size: complete binary payload.
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.good(), "Cannot write " + utf8(path.wstring()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    out.close();
    require(out.good(), "Incomplete output write");
}
struct Logger final : nvinfer1::ILogger {
    std::atomic_bool failed{false};
    std::mutex mutex;
    std::string last;
    void log(Severity severity, const char* text) noexcept override {
        // severity/text: TensorRT callback data; retain errors without throwing.
        if (severity <= Severity::kERROR) {
            failed.store(true);
            try { std::lock_guard<std::mutex> guard(mutex); last = text; } catch (...) {}
        }
        if (severity <= Severity::kWARNING) {
            OutputDebugStringA(text);
            std::fprintf(stderr, "TensorRT: %s\n", text);
        }
    }
};
struct DeviceScope {
    int previous{};
    explicit DeviceScope(int device) {
        // device: Requested CUDA ordinal; restore the calling thread on exit.
        check(cudaGetDevice(&previous)); check(cudaSetDevice(device));
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};
struct Buffer {
    void* device{};
    void* host{};
    explicit Buffer(size_t bytes = 0, bool pinned = false) {
        // bytes: Allocation size; pinned: also allocate bounded host staging.
        if (bytes) allocate(bytes, pinned);
    }
    void allocate(size_t bytes, bool pinned = false) {
        // bytes/pinned: One deferred allocation after engine IO types are known.
        require(bytes > 0 && !device && !host, "Invalid buffer allocation");
        check(cudaMalloc(&device, bytes));
        if (pinned) {
            auto status = cudaMallocHost(&host, bytes);
            if (status != cudaSuccess) { cudaFree(device); device = nullptr; check(status); }
        }
    }
    ~Buffer() { if (host) cudaFreeHost(host); if (device) cudaFree(device); }
    Buffer(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
inline std::string device_json(int device) {
    // device: CUDA ordinal whose hardware/runtime identity is requested.
    DeviceScope scope(device);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    int driver{}, runtime{};
    check(cudaDriverGetVersion(&driver)); check(cudaRuntimeGetVersion(&runtime));
    std::ostringstream out;
    out << "{\"name\":" << quote(properties.name) << ",\"uuid\":\"";
    for (unsigned char byte : properties.uuid.bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    out << "\",\"driver\":" << std::dec << driver << ",\"cuda\":" << runtime
        << ",\"compute_capability\":[" << properties.major << ',' << properties.minor << "]"
        << ",\"tensorrt\":" << getInferLibVersion() << ",\"tensorrt_build\":" << getInferLibBuildVersion()
        << ",\"tensorrt_path\":" << quote(module_path(L"nvinfer_10.dll")) << '}';
    return out.str();
}
} // namespace rsb
