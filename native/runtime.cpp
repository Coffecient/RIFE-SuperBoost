// SPDX-License-Identifier: MIT
#include "api.h"
#include "common.h"
#include "transfer.h"
#include "fusion.h"
#include <functional>
#include <map>
using namespace rsb;
namespace {
thread_local std::string error;
Logger& logger() { static auto* value = new Logger; return *value; }
template<class Action> int boundary(Action action) noexcept {
    // action: exception-throwing implementation; never let exceptions cross C ABI.
    try { error.clear(); action(); return 1; }
    catch (const std::exception& caught) { error = caught.what(); }
    catch (...) { error = "Unknown native failure"; }
    return 0;
}
struct Network {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    Network(nvinfer1::IRuntime& runtime, const fs::path& path, int w, int h, int channels, const std::string& role) {
        // runtime/path: owning runtime and prepared engine; w/h: padded dimensions;
        // channels: encoder width; role: full, encoder or synthesis contract.
        auto data = read_file(path);
        engine.reset(runtime.deserializeCudaEngine(data.data(), data.size()));
        require(bool(engine), "Engine deserialization failed; rebuild this cache entry");
        require(engine->getNbIOTensors() == (role == "full" ? 4 : role == "encoder" ? 2 : 6), "Prepared engine IO count mismatch");
        context.reset(engine->createExecutionContext());
        require(bool(context), "TensorRT context allocation failed");
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            require((engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT || engine->getTensorDataType(name) == nvinfer1::DataType::kHALF) &&
                    engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR &&
                    engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                    !engine->isShapeInferenceIO(name), "Engine requires declared floating linear device IO");
            std::string key(name);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                bool feature = key == "feature0" || key == "feature1";
                require((role == "encoder" && key == "image") || (role != "encoder" &&
                        (key == "image0" || key == "image1" || key == "timestep")) ||
                        (role == "synthesis" && feature), "Unexpected prepared engine input");
                auto dims = key == "timestep" ? nvinfer1::Dims4(1, 1, 1, 1) : nvinfer1::Dims4(1, feature ? channels : 3, h, w);
                require(context->setInputShape(name, dims), "Engine profile does not support padded dimensions");
            } else require(key == (role == "encoder" ? "feature" : "output"), "Unexpected engine output");
        }
        auto dims = context->getTensorShape(role == "encoder" ? "feature" : "output");
        require(dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == (role == "encoder" ? channels : 3) &&
                dims.d[2] == h && dims.d[3] == w, "Engine output shape mismatch");
    }
    void bind(const char* name, void* address) {
        // name/address: device IO binding retained through stream completion.
        require(context->setTensorAddress(name, address), "Tensor address binding failed");
    }
    int bytes(const char* name) const {
        // name: Declared engine tensor whose checked floating storage is queried.
        return engine->getTensorDataType(name) == nvinfer1::DataType::kHALF ? 2 : 4;
    }
    void enqueue(cudaStream_t stream) {
        // stream: this node's serialized, nonblocking CUDA stream.
        require(context->enqueueV3(stream), "TensorRT enqueue failed");
    }
};
struct Slot {
    Buffer image, feature;
    uint64_t id = 0, stamp = 0;
    bool valid = false;
    int sample_bytes = 0;
    Slot(size_t rgb_bytes, size_t feature_bytes) : image(rgb_bytes), feature(feature_bytes) {}
};
struct CapturedGraph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    uint64_t stamp = 0;
    ~CapturedGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
struct Live {
    int device, width, height, pw, ph, channels, image_bytes = 4, feature_bytes = 4;
    bool failed = false, graph_enabled;
    uint64_t budget, slot_bytes, clock = 0, hits = 0, misses = 0, evictions = 0, resets = 0, requests = 0, skipped = 0;
    float last_gpu_ms = 0;
    std::mutex mutex;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<Network> synthesis, encoder;
    Buffer left, right, output, time, raw_left, raw_right, raw_output;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::unique_ptr<CapturedGraph>> encoder_graphs;
    std::map<int, std::unique_ptr<CapturedGraph>> synthesis_graphs;
    cudaEvent_t start{}, stop{};
    Stream stream;
    Live(const fs::path& full, const wchar_t* enc, const wchar_t* syn, int w, int h, int alignment,
         int c, int gpu, uint64_t cache_budget, int max_slots, bool use_graph)
        : device(gpu), width(w), height(h), pw((w + alignment - 1) / alignment * alignment),
          ph((h + alignment - 1) / alignment * alignment), channels(c), graph_enabled(use_graph),
          budget(cache_budget), slot_bytes(0), time(4, true),
          raw_left(size_t(w) * h * 3 * 4, true), raw_right(size_t(w) * h * 3 * 4, true), raw_output(size_t(w) * h * 3 * 4, true) {
        // full/enc/syn: compiled engines; null enc/syn select full mode.
        // cache_budget/max_slots: hard RGB/feature cache limits, excluding workspace.
        require(rsb_register_fusions(), "Cannot register prepared fusion operators");
        runtime.reset(nvinfer1::createInferRuntime(logger()));
        require(bool(runtime), "TensorRT runtime creation failed");
        if (enc && syn) {
            encoder = std::make_unique<Network>(*runtime, fs::path(enc), pw, ph, c, "encoder");
            synthesis = std::make_unique<Network>(*runtime, fs::path(syn), pw, ph, c, "synthesis");
            image_bytes = synthesis->bytes("image0"); feature_bytes = encoder->bytes("feature");
            require(encoder->bytes("image") == image_bytes && synthesis->bytes("image1") == image_bytes &&
                    synthesis->bytes("output") == image_bytes && synthesis->bytes("timestep") == 4 &&
                    synthesis->bytes("feature0") == feature_bytes && synthesis->bytes("feature1") == feature_bytes,
                    "Encoder/synthesis prepared storage mismatch");
            slot_bytes = size_t(pw) * ph * (3 * image_bytes + c * feature_bytes);
            auto count = std::min(uint64_t(max_slots), budget / slot_bytes);
            require(count >= 2, "GPU frame cache budget cannot hold two RGB/feature slots");
            for (uint64_t i = 0; i < count; ++i) slots.push_back(std::make_unique<Slot>(size_t(pw) * ph * 3 * image_bytes, size_t(pw) * ph * c * feature_bytes));
            encoder_graphs.resize(size_t(count));
        } else {
            synthesis = std::make_unique<Network>(*runtime, full, pw, ph, 0, "full");
            image_bytes = synthesis->bytes("image0");
            require(synthesis->bytes("image1") == image_bytes && synthesis->bytes("output") == image_bytes &&
                    synthesis->bytes("timestep") == 4, "Full engine prepared storage mismatch");
            left.allocate(size_t(pw) * ph * 3 * image_bytes); right.allocate(size_t(pw) * ph * 3 * image_bytes);
            synthesis->bind("image0", left.device); synthesis->bind("image1", right.device);
        }
        output.allocate(size_t(pw) * ph * 3 * image_bytes);
        synthesis->bind("timestep", time.device); synthesis->bind("output", output.device);
    }
    ~Live() {
        cudaStreamSynchronize(stream.value);
        synthesis_graphs.clear(); encoder_graphs.clear();
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
    std::unique_ptr<CapturedGraph> capture(const std::function<void()>& action) {
        // action: Enqueue with stable cache-slot bindings; resolve lazy work first.
        action(); check(cudaStreamSynchronize(stream.value));
        auto result = std::make_unique<CapturedGraph>();
        check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
        try { action(); }
        catch (...) { cudaGraph_t abandoned{}; cudaStreamEndCapture(stream.value, &abandoned); if (abandoned) cudaGraphDestroy(abandoned); throw; }
        check(cudaStreamEndCapture(stream.value, &result->graph));
        check(cudaGraphInstantiateWithFlags(&result->executable, result->graph, 0));
        return result;
    }
    void execute(int a, int b) {
        // a/b: Stable cache slots. A bounded LRU avoids quadratic Graph growth.
        if (!graph_enabled) { synthesis->enqueue(stream.value); return; }
        int key = a < 0 ? -1 : a * 32 + b;
        if (!synthesis_graphs.count(key)) {
            if (synthesis_graphs.size() == 16) {
                auto oldest = std::min_element(synthesis_graphs.begin(), synthesis_graphs.end(), [](const auto& x, const auto& y) { return x.second->stamp < y.second->stamp; });
                check(cudaStreamSynchronize(stream.value)); synthesis_graphs.erase(oldest);
            }
            synthesis_graphs[key] = capture([&] { synthesis->enqueue(stream.value); });
        }
        auto& graph = *synthesis_graphs[key]; graph.stamp = ++clock;
        check(cudaGraphLaunch(graph.executable, stream.value));
    }
    void encode(int slot) {
        // slot: One protected cache entry. No output is copied through a common buffer.
        encoder->bind("image", slots[slot]->image.device); encoder->bind("feature", slots[slot]->feature.device);
        if (!graph_enabled) { encoder->enqueue(stream.value); return; }
        if (!encoder_graphs[slot]) encoder_graphs[slot] = capture([&] { encoder->enqueue(stream.value); });
        check(cudaGraphLaunch(encoder_graphs[slot]->executable, stream.value));
    }
};
void validate(const void* const* planes, const int64_t* strides, int width, int bytes) {
    // planes/strides: three row descriptors; width/bytes: active row size.
    require(planes && strides, "Missing plane descriptors");
    for (int c = 0; c < 3; ++c) require(planes[c] && strides[c] >= int64_t(width) * bytes, "Invalid plane pointer or row stride");
}
void pack(void* target, const void* const* planes, const int64_t* strides, int w, int h, int bytes) {
    // target: pinned packed staging; planes/strides: positive row-padded memory.
    for (int c = 0; c < 3; ++c) for (int y = 0; y < h; ++y)
        std::memcpy(static_cast<char*>(target) + (size_t(c) * h + y) * w * bytes,
                    static_cast<const char*>(planes[c]) + y * strides[c], size_t(w) * bytes);
}
void upload(Live& live, Buffer& raw, Buffer& image, const void* const* planes, const int64_t* strides, int bytes) {
    // live/raw/image: node-owned stream and staging; planes: borrowed CPU frame.
    pack(raw.host, planes, strides, live.width, live.height, bytes);
    check(cudaMemcpyAsync(raw.device, raw.host, size_t(live.width) * live.height * 3 * bytes, cudaMemcpyHostToDevice, live.stream.value));
    check(rsb_upload(raw.device, image.device, live.width, live.height, live.pw, live.ph, bytes, live.image_bytes, live.stream.value));
}
int acquire(Live& live, uint64_t id, bool use_ids, int pinned, Buffer& raw,
            const void* const* planes, const int64_t* strides, int bytes) {
    // pinned: other frame's slot, excluded from eviction until the pair finishes.
    int selected = -1;
    for (int i = 0; i < int(live.slots.size()); ++i) {
        auto& slot = *live.slots[i];
        if (use_ids && slot.valid && slot.id == id && slot.sample_bytes == bytes) { ++live.hits; slot.stamp = ++live.clock; return i; }
        if (i != pinned && (selected < 0 || !slot.valid || (live.slots[selected]->valid && slot.stamp < live.slots[selected]->stamp))) selected = i;
    }
    require(selected >= 0, "No unpinned GPU cache slot");
    auto& slot = *live.slots[selected]; ++live.misses;
    if (slot.valid) ++live.evictions;
    slot.valid = false;
    upload(live, raw, slot.image, planes, strides, bytes);
    live.encode(selected);
    slot.id = id; slot.sample_bytes = bytes; slot.stamp = ++live.clock; slot.valid = use_ids;
    return selected;
}
void run(Live& live, uint64_t left_id, uint64_t right_id, bool use_ids, bool enabled, float timestep, int bytes,
         const void* const* left, const int64_t* ls, const void* const* right, const int64_t* rs,
         void* const* output, const int64_t* os) {
    // IDs are immutable within this node; reset before reusing IDs for new content.
    std::lock_guard<std::mutex> guard(live.mutex);
    require(!live.failed, "Worker failed previously; recreate the node");
    require((bytes == 2 || bytes == 4) && std::isfinite(timestep) && timestep >= 0 && timestep <= 1, "Invalid IO type or interpolation time");
    validate(left, ls, live.width, bytes); validate(right, rs, live.width, bytes);
    validate(const_cast<const void* const*>(output), os, live.width, bytes);
    if (!enabled) {
        // A per-request bypass never uploads, encodes, enqueues or invalidates slots.
        for (int c = 0; c < 3; ++c) for (int y = 0; y < live.height; ++y)
            std::memmove(static_cast<char*>(output[c]) + y * os[c],
                         static_cast<const char*>(left[c]) + y * ls[c], size_t(live.width) * bytes);
        ++live.requests; ++live.skipped; live.last_gpu_ms = 0;
        return;
    }
    DeviceScope scope(live.device);
    *static_cast<float*>(live.time.host) = timestep;
    try {
        if (!live.start) check(cudaEventCreate(&live.start));
        if (!live.stop) check(cudaEventCreate(&live.stop));
        check(cudaEventRecord(live.start, live.stream.value));
        int a = -1, b = -1;
        if (live.encoder) {
            int held = -1;
            if (use_ids) for (int i=0;i<int(live.slots.size());++i) {
                const auto& slot=*live.slots[i];
                if(slot.valid && slot.id==right_id && slot.sample_bytes==bytes){held=i;break;}
            }
            a = acquire(live, left_id, use_ids, held, live.raw_left, left, ls, bytes);
            b = acquire(live, right_id, use_ids, a, live.raw_right, right, rs, bytes);
            live.synthesis->bind("image0", live.slots[a]->image.device); live.synthesis->bind("image1", live.slots[b]->image.device);
            live.synthesis->bind("feature0", live.slots[a]->feature.device); live.synthesis->bind("feature1", live.slots[b]->feature.device);
        } else {
            upload(live, live.raw_left, live.left, left, ls, bytes); upload(live, live.raw_right, live.right, right, rs, bytes);
        }
        check(cudaMemcpyAsync(live.time.device, live.time.host, 4, cudaMemcpyHostToDevice, live.stream.value));
        live.execute(a, b); check(cudaEventRecord(live.stop, live.stream.value));
        check(rsb_download(live.output.device, live.raw_output.device, live.width, live.height, live.pw, live.ph, bytes, live.image_bytes, live.stream.value));
        check(cudaMemcpyAsync(live.raw_output.host, live.raw_output.device, size_t(live.width) * live.height * 3 * bytes, cudaMemcpyDeviceToHost, live.stream.value));
        check(cudaStreamSynchronize(live.stream.value)); check(cudaEventElapsedTime(&live.last_gpu_ms, live.start, live.stop));
        for (int c = 0; c < 3; ++c) for (int y = 0; y < live.height; ++y)
            std::memcpy(static_cast<char*>(output[c]) + y * os[c], static_cast<const char*>(live.raw_output.host) +
                        (size_t(c) * live.height + y) * live.width * bytes, size_t(live.width) * bytes);
        ++live.requests;
    } catch (...) { cudaStreamSynchronize(live.stream.value); live.failed = true; throw; }
}
}
RSB_API int rsbAbi() noexcept { return 3; }
RSB_API const char* rsbError() noexcept { return error.c_str(); }
RSB_API int rsbDevice(int device, char* json, size_t capacity) noexcept {
    return boundary([&] {
        require(getInferLibVersion() == NV_TENSORRT_VERSION && getInferLibBuildVersion() == TRT_BUILD_ENTERPRISE, "TensorRT runtime/SDK version mismatch");
        auto text = device_json(device); require(json && capacity > text.size(), "Device info buffer too small");
        std::memcpy(json, text.c_str(), text.size() + 1);
    });
}
RSB_API int rsbCreateV2(const wchar_t* full, const wchar_t* enc, const wchar_t* syn, int width, int height,
                      int alignment, int channels, int device, uint64_t budget, int max_slots, int graph, void** handle) noexcept {
    return boundary([&] {
        require(handle && full, "Missing engine path or handle output"); *handle = nullptr;
        require(width >= 1 && height >= 1 && width <= 8192 && height <= 8192, "Source dimensions must be 1..8192");
        require((alignment == 64 || alignment == 128) && ((enc && syn && (channels == 4 || channels == 16)) || (!enc && !syn)), "Invalid prepared model profile");
        require(max_slots >= 2 && max_slots <= 32, "GPU cache slot limit must be 2..32");
        DeviceScope scope(device);
        require(getInferLibVersion() == NV_TENSORRT_VERSION && getInferLibBuildVersion() == TRT_BUILD_ENTERPRISE, "TensorRT runtime/SDK version mismatch");
        *handle = new Live(fs::path(full), enc, syn, width, height, alignment, channels, device, budget, max_slots, graph != 0);
    });
}
RSB_API int rsbRunFrames(void* handle, uint64_t left_id, uint64_t right_id, int use_ids, int enabled, float timestep, int bytes,
    const void* const* left, const int64_t* ls, const void* const* right, const int64_t* rs, void* const* output, const int64_t* os) noexcept {
    return boundary([&] { require(handle, "Worker already closed"); run(*static_cast<Live*>(handle), left_id, right_id, use_ids != 0, enabled != 0, timestep, bytes, left, ls, right, rs, output, os); });
}
RSB_API int rsbRun(void* handle, float timestep, int bytes,
    const void* const* left, const int64_t* ls, const void* const* right, const int64_t* rs, void* const* output, const int64_t* os) noexcept {
    return rsbRunFrames(handle, 0, 1, 0, 1, timestep, bytes, left, ls, right, rs, output, os);
}
RSB_API int rsbReset(void* handle) noexcept {
    return boundary([&] {
        require(handle, "Worker already closed"); auto& live = *static_cast<Live*>(handle);
        std::lock_guard<std::mutex> guard(live.mutex); DeviceScope scope(live.device);
        require(!live.failed, "Worker failed previously; recreate the node"); check(cudaStreamSynchronize(live.stream.value));
        for (auto& slot : live.slots) slot->valid = false;
        ++live.resets;
    });
}
RSB_API int rsbStats(void* handle, char* json, size_t capacity) noexcept {
    return boundary([&] {
        require(handle, "Worker already closed"); auto& l = *static_cast<Live*>(handle); std::lock_guard<std::mutex> guard(l.mutex);
        std::ostringstream out;
        out << "{\"slots\":" << l.slots.size() << ",\"slot_bytes\":" << l.slot_bytes
            << ",\"cache_bytes\":" << l.slots.size() * l.slot_bytes << ",\"budget_bytes\":" << l.budget
            << ",\"hits\":" << l.hits << ",\"encodes\":" << l.misses << ",\"evictions\":" << l.evictions
            << ",\"resets\":" << l.resets << ",\"requests\":" << l.requests
            << ",\"skipped\":" << l.skipped << ",\"inferences\":" << l.requests - l.skipped
            << ",\"cuda_graph\":" << (!l.synthesis_graphs.empty() ? "true" : "false")
            << ",\"synthesis_graphs\":" << l.synthesis_graphs.size() << ",\"synthesis_graph_limit\":16"
            << ",\"encoder_graphs\":" << std::count_if(l.encoder_graphs.begin(), l.encoder_graphs.end(), [](const auto& p) { return bool(p); })
            << ",\"image_storage_bytes\":" << l.image_bytes << ",\"feature_storage_bytes\":" << l.feature_bytes << ",\"binding_d2d_bytes\":0"
            << ",\"last_gpu_ms\":" << l.last_gpu_ms << ",\"context_device_bytes\":"
            << l.synthesis->engine->getDeviceMemorySizeV2() + (l.encoder ? l.encoder->engine->getDeviceMemorySizeV2() : 0) << '}';
        auto value = out.str(); require(json && capacity > value.size(), "Stats buffer too small"); std::memcpy(json, value.c_str(), value.size() + 1);
    });
}
RSB_API int rsbDestroy(void* handle) noexcept {
    return boundary([&] { if (handle) { auto* live = static_cast<Live*>(handle); DeviceScope scope(live->device); delete live; } });
}
