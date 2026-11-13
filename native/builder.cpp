// SPDX-License-Identifier: MIT
#include "common.h"
#include "fusion.h"
#include <NvOnnxParser.h>
#include <iostream>
#include <map>
#include <chrono>
using namespace rsb;
namespace {
class BuildMonitor final : public nvinfer1::IProgressMonitor {
    std::mutex mutex;
    std::map<std::string, int> totals;
    std::chrono::steady_clock::time_point last{};
public:
    void phaseStart(const char* name, const char*, int32_t steps) noexcept override {
        // name/steps: TensorRT's real phase-local progress, not an overall ETA.
        try { std::lock_guard<std::mutex> guard(mutex); totals[name] = steps; } catch (...) {}
    }
    bool stepComplete(const char* name, int32_t step) noexcept override {
        // name/step: Completed work in the current optimizer phase.
        try {
            std::lock_guard<std::mutex> guard(mutex);
            auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::seconds(1)) {
                std::cerr << "TensorRT phase: " << name << " | " << step << '/' << totals[name] << std::endl;
                last = now;
            }
        } catch (...) {}
        return true;
    }
    void phaseFinish(const char* name) noexcept override {
        // name: Finished phase; its counter is no longer active.
        try { std::lock_guard<std::mutex> guard(mutex); totals.erase(name); } catch (...) {}
    }
};
}
int wmain(int argc, wchar_t** argv) {
    // argv: named UTF-16 paths/options; this executable never rewrites ONNX.
    try {
        std::map<std::wstring, std::wstring> args;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc && args.emplace(argv[i], argv[i + 1]).second, "Invalid or duplicate builder argument");
        }
        auto get = [&](const wchar_t* name) {
            require(args.count(name), "Missing builder argument: " + utf8(name));
            return args.at(name);
        };
        auto number = [&](const wchar_t* name, int low, int high) {
            auto text = get(name); size_t used{}; int value = std::stoi(text, &used);
            require(used == text.size() && value >= low && value <= high, "Builder integer out of range");
            return value;
        };
        int device = number(L"--device", 0, 255);
        DeviceScope scope(device);
        require(getInferLibVersion() == NV_TENSORRT_VERSION, "Builder TensorRT runtime/SDK mismatch");
        require(getInferLibBuildVersion() == TRT_BUILD_ENTERPRISE, "Builder TensorRT build-number mismatch");
        fs::path runtime_dir(get(L"--runtime-dir"));
        for (auto name : {L"nvinfer_10.dll", L"nvonnxparser_10.dll"})
            require(fs::equivalent(fs::u8path(module_path(name)), runtime_dir / name), "Builder loaded an unexpected NVIDIA DLL");
        if (args.count(L"--probe")) {
            std::cout << device_json(device) << std::endl;
            return 0;
        }
        int width = number(L"--width", 64, 8192), height = number(L"--height", 64, 8192);
        int alignment = number(L"--alignment", 64, 128);
        require((alignment == 64 || alignment == 128) && width % alignment == 0 && height % alignment == 0,
                "Engine shape does not match model alignment");
        auto role = get(L"--role");
        auto precision = get(L"--precision");
        require(precision == L"fp16" || precision == L"fp32", "Unsupported precision policy");
        int channels = number(L"--channels", 0, 16);
        require(role == L"full" || role == L"encoder" || role == L"synthesis", "Unsupported graph role");
        if (role != L"full") require(channels == 4 || channels == 16, "Invalid encoder width");
        int workspace = number(L"--workspace-mib", 64, 4096);
        Logger logger;
        require(rsb_register_fusions(), "Cannot register the prepared TensorRT fusion operators");
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        require(bool(builder), "Builder creation failed");
        const uint32_t flags = 1U << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        auto graph_path = fs::absolute(fs::path(get(L"--onnx")));
        auto engine_path = fs::absolute(fs::path(get(L"--engine")));
        auto timing_path = fs::absolute(fs::path(get(L"--timing")));
        auto onnx = read_file(graph_path);
        // Resolve standard ONNX external_data beside the graph, including Unicode
        // installation roots. Only this isolated builder's working directory changes;
        // all output paths are absolute before the change. No graph is rewritten.
        fs::current_path(graph_path.parent_path());
        std::cerr << "Parsing prepared ONNX: " << utf8(graph_path.wstring()) << std::endl;
        auto graph_name = utf8(graph_path.filename().wstring());
        require(parser->parse(onnx.data(), onnx.size(), graph_name.c_str()), "ONNX parse failed: " + logger.last);
        int neural_layers = 0, sampling_layers = 0, fusion_layers = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* layer = network->getLayer(i);
            if (layer->getType() == nvinfer1::LayerType::kPLUGIN_V3) ++fusion_layers;
            if (layer->getType() == nvinfer1::LayerType::kCONVOLUTION || layer->getType() == nvinfer1::LayerType::kDECONVOLUTION) {
                auto expected = precision == L"fp16" ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
                require(layer->getInput(0)->getType() == expected && layer->getOutput(0)->getType() == expected,
                        "Prepared graph neural precision does not match the selected policy");
                ++neural_layers;
            }
            if (layer->getType() == nvinfer1::LayerType::kGRID_SAMPLE) {
                require(layer->getInput(0)->getType() == nvinfer1::DataType::kFLOAT &&
                        layer->getInput(1)->getType() == nvinfer1::DataType::kFLOAT &&
                        layer->getOutput(0)->getType() == nvinfer1::DataType::kFLOAT, "GridSample must retain FP32 source/grid/output");
                ++sampling_layers;
            }
        }
        require(neural_layers > 0, "Prepared graph contains no neural layers");
        require(network->getNbInputs() == (role == L"full" ? 3 : role == L"encoder" ? 1 : 5) &&
                network->getNbOutputs() == 1, "Unexpected prepared graph IO count");
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        BuildMonitor monitor;
        config->setProgressMonitor(&monitor);
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(workspace) << 20);
        config->setBuilderOptimizationLevel(3);
        config->setMaxAuxStreams(0);
        auto* profile = builder->createOptimizationProfile();
        require(profile, "Optimization profile allocation failed");
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto* tensor = network->getInput(i);
            std::string name(tensor->getName());
            if (name == "timestep") {
                require(tensor->getType() == nvinfer1::DataType::kFLOAT, "Runtime scalar time must remain FP32");
                continue;
            }
            bool feature = name == "feature0" || name == "feature1";
            auto option = feature ? L"--feature-precision" : L"--io-precision";
            auto storage = args.count(option) ? get(option) : L"fp32";
            require(storage == L"fp16" || storage == L"fp32", "Invalid prepared storage precision");
            require(tensor->getType() == (storage == L"fp16" ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT),
                    "Prepared graph input storage differs from the descriptor");
            require((role == L"encoder" && name == "image") ||
                    (role != L"encoder" && (name == "image0" || name == "image1")) ||
                    (role == L"synthesis" && feature), "Unexpected graph input name");
            for (auto selector : {nvinfer1::OptProfileSelector::kMIN, nvinfer1::OptProfileSelector::kOPT, nvinfer1::OptProfileSelector::kMAX})
                require(profile->setDimensions(name.c_str(), selector, nvinfer1::Dims4(1, feature ? channels : 3, height, width)), "Invalid profile shape");
        }
        require(config->addOptimizationProfile(profile) >= 0, "Profile registration failed");
        std::vector<char> cache_data;
        fs::path timing(timing_path);
        if (fs::exists(timing)) cache_data = read_file(timing);
        std::unique_ptr<nvinfer1::ITimingCache> cache(config->createTimingCache(cache_data.data(), cache_data.size()));
        require(cache && config->setTimingCache(*cache, false), "Timing cache rejected");
        std::cerr << "Selecting GPU tactics and compiling the engine" << std::endl;
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        require(bool(plan), "Engine build failed: " + logger.last);
        write_file(engine_path, plan->data(), plan->size());
        std::unique_ptr<nvinfer1::IHostMemory> serialized_cache(cache->serialize());
        require(bool(serialized_cache), "Timing cache serialization failed");
        write_file(timing, serialized_cache->data(), serialized_cache->size());
        std::cout << "{\"status\":\"built\",\"engine_bytes\":" << plan->size()
                  << ",\"precision\":" << quote(utf8(precision)) << ",\"neural_layers\":" << neural_layers
                  << ",\"sampling_layers_fp32\":" << sampling_layers << ",\"fusion_layers\":" << fusion_layers
                  << ",\"tf32\":false,\"device\":" << device_json(device) << '}' << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RIFE-SuperBoost builder: " << error.what() << std::endl;
        return 1;
    }
}
