#pragma once

#include "framework/context.h"

#include "OmModelRunner.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class OmInferenceContext : public GryFlux::Context
{
public:
    struct Config
    {
        std::string modelPath;
        int deviceId = 0;
    };

    explicit OmInferenceContext(Config config);
    ~OmInferenceContext() override;

    bool init(std::string *error);

    const Config &config() const { return config_; }
    size_t inputByteSize() const;
    size_t numOutputs() const;
    size_t outputByteSize(size_t index) const;
    const void *outputHostData(size_t index) const;

    void run(const void *inputData, size_t inputBytes);

private:
    void cleanup() noexcept;

private:
    Config config_;
    bool initialized_ = false;
    bool environmentAcquired_ = false;
    VQCodecRuntime::OmModelRunner runner_;
    VQCodecRuntime::DeviceExecutionStage stage_;
    std::vector<VQCodecRuntime::TensorData> inputTensors_;
    std::vector<VQCodecRuntime::TensorData> outputTensors_;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateOmInferenceContexts(
    const std::string &omModelPath,
    int deviceId,
    size_t instanceCount);
