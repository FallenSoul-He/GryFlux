#include "OmInferenceContext.h"

#include "AclEnvironment.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

OmInferenceContext::OmInferenceContext(Config config)
    : config_(std::move(config))
    , runner_(config_.modelPath, config_.deviceId)
{
}

OmInferenceContext::~OmInferenceContext()
{
    cleanup();
}

bool OmInferenceContext::init(std::string *error)
{
    if (initialized_) {
        return true;
    }

    if (!VQCodecRuntime::AclEnvironment::acquire(config_.deviceId, error)) {
        return false;
    }
    environmentAcquired_ = true;

    if (!runner_.init(error)) {
        cleanup();
        return false;
    }

    if (!runner_.prepareStage(stage_, error)) {
        cleanup();
        return false;
    }

    inputTensors_.resize(runner_.inputDescs().size());
    for (size_t i = 0; i < runner_.inputDescs().size(); ++i) {
        inputTensors_[i].dtype = runner_.inputDescs()[i].dtype;
        inputTensors_[i].dims = runner_.inputDescs()[i].dims;
    }

    outputTensors_.resize(runner_.outputDescs().size());
    initialized_ = true;
    return true;
}

size_t OmInferenceContext::inputByteSize() const
{
    if (runner_.inputDescs().empty()) {
        return 0;
    }
    return runner_.inputDescs()[0].bytes;
}

size_t OmInferenceContext::numOutputs() const
{
    return runner_.outputDescs().size();
}

size_t OmInferenceContext::outputByteSize(size_t index) const
{
    if (index >= runner_.outputDescs().size()) {
        throw std::out_of_range("OmInferenceContext: output index out of range");
    }
    return runner_.outputDescs()[index].bytes;
}

const void *OmInferenceContext::outputHostData(size_t index) const
{
    if (index >= outputTensors_.size()) {
        throw std::out_of_range("OmInferenceContext: output index out of range");
    }
    return outputTensors_[index].data();
}

aclmdlIODims OmInferenceContext::outputDims(size_t index) const
{
    if (index >= runner_.outputDescs().size()) {
        throw std::out_of_range("OmInferenceContext: output index out of range");
    }

    aclmdlIODims dims {};
    const auto &srcDims = runner_.outputDescs()[index].dims;
    dims.dimCount = std::min<size_t>(srcDims.size(), ACL_MAX_DIM_CNT);
    for (size_t i = 0; i < dims.dimCount; ++i) {
        dims.dims[i] = static_cast<int64_t>(srcDims[i]);
    }
    return dims;
}

aclFormat OmInferenceContext::outputFormat(size_t index) const
{
    if (index >= runner_.outputDescs().size()) {
        throw std::out_of_range("OmInferenceContext: output index out of range");
    }
    return runner_.outputDescs()[index].format;
}

aclDataType OmInferenceContext::outputDataType(size_t index) const
{
    if (index >= runner_.outputDescs().size()) {
        throw std::out_of_range("OmInferenceContext: output index out of range");
    }
    return runner_.outputDescs()[index].dtype;
}

void OmInferenceContext::run(const void *inputData, size_t inputBytes)
{
    if (!initialized_) {
        throw std::runtime_error("OmInferenceContext: context is not initialized.");
    }

    if (runner_.inputDescs().size() != 1) {
        throw std::runtime_error("OmInferenceContext: realesrgan expects exactly one model input.");
    }

    auto &inputTensor = inputTensors_[0];
    inputTensor.dtype = runner_.inputDescs()[0].dtype;
    inputTensor.dims = runner_.inputDescs()[0].dims;
    inputTensor.bindExternal(
        reinterpret_cast<uint8_t *>(const_cast<void *>(inputData)),
        inputBytes);

    std::string error;
    if (!runner_.copyInputsToStage(inputTensors_, stage_, &error)) {
        throw std::runtime_error("OmInferenceContext: copyInputsToStage failed: " + error);
    }
    if (!runner_.executeStage(stage_, &error)) {
        throw std::runtime_error("OmInferenceContext: executeStage failed: " + error);
    }
    if (!runner_.copyOutputsFromStage(stage_, outputTensors_, &error)) {
        throw std::runtime_error("OmInferenceContext: copyOutputsFromStage failed: " + error);
    }
}

void OmInferenceContext::cleanup() noexcept
{
    stage_.release();
    runner_.destroy();
    outputTensors_.clear();
    inputTensors_.clear();
    initialized_ = false;

    if (environmentAcquired_) {
        VQCodecRuntime::AclEnvironment::release(config_.deviceId);
        environmentAcquired_ = false;
    }
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateOmInferenceContexts(
    const std::string &omModelPath,
    int deviceId,
    size_t instanceCount)
{
    if (instanceCount == 0) {
        throw std::runtime_error("ACL context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instanceCount);
    for (size_t index = 0; index < instanceCount; ++index) {
        auto context = std::make_shared<OmInferenceContext>(
            OmInferenceContext::Config{omModelPath, deviceId});
        std::string initError;
        if (!context->init(&initError)) {
            throw std::runtime_error("OmInferenceContext init failed: " + initError);
        }
        contexts.push_back(std::move(context));
    }
    return contexts;
}
