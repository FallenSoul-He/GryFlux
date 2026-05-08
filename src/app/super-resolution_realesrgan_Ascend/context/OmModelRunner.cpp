#include "OmModelRunner.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>

namespace VQCodecRuntime
{
namespace
{

void setErrorText(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

std::string prefixFor(bool isInput)
{
    return isInput ? "input" : "output";
}

bool sameTensorDescs(const std::vector<TensorDesc> &lhs, const std::vector<TensorDesc> &rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].index != rhs[i].index ||
            lhs[i].dtype != rhs[i].dtype ||
            lhs[i].format != rhs[i].format ||
            lhs[i].dims != rhs[i].dims ||
            lhs[i].bytes != rhs[i].bytes) {
            return false;
        }
    }
    return true;
}

void destroyDataset(aclmdlDataset *&dataset)
{
    if (dataset == nullptr) {
        return;
    }

    const size_t count = aclmdlGetDatasetNumBuffers(dataset);
    for (size_t i = 0; i < count; ++i) {
        aclDataBuffer *buffer = aclmdlGetDatasetBuffer(dataset, i);
        if (buffer != nullptr) {
            aclDestroyDataBuffer(buffer);
        }
    }
    aclmdlDestroyDataset(dataset);
    dataset = nullptr;
}

bool addDatasetTensorBuffer(aclmdlDataset *dataset,
                            void *bufferPtr,
                            size_t bytes,
                            const std::string &label,
                            std::string *error)
{
    aclDataBuffer *buffer = aclCreateDataBuffer(bufferPtr, bytes);
    if (buffer == nullptr) {
        setErrorText(error, "aclCreateDataBuffer failed for " + label);
        return false;
    }

    if (aclmdlAddDatasetBuffer(dataset, buffer) != ACL_ERROR_NONE) {
        aclDestroyDataBuffer(buffer);
        setErrorText(error, "aclmdlAddDatasetBuffer failed for " + label);
        return false;
    }

    return true;
}

class DirectExecutionStage
{
public:
    DirectExecutionStage() = default;
    ~DirectExecutionStage()
    {
        release();
    }

    DirectExecutionStage(const DirectExecutionStage &) = delete;
    DirectExecutionStage &operator=(const DirectExecutionStage &) = delete;

    bool prepare(const std::vector<TensorData> &inputs,
                 std::vector<TensorData> &outputs,
                 const std::vector<TensorDesc> &inputDescs,
                 const std::vector<TensorDesc> &outputDescs,
                 std::string *error)
    {
        inputDataset_ = aclmdlCreateDataset();
        outputDataset_ = aclmdlCreateDataset();
        if (inputDataset_ == nullptr || outputDataset_ == nullptr) {
            setError(error, "aclmdlCreateDataset failed for direct execution stage");
            release();
            return false;
        }

        for (size_t i = 0; i < inputDescs.size(); ++i) {
            auto *buffer = aclCreateDataBuffer(const_cast<uint8_t *>(inputs[i].data()), inputDescs[i].bytes);
            if (buffer == nullptr) {
                setError(error, "aclCreateDataBuffer failed for direct input[" + std::to_string(i) + "]");
                release();
                return false;
            }
            if (aclmdlAddDatasetBuffer(inputDataset_, buffer) != ACL_ERROR_NONE) {
                aclDestroyDataBuffer(buffer);
                setError(error, "aclmdlAddDatasetBuffer failed for direct input[" + std::to_string(i) + "]");
                release();
                return false;
            }
        }

        if (outputs.size() != outputDescs.size()) {
            outputs.resize(outputDescs.size());
        }
        for (size_t i = 0; i < outputDescs.size(); ++i) {
            outputs[i].dtype = outputDescs[i].dtype;
            outputs[i].dims = outputDescs[i].dims;
            outputs[i].ensureOwnedBytes(outputDescs[i].bytes);
            auto *buffer = aclCreateDataBuffer(outputs[i].mutableData(), outputDescs[i].bytes);
            if (buffer == nullptr) {
                setError(error, "aclCreateDataBuffer failed for direct output[" + std::to_string(i) + "]");
                release();
                return false;
            }
            if (aclmdlAddDatasetBuffer(outputDataset_, buffer) != ACL_ERROR_NONE) {
                aclDestroyDataBuffer(buffer);
                setError(error, "aclmdlAddDatasetBuffer failed for direct output[" + std::to_string(i) + "]");
                release();
                return false;
            }
        }

        return true;
    }

    aclmdlDataset *inputDataset() const { return inputDataset_; }
    aclmdlDataset *outputDataset() const { return outputDataset_; }

private:
    void release()
    {
        if (inputDataset_ != nullptr) {
            const size_t count = aclmdlGetDatasetNumBuffers(inputDataset_);
            for (size_t i = 0; i < count; ++i) {
                aclDataBuffer *buffer = aclmdlGetDatasetBuffer(inputDataset_, i);
                if (buffer != nullptr) {
                    aclDestroyDataBuffer(buffer);
                }
            }
            aclmdlDestroyDataset(inputDataset_);
            inputDataset_ = nullptr;
        }

        if (outputDataset_ != nullptr) {
            const size_t count = aclmdlGetDatasetNumBuffers(outputDataset_);
            for (size_t i = 0; i < count; ++i) {
                aclDataBuffer *buffer = aclmdlGetDatasetBuffer(outputDataset_, i);
                if (buffer != nullptr) {
                    aclDestroyDataBuffer(buffer);
                }
            }
            aclmdlDestroyDataset(outputDataset_);
            outputDataset_ = nullptr;
        }
    }

    static void setError(std::string *error, const std::string &message)
    {
        if (error != nullptr) {
            *error = message;
        }
    }

private:
    aclmdlDataset *inputDataset_ = nullptr;
    aclmdlDataset *outputDataset_ = nullptr;
};

} // namespace

DeviceExecutionStage::~DeviceExecutionStage()
{
    release();
}

bool DeviceExecutionStage::prepare(aclrtContext context,
                                   const std::vector<TensorDesc> &inputDescs,
                                   const std::vector<TensorDesc> &outputDescs,
                                   std::string *error)
{
    if (ready_ &&
        context_ == context &&
        sameTensorDescs(inputDescs_, inputDescs) &&
        sameTensorDescs(outputDescs_, outputDescs)) {
        return true;
    }

    if (ready_) {
        release();
    }

    context_ = context;
    inputDescs_ = inputDescs;
    outputDescs_ = outputDescs;
    return buildDatasets(error);
}

bool DeviceExecutionStage::buildDatasets(std::string *error)
{
    if (context_ == nullptr) {
        setError(error, "null ACL context for DeviceExecutionStage");
        return false;
    }

    if (aclrtSetCurrentContext(context_) != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed for DeviceExecutionStage");
        return false;
    }

    inputDataset_ = aclmdlCreateDataset();
    outputDataset_ = aclmdlCreateDataset();
    if (inputDataset_ == nullptr || outputDataset_ == nullptr) {
        setError(error, "aclmdlCreateDataset failed for DeviceExecutionStage");
        release();
        return false;
    }

    inputBuffers_.resize(inputDescs_.size(), nullptr);
    for (size_t i = 0; i < inputDescs_.size(); ++i) {
        const aclError mallocRet = aclrtMalloc(&inputBuffers_[i], inputDescs_[i].bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (mallocRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMalloc failed for stage input[" + std::to_string(i) + "], code=" + std::to_string(mallocRet));
            release();
            return false;
        }
        aclDataBuffer *buffer = aclCreateDataBuffer(inputBuffers_[i], inputDescs_[i].bytes);
        if (buffer == nullptr) {
            setError(error, "failed to add stage input dataset buffer");
            release();
            return false;
        }
        if (aclmdlAddDatasetBuffer(inputDataset_, buffer) != ACL_ERROR_NONE) {
            aclDestroyDataBuffer(buffer);
            setError(error, "failed to add stage input dataset buffer");
            release();
            return false;
        }
    }

    outputBuffers_.resize(outputDescs_.size(), nullptr);
    for (size_t i = 0; i < outputDescs_.size(); ++i) {
        const aclError mallocRet = aclrtMalloc(&outputBuffers_[i], outputDescs_[i].bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (mallocRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMalloc failed for stage output[" + std::to_string(i) + "], code=" + std::to_string(mallocRet));
            release();
            return false;
        }
        aclDataBuffer *buffer = aclCreateDataBuffer(outputBuffers_[i], outputDescs_[i].bytes);
        if (buffer == nullptr) {
            setError(error, "failed to add stage output dataset buffer");
            release();
            return false;
        }
        if (aclmdlAddDatasetBuffer(outputDataset_, buffer) != ACL_ERROR_NONE) {
            aclDestroyDataBuffer(buffer);
            setError(error, "failed to add stage output dataset buffer");
            release();
            return false;
        }
    }

    ready_ = true;
    return true;
}

void DeviceExecutionStage::release()
{
    if (!ready_ && inputDataset_ == nullptr && outputDataset_ == nullptr) {
        return;
    }

    if (context_ != nullptr) {
        aclrtSetCurrentContext(context_);
    }

    if (inputDataset_ != nullptr) {
        for (size_t i = 0; i < inputDescs_.size(); ++i) {
            aclDataBuffer *buf = aclmdlGetDatasetBuffer(inputDataset_, i);
            if (buf != nullptr) {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(inputDataset_);
        inputDataset_ = nullptr;
    }

    if (outputDataset_ != nullptr) {
        for (size_t i = 0; i < outputDescs_.size(); ++i) {
            aclDataBuffer *buf = aclmdlGetDatasetBuffer(outputDataset_, i);
            if (buf != nullptr) {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(outputDataset_);
        outputDataset_ = nullptr;
    }

    for (void *&ptr : inputBuffers_) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
    }
    for (void *&ptr : outputBuffers_) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
    }

    inputBuffers_.clear();
    outputBuffers_.clear();
    inputDescs_.clear();
    outputDescs_.clear();
    context_ = nullptr;
    ready_ = false;
}

void DeviceExecutionStage::setError(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

size_t aclDataTypeSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_FLOAT16:
            return sizeof(uint16_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_UINT32:
            return sizeof(uint32_t);
        case ACL_UINT16:
            return sizeof(uint16_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_INT8:
            return sizeof(int8_t);
        default:
            return 0;
    }
}

std::string aclDataTypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return "ACL_FLOAT";
        case ACL_FLOAT16:
            return "ACL_FLOAT16";
        case ACL_INT64:
            return "ACL_INT64";
        case ACL_INT32:
            return "ACL_INT32";
        case ACL_UINT32:
            return "ACL_UINT32";
        case ACL_UINT16:
            return "ACL_UINT16";
        case ACL_UINT8:
            return "ACL_UINT8";
        case ACL_INT8:
            return "ACL_INT8";
        default:
            return "ACL_UNKNOWN";
    }
}

std::string dimsToString(const std::vector<int64_t> &dims)
{
    std::string text = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        text += std::to_string(dims[i]);
        if (i + 1 < dims.size()) {
            text += ", ";
        }
    }
    text += "]";
    return text;
}

size_t TensorData::elementCount() const
{
    if (dims.empty()) {
        return 0;
    }
    size_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            return 0;
        }
        const auto dimValue = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / dimValue) {
            return 0;
        }
        count *= dimValue;
    }
    return count;
}

bool tensorMatchesDesc(const TensorData &tensor, const TensorDesc &desc, std::string *error)
{
    if (tensor.dtype != desc.dtype) {
        setErrorText(error,
                     "tensor dtype mismatch, expect=" + aclDataTypeName(desc.dtype) +
                     ", got=" + aclDataTypeName(tensor.dtype));
        return false;
    }

    if (tensor.dims != desc.dims) {
        setErrorText(error,
                     "tensor dims mismatch, expect=" + dimsToString(desc.dims) +
                     ", got=" + dimsToString(tensor.dims));
        return false;
    }

    if (tensor.byteSize() != desc.bytes) {
        setErrorText(error,
                     "tensor byte-size mismatch, expect=" + std::to_string(desc.bytes) +
                     ", got=" + std::to_string(tensor.byteSize()));
        return false;
    }

    return true;
}

OmModelRunner::OmModelRunner(std::string modelPath, int deviceId)
    : modelPath_(std::move(modelPath))
    , deviceId_(deviceId)
{
}

OmModelRunner::~OmModelRunner()
{
    destroy();
}

bool OmModelRunner::init(std::string *error)
{
    if (initialized_) {
        return true;
    }

    const aclError contextRet = aclrtCreateContext(&context_, deviceId_);
    if (contextRet != ACL_ERROR_NONE) {
        setError(error, "aclrtCreateContext failed, code=" + std::to_string(contextRet));
        return false;
    }

    const aclError setContextRet = aclrtSetCurrentContext(context_);
    if (setContextRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setContextRet));
        destroy();
        return false;
    }

    const aclError streamRet = aclrtCreateStream(&stream_);
    if (streamRet != ACL_ERROR_NONE) {
        setError(error, "aclrtCreateStream failed, code=" + std::to_string(streamRet));
        destroy();
        return false;
    }

    const aclError loadRet = aclmdlLoadFromFile(modelPath_.c_str(), &modelId_);
    if (loadRet != ACL_ERROR_NONE) {
        setError(error, "aclmdlLoadFromFile failed for " + modelPath_ + ", code=" + std::to_string(loadRet));
        destroy();
        return false;
    }
    modelLoaded_ = true;

    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        setError(error, "aclmdlCreateDesc failed for " + modelPath_);
        destroy();
        return false;
    }
    if (aclmdlGetDesc(modelDesc_, modelId_) != ACL_ERROR_NONE) {
        setError(error, "aclmdlGetDesc failed for " + modelPath_);
        destroy();
        return false;
    }

    const size_t inputCount = aclmdlGetNumInputs(modelDesc_);
    inputDescs_.resize(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
        if (!queryTensorDesc(true, i, inputDescs_[i], error)) {
            destroy();
            return false;
        }
    }

    const size_t outputCount = aclmdlGetNumOutputs(modelDesc_);
    outputDescs_.resize(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
        if (!queryTensorDesc(false, i, outputDescs_[i], error)) {
            destroy();
            return false;
        }
    }

    initialized_ = true;
    return true;
}

void OmModelRunner::destroy()
{
    if (!initialized_ &&
        !modelLoaded_ &&
        context_ == nullptr &&
        stream_ == nullptr &&
        modelDesc_ == nullptr) {
        return;
    }

    if (context_ != nullptr) {
        aclrtSetCurrentContext(context_);
    }
    reusableStage_.release();
    reusableInputHostTensors_.clear();

    if (modelDesc_ != nullptr) {
        aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelLoaded_) {
        aclmdlUnload(modelId_);
        modelLoaded_ = false;
    }

    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }

    if (context_ != nullptr) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }

    inputDescs_.clear();
    outputDescs_.clear();
    initialized_ = false;
}

bool OmModelRunner::execute(const std::vector<TensorData> &inputs,
                            std::vector<TensorData> &outputs,
                            std::string *error)
{
    return execute(inputs, outputs, nullptr, error);
}

bool OmModelRunner::execute(const std::vector<TensorData> &inputs,
                            std::vector<TensorData> &outputs,
                            ExecutionStats *stats,
                            std::string *error)
{
    if (!initialized_ && !init(error)) {
        return false;
    }

    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    if (inputs.size() != inputDescs_.size()) {
        setError(error,
                 "input count mismatch for " + modelPath_ +
                 ", expect=" + std::to_string(inputDescs_.size()) +
                 ", got=" + std::to_string(inputs.size()));
        return false;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!tensorMatchesDesc(inputs[i], inputDescs_[i], error)) {
            setError(error, "input[" + std::to_string(i) + "] " + (error != nullptr ? *error : std::string("mismatch")));
            return false;
        }
        if (inputs[i].data() == nullptr) {
            setError(error, "input[" + std::to_string(i) + "] has null data");
            return false;
        }
    }

    aclmdlDataset *inputDataset = aclmdlCreateDataset();
    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    if (inputDataset == nullptr || outputDataset == nullptr) {
        destroyDataset(inputDataset);
        destroyDataset(outputDataset);
        setError(error, "aclmdlCreateDataset failed for single-copy execution");
        return false;
    }

    if (reusableInputHostTensors_.size() != inputDescs_.size()) {
        reusableInputHostTensors_.assign(inputDescs_.size(), TensorData{});
    }

    const auto inputCopyStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < inputDescs_.size(); ++i) {
        auto &stagingTensor = reusableInputHostTensors_[i];
        stagingTensor.dtype = inputDescs_[i].dtype;
        stagingTensor.dims = inputDescs_[i].dims;
        if (!stagingTensor.ensureAclHostBytes(inputDescs_[i].bytes, error)) {
            setError(error,
                     "failed to allocate host staging for input[" + std::to_string(i) + "]: " +
                     (error != nullptr ? *error : std::string("unknown error")));
            destroyDataset(inputDataset);
            destroyDataset(outputDataset);
            return false;
        }
        std::memcpy(stagingTensor.mutableData(), inputs[i].data(), inputDescs_[i].bytes);
        if (!addDatasetTensorBuffer(inputDataset,
                                    stagingTensor.mutableData(),
                                    inputDescs_[i].bytes,
                                    "input[" + std::to_string(i) + "]",
                                    error)) {
            destroyDataset(inputDataset);
            destroyDataset(outputDataset);
            return false;
        }
    }
    const auto inputCopyEnd = std::chrono::steady_clock::now();

    outputs.clear();
    outputs.resize(outputDescs_.size());
    for (size_t i = 0; i < outputDescs_.size(); ++i) {
        outputs[i].dtype = outputDescs_[i].dtype;
        outputs[i].dims = outputDescs_[i].dims;
        if (!outputs[i].ensureAclHostBytes(outputDescs_[i].bytes, error)) {
            setError(error,
                     "failed to allocate host output[" + std::to_string(i) + "]: " +
                     (error != nullptr ? *error : std::string("unknown error")));
            destroyDataset(inputDataset);
            destroyDataset(outputDataset);
            return false;
        }
        if (!addDatasetTensorBuffer(outputDataset,
                                    outputs[i].mutableData(),
                                    outputDescs_[i].bytes,
                                    "output[" + std::to_string(i) + "]",
                                    error)) {
            destroyDataset(inputDataset);
            destroyDataset(outputDataset);
            return false;
        }
    }

    const auto executeStart = std::chrono::steady_clock::now();
    const aclError execRet = aclmdlExecuteAsync(modelId_, inputDataset, outputDataset, stream_);
    if (execRet != ACL_ERROR_NONE) {
        destroyDataset(inputDataset);
        destroyDataset(outputDataset);
        setError(error, "aclmdlExecuteAsync failed for " + modelPath_ + ", code=" + std::to_string(execRet));
        return false;
    }

    const aclError syncRet = aclrtSynchronizeStream(stream_);
    if (syncRet != ACL_ERROR_NONE) {
        destroyDataset(inputDataset);
        destroyDataset(outputDataset);
        setError(error, "aclrtSynchronizeStream failed for " + modelPath_ + ", code=" + std::to_string(syncRet));
        return false;
    }
    const auto executeEnd = std::chrono::steady_clock::now();
    destroyDataset(inputDataset);
    destroyDataset(outputDataset);
    if (stats != nullptr) {
        stats->inputCopyMs =
            std::chrono::duration<double, std::milli>(inputCopyEnd - inputCopyStart).count();
        stats->executeMs =
            std::chrono::duration<double, std::milli>(executeEnd - executeStart).count();
        stats->outputCopyMs = 0.0;
    }
    return true;
}

bool OmModelRunner::prepareStage(DeviceExecutionStage &stage, std::string *error)
{
    if (!initialized_ && !init(error)) {
        return false;
    }
    return stage.prepare(context_, inputDescs_, outputDescs_, error);
}

bool OmModelRunner::copyInputsToStage(const std::vector<TensorData> &inputs,
                                      DeviceExecutionStage &stage,
                                      std::string *error)
{
    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    if (inputs.size() != inputDescs_.size()) {
        setError(error,
                 "input count mismatch for " + modelPath_ +
                 ", expect=" + std::to_string(inputDescs_.size()) +
                 ", got=" + std::to_string(inputs.size()));
        return false;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!tensorMatchesDesc(inputs[i], inputDescs_[i], error)) {
            setError(error, "input[" + std::to_string(i) + "] " + (error != nullptr ? *error : std::string("mismatch")));
            return false;
        }
        const aclError copyRet = aclrtMemcpy(stage.inputBuffers()[i],
                                             inputDescs_[i].bytes,
                                             inputs[i].data(),
                                             inputs[i].byteSize(),
                                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (copyRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMemcpy H2D failed for input[" + std::to_string(i) + "], code=" + std::to_string(copyRet));
            return false;
        }
    }

    return true;
}

bool OmModelRunner::executeStage(DeviceExecutionStage &stage, std::string *error)
{
    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    const aclError execRet = aclmdlExecuteAsync(modelId_, stage.inputDataset(), stage.outputDataset(), stream_);
    if (execRet != ACL_ERROR_NONE) {
        setError(error, "aclmdlExecuteAsync failed for " + modelPath_ + ", code=" + std::to_string(execRet));
        return false;
    }

    const aclError syncRet = aclrtSynchronizeStream(stream_);
    if (syncRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSynchronizeStream failed for " + modelPath_ + ", code=" + std::to_string(syncRet));
        return false;
    }
    return true;
}

bool OmModelRunner::copyOutputsFromStage(DeviceExecutionStage &stage,
                                         std::vector<TensorData> &outputs,
                                         std::string *error)
{
    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    if (outputs.size() != outputDescs_.size()) {
        outputs.resize(outputDescs_.size());
    }
    for (size_t i = 0; i < outputDescs_.size(); ++i) {
        outputs[i].dtype = outputDescs_[i].dtype;
        outputs[i].dims = outputDescs_[i].dims;
        if (outputs[i].byteSize() != outputDescs_[i].bytes || !outputs[i].ownsData()) {
            outputs[i].ensureOwnedBytes(outputDescs_[i].bytes);
        }

        const aclError copyRet = aclrtMemcpy(outputs[i].mutableData(),
                                             outputs[i].byteSize(),
                                             stage.outputBuffers()[i],
                                             outputDescs_[i].bytes,
                                             ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMemcpy D2H failed for output[" + std::to_string(i) + "], code=" + std::to_string(copyRet));
            return false;
        }
    }
    return true;
}

bool OmModelRunner::queryTensorDesc(bool isInput, size_t index, TensorDesc &desc, std::string *error)
{
    desc.index = index;
    desc.bytes = isInput ? aclmdlGetInputSizeByIndex(modelDesc_, index) : aclmdlGetOutputSizeByIndex(modelDesc_, index);
    desc.dtype = isInput ? aclmdlGetInputDataType(modelDesc_, index) : aclmdlGetOutputDataType(modelDesc_, index);
    desc.format = isInput ? aclmdlGetInputFormat(modelDesc_, index) : aclmdlGetOutputFormat(modelDesc_, index);

    aclmdlIODims ioDims {};
    const aclError dimRet = isInput ? aclmdlGetInputDims(modelDesc_, index, &ioDims) : aclmdlGetOutputDims(modelDesc_, index, &ioDims);
    if (dimRet != ACL_ERROR_NONE) {
        setError(error,
                 prefixFor(isInput) + "[" + std::to_string(index) + "] aclmdlGet*Dims failed, code=" + std::to_string(dimRet));
        return false;
    }

    desc.dims.assign(ioDims.dims, ioDims.dims + ioDims.dimCount);
    return true;
}

void OmModelRunner::setError(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

} // namespace VQCodecRuntime
