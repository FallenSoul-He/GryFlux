#pragma once

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace VQCodecRuntime
{

size_t aclDataTypeSize(aclDataType dtype);
std::string aclDataTypeName(aclDataType dtype);
std::string dimsToString(const std::vector<int64_t> &dims);

struct TensorDesc
{
    size_t index = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_UNDEFINED;
    std::vector<int64_t> dims;
    size_t bytes = 0;
};

struct TensorData
{
    aclDataType dtype = ACL_DT_UNDEFINED;
    std::vector<int64_t> dims;
    std::vector<uint8_t> bytes;
    std::shared_ptr<uint8_t> managedBuffer;
    uint8_t *externalData = nullptr;
    size_t externalBytes = 0;

    bool empty() const
    {
        return byteSize() == 0;
    }

    size_t elementCount() const;
    bool ownsData() const { return externalData == nullptr || static_cast<bool>(managedBuffer); }
    size_t byteSize() const { return externalData != nullptr ? externalBytes : bytes.size(); }
    const uint8_t *data() const { return externalData != nullptr ? externalData : bytes.data(); }
    uint8_t *mutableData() { return externalData != nullptr ? externalData : bytes.data(); }

    void bindExternal(uint8_t *data, size_t size)
    {
        managedBuffer.reset();
        bytes.clear();
        bytes.shrink_to_fit();
        externalData = data;
        externalBytes = size;
    }

    void ensureOwnedBytes(size_t size)
    {
        managedBuffer.reset();
        externalData = nullptr;
        externalBytes = 0;
        bytes.resize(size);
    }

    bool ensureAclHostBytes(size_t size, std::string *error = nullptr)
    {
        if (managedBuffer && externalData == managedBuffer.get() && externalBytes == size) {
            return true;
        }
        managedBuffer.reset();
        bytes.clear();
        bytes.shrink_to_fit();
        void *hostPtr = nullptr;
        const aclError ret = aclrtMallocHost(&hostPtr, size);
        if (ret != ACL_ERROR_NONE) {
            externalData = nullptr;
            externalBytes = 0;
            if (error != nullptr) {
                *error = "aclrtMallocHost failed, code=" + std::to_string(ret);
            }
            return false;
        }
        managedBuffer = std::shared_ptr<uint8_t>(
            reinterpret_cast<uint8_t *>(hostPtr),
            [](uint8_t *ptr)
            {
                if (ptr != nullptr) {
                    aclrtFreeHost(ptr);
                }
            });
        externalData = managedBuffer.get();
        externalBytes = size;
        return true;
    }

    template<typename T>
    const T *dataAs() const
    {
        return reinterpret_cast<const T *>(data());
    }

    template<typename T>
    T *dataAs()
    {
        return reinterpret_cast<T *>(mutableData());
    }
};

struct ExecutionStats
{
    double inputCopyMs = 0.0;
    double executeMs = 0.0;
    double outputCopyMs = 0.0;
};

bool tensorMatchesDesc(const TensorData &tensor, const TensorDesc &desc, std::string *error = nullptr);

class DeviceExecutionStage
{
public:
    DeviceExecutionStage() = default;
    ~DeviceExecutionStage();

    DeviceExecutionStage(const DeviceExecutionStage &) = delete;
    DeviceExecutionStage &operator=(const DeviceExecutionStage &) = delete;

    bool prepare(aclrtContext context,
                 const std::vector<TensorDesc> &inputDescs,
                 const std::vector<TensorDesc> &outputDescs,
                 std::string *error);
    void release();

    bool ready() const { return ready_; }
    const std::vector<TensorDesc> &inputDescs() const { return inputDescs_; }
    const std::vector<TensorDesc> &outputDescs() const { return outputDescs_; }
    const std::vector<void *> &inputBuffers() const { return inputBuffers_; }
    const std::vector<void *> &outputBuffers() const { return outputBuffers_; }
    aclmdlDataset *inputDataset() const { return inputDataset_; }
    aclmdlDataset *outputDataset() const { return outputDataset_; }

private:
    bool buildDatasets(std::string *error);
    static void setError(std::string *error, const std::string &message);

private:
    aclrtContext context_ = nullptr;
    bool ready_ = false;
    std::vector<TensorDesc> inputDescs_;
    std::vector<TensorDesc> outputDescs_;
    std::vector<void *> inputBuffers_;
    std::vector<void *> outputBuffers_;
    aclmdlDataset *inputDataset_ = nullptr;
    aclmdlDataset *outputDataset_ = nullptr;
};

class OmModelRunner
{
public:
    OmModelRunner(std::string modelPath, int deviceId);
    ~OmModelRunner();

    bool init(std::string *error);
    void destroy();

    bool execute(const std::vector<TensorData> &inputs,
                 std::vector<TensorData> &outputs,
                 std::string *error);
    bool execute(const std::vector<TensorData> &inputs,
                 std::vector<TensorData> &outputs,
                 ExecutionStats *stats,
                 std::string *error);
    bool prepareStage(DeviceExecutionStage &stage, std::string *error);
    bool copyInputsToStage(const std::vector<TensorData> &inputs,
                           DeviceExecutionStage &stage,
                           std::string *error);
    bool executeStage(DeviceExecutionStage &stage, std::string *error);
    bool copyOutputsFromStage(DeviceExecutionStage &stage,
                              std::vector<TensorData> &outputs,
                              std::string *error);

    const std::string &modelPath() const { return modelPath_; }
    int deviceId() const { return deviceId_; }
    aclrtContext context() const { return context_; }
    aclrtStream stream() const { return stream_; }
    const std::vector<TensorDesc> &inputDescs() const { return inputDescs_; }
    const std::vector<TensorDesc> &outputDescs() const { return outputDescs_; }
    bool initialized() const { return initialized_; }

private:
    bool queryTensorDesc(bool isInput, size_t index, TensorDesc &desc, std::string *error);
    static void setError(std::string *error, const std::string &message);

private:
    std::string modelPath_;
    int deviceId_ = 0;
    bool initialized_ = false;
    bool modelLoaded_ = false;

    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t modelId_ = 0;
    aclmdlDesc *modelDesc_ = nullptr;
    std::vector<TensorDesc> inputDescs_;
    std::vector<TensorDesc> outputDescs_;
    DeviceExecutionStage reusableStage_;
    std::vector<TensorData> reusableInputHostTensors_;
};

} // namespace VQCodecRuntime
