#include "InferenceNode.h"

#include "context/OmInferenceContext.h"
#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <cstring>
#include <stdexcept>

namespace PipelineNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<DeepLabPacket &>(packet);
    auto &npu = static_cast<OmInferenceContext &>(ctx);

    const size_t inputBytes = p.input_tensor.size() * sizeof(float);
    if (inputBytes != npu.inputByteSize())
    {
        throw std::runtime_error("InferenceNode: input_tensor size does not match model input buffer.");
    }

    if (npu.numOutputs() == 0)
    {
        throw std::runtime_error("InferenceNode: model has no outputs.");
    }

    npu.run(p.input_tensor.data(), inputBytes);

    const size_t outputBytes = npu.outputByteSize(0);
    if (outputBytes % sizeof(float) != 0)
    {
        throw std::runtime_error("InferenceNode: output buffer size is not float-aligned.");
    }

    const size_t outputFloatCount = outputBytes / sizeof(float);
    if (p.output_tensor.size() != outputFloatCount)
    {
        p.output_tensor.resize(outputFloatCount);
    }

    std::memcpy(
        p.output_tensor.data(),
        npu.outputHostData(0),
        outputBytes);

    LOG.debug("Packet %d: inference done", p.frame_id);
}

} // namespace PipelineNodes
