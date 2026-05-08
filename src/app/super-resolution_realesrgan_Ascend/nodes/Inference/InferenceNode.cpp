#include "InferenceNode.h"

#include "context/OmInferenceContext.h"
#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <cstring>
#include <stdexcept>

namespace PipelineNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<RealEsrganPacket &>(packet);
    auto &npu = static_cast<OmInferenceContext &>(ctx);

    if (p.lr_image.empty())
    {
        throw std::runtime_error("InferenceNode: lr_image is empty.");
    }

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
    p.output_buffer.resize(outputBytes);

    std::memcpy(
        p.output_buffer.data(),
        npu.outputHostData(0),
        outputBytes);

    p.output_dims = npu.outputDims(0);
    p.output_format = npu.outputFormat(0);
    p.output_data_type = npu.outputDataType(0);

    LOG.debug("Packet %d: inference done", p.frame_id);
}

} // namespace PipelineNodes
