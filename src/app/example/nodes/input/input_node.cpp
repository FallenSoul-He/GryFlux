/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Input Node Implementation
 *************************************************************************************************************************/
#include "input_node.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // Initialize: rawVec[i] = id (填充 256 个元素)
    const size_t VEC_SIZE = 256;
    p.rawVec.resize(VEC_SIZE);
    for (size_t i = 0; i < VEC_SIZE; ++i)
    {
        p.rawVec[i] = static_cast<float>(p.id);
    }

    LOG.debug("Packet %d: Input (generated vector of size %zu, value = %d)",
             p.id, VEC_SIZE, p.id);
}

} // namespace PipelineNodes
