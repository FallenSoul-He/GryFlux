/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Object Tracker Node Implementation
 *************************************************************************************************************************/
#include "output_node.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"
#include <algorithm>

namespace PipelineNodes
{

void ObjectTrackerNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // ObjectTracker: trackVec[i] = detectionVec[i] + featureVec[i]
    // 融合两个并行分支的结果
    size_t size = std::min(p.detectionVec.size(), p.featureVec.size());
    p.trackVec.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        p.trackVec[i] = p.detectionVec[i] + p.featureVec[i];
    }

    LOG.debug("Packet %d: ObjectTracker (fusion of 2 branches, size = %zu)",
             p.id, p.trackVec.size());
}

} // namespace PipelineNodes
