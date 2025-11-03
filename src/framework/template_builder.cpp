/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#include "framework/template_builder.h"
#include "utils/logger.h"
#include <stdexcept>

namespace GryFlux
{

    TemplateBuilder::TemplateBuilder(std::shared_ptr<GraphTemplate> tmpl)
        : template_(tmpl)
    {
    }

    void TemplateBuilder::setInputNode(const std::string &nodeId,
                                        GraphTemplate::TemplateNode::TaskFunc taskFunc)
    {
        if (nodeIdToIndex_.count(nodeId) > 0)
        {
            throw std::runtime_error("Node ID '" + nodeId + "' already exists");
        }

        GraphTemplate::TemplateNode node;
        node.type = GraphTemplate::NodeType::Input;
        node.nodeId = nodeId;
        node.resourceTypeName = ""; // 输入节点不需要资源
        node.taskFunc = taskFunc;

        size_t index = template_->nodes_.size();
        template_->nodes_.push_back(node);
        nodeIdToIndex_[nodeId] = index;

        LOG.debug("Added input node '%s' at index %zu", nodeId.c_str(), index);
    }

    void TemplateBuilder::addTask(const std::string &nodeId,
                                   GraphTemplate::TemplateNode::TaskFunc taskFunc,
                                   const std::string &resourceTypeName,
                                   const std::vector<std::string> &predecessorIds)
    {
        if (nodeIdToIndex_.count(nodeId) > 0)
        {
            throw std::runtime_error("Node ID '" + nodeId + "' already exists");
        }

        GraphTemplate::TemplateNode node;
        node.type = GraphTemplate::NodeType::Task;
        node.nodeId = nodeId;
        node.resourceTypeName = resourceTypeName;
        node.taskFunc = taskFunc;

        // 解析前驱节点索引
        for (const auto &predId : predecessorIds)
        {
            auto it = nodeIdToIndex_.find(predId);
            if (it == nodeIdToIndex_.end())
            {
                throw std::runtime_error("Predecessor node '" + predId + "' not found for node '" + nodeId + "'");
            }
            node.predecessorIndices.push_back(it->second);
        }

        size_t index = template_->nodes_.size();
        template_->nodes_.push_back(node);
        nodeIdToIndex_[nodeId] = index;

        LOG.debug("Added task node '%s' at index %zu with %zu predecessors",
                  nodeId.c_str(), index, predecessorIds.size());
    }

    void TemplateBuilder::setOutputNode(const std::string &nodeId,
                                         GraphTemplate::TemplateNode::TaskFunc taskFunc,
                                         const std::vector<std::string> &predecessorIds)
    {
        if (nodeIdToIndex_.count(nodeId) > 0)
        {
            throw std::runtime_error("Node ID '" + nodeId + "' already exists");
        }

        GraphTemplate::TemplateNode node;
        node.type = GraphTemplate::NodeType::Output;
        node.nodeId = nodeId;
        node.resourceTypeName = ""; // 输出节点不需要资源
        node.taskFunc = taskFunc;

        // 解析前驱节点索引
        for (const auto &predId : predecessorIds)
        {
            auto it = nodeIdToIndex_.find(predId);
            if (it == nodeIdToIndex_.end())
            {
                throw std::runtime_error("Predecessor node '" + predId + "' not found for node '" + nodeId + "'");
            }
            node.predecessorIndices.push_back(it->second);
        }

        size_t index = template_->nodes_.size();
        template_->nodes_.push_back(node);
        nodeIdToIndex_[nodeId] = index;

        LOG.debug("Added output node '%s' at index %zu with %zu predecessors",
                  nodeId.c_str(), index, predecessorIds.size());
    }

    void TemplateBuilder::finalizeBuild()
    {
        // 建立反向链接（后继节点）
        for (size_t i = 0; i < template_->nodes_.size(); ++i)
        {
            auto &node = template_->nodes_[i];

            for (size_t predIdx : node.predecessorIndices)
            {
                // 给前驱节点添加当前节点为后继
                template_->nodes_[predIdx].successorIndices.push_back(i);
            }
        }

        LOG.info("Finalized graph template with %zu nodes", template_->nodes_.size());
    }

} // namespace GryFlux
