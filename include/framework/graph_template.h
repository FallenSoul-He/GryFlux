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
#pragma once

#include "context.h"
#include "data_packet.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace GryFlux
{

    // 前向声明
    class TemplateBuilder;

    /**
     * @brief 图模板 - 不可变的DAG拓扑结构
     *
     * 存储图的拓扑结构和任务函数，所有数据包共享同一个图模板。
     * 只在初始化时构建一次，后续所有数据包复用此模板。
     */
    class GraphTemplate : public std::enable_shared_from_this<GraphTemplate>
    {
    public:
        /**
         * @brief 节点类型
         */
        enum class NodeType
        {
            Input,  // 输入节点（初始化数据包）
            Task,   // 任务节点（修改数据包）
            Output  // 输出节点（标记完成）
        };

        /**
         * @brief 模板节点（只存拓扑和函数，不存状态）
         */
        struct TemplateNode
        {
            NodeType type;
            std::string nodeId;
            std::string resourceTypeName; // 空字符串 = CPU任务

            // 任务函数（直接修改 DataPacket）
            // DataPacket& - 引用（表达借用语义，防止delete/重新赋值，强制非空）
            // Context& - 引用（防止对指针的误操作，CPU任务使用NullContext::instance()）
            using TaskFunc = std::function<void(DataPacket &, Context &)>;
            TaskFunc taskFunc;

            // 拓扑关系（存索引，不存指针）
            std::vector<size_t> predecessorIndices;
            std::vector<size_t> successorIndices;
        };

        /**
         * @brief 构建图（从用户回调）
         *
         * @param buildFunc 用户定义的图构建函数
         * @return 图模板
         */
        static std::shared_ptr<GraphTemplate> buildOnce(
            const std::function<void(TemplateBuilder *)> &buildFunc);

        /**
         * @brief 获取节点数量
         */
        size_t getNodeCount() const { return nodes_.size(); }

        /**
         * @brief 获取节点
         *
         * @param idx 节点索引
         * @return 节点引用
         */
        const TemplateNode &getNode(size_t idx) const { return nodes_[idx]; }

        /**
         * @brief 获取输入节点索引（约定：总是索引0）
         */
        size_t getInputNodeIndex() const { return 0; }

        /**
         * @brief 获取输出节点索引（约定：总是最后一个）
         */
        size_t getOutputNodeIndex() const { return nodes_.size() - 1; }

    private:
        friend class TemplateBuilder;

        std::vector<TemplateNode> nodes_; // 扁平化数组（cache-friendly）
    };

} // namespace GryFlux
