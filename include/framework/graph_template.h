/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
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

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

namespace GryFlux
{

    // 前向声明
    class TemplateBuilder;
    class NodeBase;
    class DataPacket;
    class TaskScheduler;

    namespace detail
    {
        /**
         * @brief 节点类型（GraphTemplate 内部使用）
         */
        enum class NodeType
        {
            Input,  // 输入节点（初始化数据包）
            Task,   // 任务节点（修改数据包）
            Output  // 输出节点（标记完成）
        };

        /**
         * @brief 任务定义（只存拓扑和节点定义，不存运行时状态）
         */
        struct TaskDef
        {
            NodeType type;
            std::string nodeId;
            std::string resourceTypeName; // 空字符串 = CPU任务

            // 拓扑关系（存索引，不存指针）
            // parent = 依赖的父节点（前驱），child = 被触发的子节点（后继）
            std::vector<size_t> parentIndices;
            std::vector<size_t> childIndices;

            // 节点实现对象（节点算子）
            std::shared_ptr<NodeBase> nodeImpl;
        };
    } // namespace detail

    /**
     * @brief 图模板 - 不可变的DAG拓扑结构
     *
     * 存储图的拓扑结构和节点实现对象，所有数据包共享同一个图模板。
     * 只在初始化时构建一次，后续所有数据包复用此模板。
     */
    class GraphTemplate : public std::enable_shared_from_this<GraphTemplate>
    {
    public:
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
        size_t getNodeCount() const { return tasks_.size(); }

        /**
         * @brief 获取节点实现对象（NodeBase，调试用）
         *
         * @param idx 节点索引
         * @return 节点实现对象（始终非空）
         */
        std::shared_ptr<NodeBase> getNode(size_t idx) const { return getTask(idx).nodeImpl; }

        /**
         * @brief 通过 nodeId 查找节点索引（线性查找，调试场景足够）
         *
         * @throws std::runtime_error 若未找到
         */
        size_t getNodeIndexById(const std::string &nodeId) const
        {
            for (size_t i = 0; i < tasks_.size(); ++i)
            {
                if (tasks_[i].nodeId == nodeId)
                {
                    return i;
                }
            }
            throw std::runtime_error("Node ID '" + nodeId + "' not found");
        }

        /**
         * @brief 通过 nodeId 获取节点实现对象（调试/离线调用用）
         *
         * @throws std::runtime_error 若未找到
         */
        std::shared_ptr<NodeBase> getNodeById(const std::string &nodeId) const
        {
            return getNode(getNodeIndexById(nodeId));
        }

        /**
         * @brief 获取输入节点索引（约定：总是索引0）
         */
        size_t getInputNodeIndex() const { return 0; }

        /**
         * @brief 获取输出节点索引（约定：总是最后一个）
         */
        size_t getOutputNodeIndex() const { return tasks_.size() - 1; }

    private:
        friend class TemplateBuilder;
        friend class DataPacket;
        friend class TaskScheduler;

        const detail::TaskDef &getTask(size_t idx) const { return tasks_[idx]; }

        std::vector<detail::TaskDef> tasks_; // 扁平化数组（cache-friendly）
    };

} // namespace GryFlux
