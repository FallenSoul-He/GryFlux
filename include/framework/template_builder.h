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

#include "graph_template.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace GryFlux
{

    /**
     * @brief 图构建器 - 用户友好的图构建接口
     *
     * 用于构建DAG图的拓扑结构。
     */
    class TemplateBuilder
    {
    public:
        explicit TemplateBuilder(std::shared_ptr<GraphTemplate> tmpl);

        /**
         * @brief 设置输入节点
         *
         * @param nodeId 节点ID
         * @param taskFunc 任务函数
         */
        void setInputNode(const std::string &nodeId,
                          GraphTemplate::TemplateNode::TaskFunc taskFunc);

        /**
         * @brief 设置输入节点（模板版本）- 简化类节点创建
         *
         * @tparam NodeType 节点类类型（必须继承 NodeBase）
         * @tparam Args 构造参数类型（自动推导）
         *
         * @param nodeId 节点ID
         * @param args 转发给 NodeType 构造函数的参数
         */
        template<typename NodeType, typename... Args>
        void setInputNode(const std::string &nodeId, Args&&... args)
        {
            auto node = std::make_shared<NodeType>(std::forward<Args>(args)...);
            setInputNode(nodeId, [node](DataPacket &packet, Context &ctx) {
                (*node)(packet, ctx);
            });
        }

        /**
         * @brief 添加任务节点
         *
         * @param nodeId 节点ID
         * @param taskFunc 任务函数
         * @param resourceTypeName 资源类型名称（空字符串表示CPU任务）
         * @param predecessorIds 前驱节点ID列表
         */
        void addTask(const std::string &nodeId,
                     GraphTemplate::TemplateNode::TaskFunc taskFunc,
                     const std::string &resourceTypeName,
                     const std::vector<std::string> &predecessorIds);

        /**
         * @brief 添加类节点（模板版本）- 简化类节点创建
         *
         * 用户友好的接口，自动创建节点对象并管理生命周期。
         *
         * @tparam NodeType 节点类类型（必须继承 NodeBase）
         * @tparam Args 构造参数类型（自动推导）
         *
         * @param nodeId 节点ID
         * @param resourceTypeName 资源类型名称（空字符串表示CPU任务）
         * @param predecessorIds 前驱节点ID列表
         * @param args 转发给 NodeType 构造函数的参数
         *
         * @example
         * @code
         * // 创建带配置参数的后处理节点
         * builder->addTask<PostprocessNode>(
         *     "postprocess",
         *     "",              // CPU task
         *     {"inference"},   // Dependencies
         *     0.45f,           // NMS threshold
         *     0.5f             // Confidence threshold
         * );
         * @endcode
         */
        template<typename NodeType, typename... Args>
        void addTask(const std::string &nodeId,
                     const std::string &resourceTypeName,
                     const std::vector<std::string> &predecessorIds,
                     Args&&... args)
        {
            // 1. 创建节点对象（shared_ptr 管理生命周期）
            auto node = std::make_shared<NodeType>(std::forward<Args>(args)...);

            // 2. 包装成 lambda（捕获 shared_ptr，调用 operator()）
            auto taskFunc = [node](DataPacket &packet, Context &ctx) {
                (*node)(packet, ctx);
            };

            // 3. 调用原有的 addTask 函数
            addTask(nodeId, taskFunc, resourceTypeName, predecessorIds);
        }

        /**
         * @brief 设置输出节点
         *
         * @param nodeId 节点ID
         * @param taskFunc 任务函数
         * @param predecessorIds 前驱节点ID列表
         */
        void setOutputNode(const std::string &nodeId,
                           GraphTemplate::TemplateNode::TaskFunc taskFunc,
                           const std::vector<std::string> &predecessorIds);

        /**
         * @brief 设置输出节点（模板版本）- 简化类节点创建
         *
         * @tparam NodeType 节点类类型（必须继承 NodeBase）
         * @tparam Args 构造参数类型（自动推导）
         *
         * @param nodeId 节点ID
         * @param predecessorIds 前驱节点ID列表
         * @param args 转发给 NodeType 构造函数的参数
         */
        template<typename NodeType, typename... Args>
        void setOutputNode(const std::string &nodeId,
                           const std::vector<std::string> &predecessorIds,
                           Args&&... args)
        {
            auto node = std::make_shared<NodeType>(std::forward<Args>(args)...);
            setOutputNode(nodeId, [node](DataPacket &packet, Context &ctx) {
                (*node)(packet, ctx);
            }, predecessorIds);
        }

        /**
         * @brief 完成构建（建立后继链接）
         */
        void finalizeBuild();

    private:
        std::shared_ptr<GraphTemplate> template_;
        std::unordered_map<std::string, size_t> nodeIdToIndex_;
    };

} // namespace GryFlux
