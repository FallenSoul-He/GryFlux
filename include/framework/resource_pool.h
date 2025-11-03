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
#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <chrono>

namespace GryFlux
{

    /**
     * @brief 资源池 - 管理有限的硬件资源
     *
     * 用于管理NPU、DPU、Tracker等硬件资源，控制并发度。
     */
    class ResourcePool
    {
    public:
        ResourcePool() = default;
        ~ResourcePool() = default;

        // 禁止拷贝和赋值
        ResourcePool(const ResourcePool &) = delete;
        ResourcePool &operator=(const ResourcePool &) = delete;

        /**
         * @brief 注册资源类型
         *
         * @param typeName 资源类型名称（如 "npu", "tracker" 等）
         * @param instances 资源实例列表
         */
        void registerResourceType(const std::string &typeName,
                                   std::vector<std::shared_ptr<Context>> instances);

        /**
         * @brief 获取资源（阻塞直到可用或超时）
         *
         * @param typeName 资源类型名称
         * @param timeout 超时时间
         * @return 资源上下文，超时返回 nullptr
         */
        std::shared_ptr<Context> acquire(const std::string &typeName,
                                          std::chrono::milliseconds timeout = std::chrono::seconds(10));

        /**
         * @brief 释放资源
         *
         * @param typeName 资源类型名称
         * @param context 资源上下文
         */
        void release(const std::string &typeName, std::shared_ptr<Context> context);

        /**
         * @brief 查询可用资源数量
         *
         * @param typeName 资源类型名称
         * @return 可用资源数量
         */
        size_t getAvailableCount(const std::string &typeName) const;

    private:
        struct ResourceTypePool
        {
            std::queue<std::shared_ptr<Context>> availableContexts;
            std::vector<std::shared_ptr<Context>> allContexts;
            std::mutex poolMutex;
            std::condition_variable availabilityCondition;
        };

        mutable std::mutex resourceRegistryMutex_;
        std::unordered_map<std::string, ResourceTypePool> resourceTypePools_;
    };

} // namespace GryFlux
