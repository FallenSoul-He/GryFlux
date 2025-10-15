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
#include "framework/processing_task.h"
#include "utils/logger.h"

#include <utility>

namespace GryFlux
{

    ProcessingTaskPool::ProcessingTaskPool(size_t capacity, Factory factory)
        : stopped_(false)
    {
        if (capacity == 0)
        {
            capacity = 1;
        }

        addInstances(std::move(factory), capacity);
    }

    ProcessingTaskPool::~ProcessingTaskPool()
    {
        shutdown();
    }

    ProcessingTaskPool::TaskPtr ProcessingTaskPool::acquire()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]
                 { return stopped_ || !idleTasks_.empty(); });

        if (stopped_)
        {
            return nullptr;
        }

        auto task = idleTasks_.front();
        idleTasks_.pop_front();
        return task;
    }

    void ProcessingTaskPool::release(TaskPtr task)
    {
        if (!task)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_)
            {
                return;
            }
            idleTasks_.push_back(std::move(task));
        }
        cv_.notify_one();
    }

    void ProcessingTaskPool::addInstances(Factory factory, size_t count)
    {
        if (!factory)
        {
            throw std::invalid_argument("ProcessingTaskPool requires a valid factory");
        }

        if (count == 0)
        {
            count = 1;
        }

        std::vector<TaskPtr> newTasks;
        newTasks.reserve(count);

        try
        {
            for (size_t i = 0; i < count; ++i)
            {
                auto task = factory();
                if (!task)
                {
                    throw std::runtime_error("ProcessingTaskPool factory produced null instance");
                }
                newTasks.push_back(std::move(task));
            }
        }
        catch (...)
        {
            throw;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_)
            {
                throw std::runtime_error("Cannot add instances to a stopped ProcessingTaskPool");
            }

            for (auto &task : newTasks)
            {
                allTasks_.push_back(task);
                idleTasks_.push_back(std::move(task));
            }
        }

        cv_.notify_all();
    }

    void ProcessingTaskPool::shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_)
            {
                return;
            }

            stopped_ = true;
            idleTasks_.clear();
            allTasks_.clear();
        }

        cv_.notify_all();
    }

    size_t ProcessingTaskPool::available() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return idleTasks_.size();
    }

    size_t ProcessingTaskPool::capacity() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return allTasks_.size();
    }

    TaskRegistry::~TaskRegistry()
    {
        for (auto &entry : taskPools_)
        {
            if (entry.second)
            {
                entry.second->shutdown();
            }
        }
        taskPools_.clear();
    }

    std::function<std::shared_ptr<DataObject>(const std::vector<std::shared_ptr<DataObject>> &)> TaskRegistry::getProcessFunction(const std::string &taskId)
    {
        auto it = taskPools_.find(taskId);
        if (it == taskPools_.end())
        {
            throw std::runtime_error("Task not found: " + taskId);
        }

        auto pool = it->second;

        return [pool, taskId](const std::vector<std::shared_ptr<DataObject>> &inputs) -> std::shared_ptr<DataObject>
        {
            auto task = pool->acquire();
            if (!task)
            {
                LOG.error("[TaskRegistry] Failed to acquire task instance for [%s]", taskId.c_str());
                return nullptr;
            }

            struct PoolReleaser
            {
                std::shared_ptr<ProcessingTaskPool> pool;
                std::shared_ptr<ProcessingTask> task;
                ~PoolReleaser()
                {
                    if (pool && task)
                    {
                        pool->release(std::move(task));
                    }
                }
            } releaser{pool, task};

            return task->process(inputs);
        };
    }

} // namespace GryFlux
