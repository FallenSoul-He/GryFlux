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

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <type_traits>
#include "utils/noncopyable.h"

namespace GryFlux
{

    // 线程池实现
    class ThreadPool
        : private NonCopyableNonMovable
    {
    public:
        explicit ThreadPool(size_t numThreads);
        ~ThreadPool();

        // 提交任务到线程池
        template <class F>
        auto enqueue(F &&f) -> std::future<typename std::result_of<F()>::type>
        {
            using return_type = typename std::result_of<F()>::type;

            auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
            std::future<return_type> res = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                if (stop_)
                {
                    throw std::runtime_error("enqueue on stopped ThreadPool");
                }
                tasks_.emplace([task]()
                               { (*task)(); });
            }

            condition_.notify_one();
            return res;
        }

        // 获取线程池中的线程数量
        size_t getThreadCount() const
        {
            return workers_.size();
        }

        // 获取当前待处理任务数量
        size_t getTaskCount() const
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            return tasks_.size();
        }

    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        mutable std::mutex queueMutex_;
        std::condition_variable condition_;
        bool stop_;
    };
}
