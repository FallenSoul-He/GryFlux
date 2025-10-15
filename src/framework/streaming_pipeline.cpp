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
#include "framework/streaming_pipeline.h"
#include "utils/logger.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <cmath>

namespace GryFlux
{

    StreamingPipeline::StreamingPipeline(size_t workerCount, size_t schedulerThreadCount, size_t queueSize)
        : inputQueue_(std::make_shared<threadsafe_queue<std::shared_ptr<DataObject>>>()),
          outputQueue_(std::make_shared<threadsafe_queue<std::shared_ptr<DataObject>>>()),
        outputNodeId_("output"),
          running_(false),
          queueMaxSize_(queueSize),
          workerCount_(workerCount > 0 ? workerCount : 1),
          schedulerThreadCount_(schedulerThreadCount),
          processedItems_(0),
          errorCount_(0),
          totalProcessingTime_(0.0),
          profilingEnabled_(false) {}

    StreamingPipeline::~StreamingPipeline()
    {
        stop();
    }

    void StreamingPipeline::start()
    {
        if (running_.load())
        {
            return;
        }

        if (!processor_)
        {
            throw std::runtime_error("Processor function not set");
        }

        // 重置统计数据
        processedItems_.store(0);
        errorCount_.store(0);
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            taskStats_.clear();
            workerTaskStats_.clear();
            workerProcessedItems_.clear();
            workerTotalProcessingTime_.clear();
            totalProcessingTime_ = 0.0;
        }

        workerInstances_.clear();

        startTime_ = std::chrono::high_resolution_clock::now();

        running_ = true;
        input_active_ = true;
        output_active_ = true;
 
        size_t instanceCount = workerCount_ > 0 ? workerCount_ : 1;
        if (instanceCount == 0)
        {
            instanceCount = 1;
        }

        workerCount_ = instanceCount;

        builderPool_ = std::make_unique<PipelineBuilderPool>(instanceCount, schedulerThreadCount_);

        workerInstances_.reserve(instanceCount);
        for (size_t i = 0; i < instanceCount; ++i)
        {
            workerInstances_.push_back(std::make_shared<PipelineInstance>(builderPool_.get()));
        }
        launchWorkers();

        LOG.debug("[Pipeline] Started streaming pipeline with %zu workers", workerCount_);
    }

    void StreamingPipeline::stop()
    {
        if (!running_.exchange(false))
        {
            return;
        }

        input_active_ = false;

        joinWorkers();

        output_active_ = false;

        for (auto &instance : workerInstances_)
        {
            if (instance)
            {
                instance->reset();
            }
        }
        workerInstances_.clear();

        if (builderPool_)
        {
            builderPool_->shutdown();
            builderPool_.reset();
        }

        // 只有在启用性能分析时才输出统计数据
        if (profilingEnabled_)
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto totalTime = std::chrono::duration<double, std::milli>(endTime - startTime_).count();

            LOG.info("[Pipeline] Statistics:");
            LOG.info("  - Total items processed: %zu", processedItems_);
            LOG.info("  - Error count: %zu", errorCount_);
            LOG.info("  - Total running time: %.3f ms", totalTime);

            if (processedItems_ > 0)
            {
                const double avgEndToEndTime = totalTime / processedItems_;
                LOG.info("  - Average end-to-end time per item: %.3f ms", avgEndToEndTime);
                LOG.info("  - Processing rate: %.2f items/s", (processedItems_ * 1000.0 / totalTime));

                if (totalProcessingTime_ > 0.0)
                {
                    const double avgWorkerProcessingTime = totalProcessingTime_ / processedItems_;
                    LOG.info("  - Average worker processing time per item: %.3f ms", avgWorkerProcessingTime);
                }
            }

            // 输出同名任务的全局平均执行时间
            if (!taskStats_.empty())
            {
                LOG.info("[Pipeline] Global average execution time for tasks with the same name:");
                for (const auto &taskStat : taskStats_)
                {
                    const std::string &taskName = taskStat.first;
                    double totalTime = taskStat.second.first;
                    size_t count = taskStat.second.second;
                    double avgTime = totalTime / count;

                    LOG.info("  - Task [%s]: %.3f ms (average of %zu executions across all items)",
                             taskName.c_str(), avgTime, count);
                }
            }

            if (!workerProcessedItems_.empty())
            {
                LOG.info("[Pipeline] Per-worker execution statistics:");
                for (const auto &workerEntry : workerProcessedItems_)
                {
                    size_t workerIndex = workerEntry.first;
                    size_t frameCount = workerEntry.second;

                    double workerTotalTime = 0.0;
                    auto totalIt = workerTotalProcessingTime_.find(workerIndex);
                    if (totalIt != workerTotalProcessingTime_.end())
                    {
                        workerTotalTime = totalIt->second;
                    }

                    double avgFrameTime = frameCount > 0 ? workerTotalTime / static_cast<double>(frameCount) : 0.0;
                    LOG.info("  - Worker [%zu]: processed %zu items, avg frame time %.3f ms",
                             workerIndex, frameCount, avgFrameTime);

                    auto taskIt = workerTaskStats_.find(workerIndex);
                    if (taskIt != workerTaskStats_.end() && !taskIt->second.empty())
                    {
                        for (const auto &taskEntry : taskIt->second)
                        {
                            const std::string &taskName = taskEntry.first;
                            double taskTotal = taskEntry.second.first;
                            size_t taskCount = taskEntry.second.second;
                            double avgTaskTime = taskCount > 0 ? taskTotal / static_cast<double>(taskCount) : 0.0;
                            LOG.info("      * Task [%s]: %.3f ms (average of %zu executions)",
                                     taskName.c_str(), avgTaskTime, taskCount);
                        }
                    }
                }
            }
        }
        else
        {
            LOG.debug("[Pipeline] Stopped streaming pipeline");
        }
    }

    void StreamingPipeline::setProcessor(ProcessorFunction processor)
    {
        if (running_.load())
        {
            throw std::runtime_error("Cannot set processor while pipeline is running");
        }
        processor_ = std::move(processor);
    }

    bool StreamingPipeline::addInput(std::shared_ptr<DataObject> data)
    {
        if (!data || !input_active_.load())
        {
            return false;
        }

        if (queueMaxSize_ > 0)
        {
            while (input_active_.load() && static_cast<size_t>(inputQueue_->size()) >= queueMaxSize_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        if (!input_active_.load())
        {
            return false;
        }

        inputQueue_->push(std::move(data));
        return true;
    }

    bool StreamingPipeline::tryGetOutput(std::shared_ptr<DataObject> &output)
    {
        return outputQueue_->try_pop(output);
    }

    void StreamingPipeline::getOutput(std::shared_ptr<DataObject> &output)
    {
        outputQueue_->wait_and_pop(output);
    }

    void StreamingPipeline::setOutputNodeId(const std::string &outputId)
    {
        if (running_.load())
        {
            throw std::runtime_error("Cannot set output node ID while pipeline is running");
        }
        outputNodeId_ = outputId;
    }

    bool StreamingPipeline::inputEmpty() const
    {
        return inputQueue_->empty();
    }

    bool StreamingPipeline::outputEmpty() const
    {
        return outputQueue_->empty();
    }

    size_t StreamingPipeline::inputSize() const
    {
        return static_cast<size_t>(inputQueue_->size());
    }

    size_t StreamingPipeline::outputSize() const
    {
        return static_cast<size_t>(outputQueue_->size());
    }

    size_t StreamingPipeline::getProcessedItemCount() const
    {
        return processedItems_.load();
    }
    
    size_t StreamingPipeline::getErrorCount() const
    {
        return errorCount_.load();
    }

    bool StreamingPipeline::isRunning() const
    {
        return running_.load();
    }

    void StreamingPipeline::launchWorkers()
    {
        processingWorkers_.clear();
        processingWorkers_.reserve(workerCount_);
        for (size_t i = 0; i < workerCount_; ++i)
        {
            processingWorkers_.emplace_back(&StreamingPipeline::processingLoop, this, i);
        }
    }

    void StreamingPipeline::joinWorkers()
    {
        for (auto &worker : processingWorkers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        processingWorkers_.clear();
    }

    void StreamingPipeline::setSchedulerThreadCount(size_t threadCount)
    {
        if (running_.load())
        {
            throw std::runtime_error("Cannot change scheduler thread count while running");
        }
        schedulerThreadCount_ = threadCount;
    }

    void StreamingPipeline::processingLoop(size_t workerIndex)
    {
        while (running_.load() || !inputQueue_->empty())
        {
            std::shared_ptr<DataObject> input;
            if (!inputQueue_->try_pop(input))
            {
                if (!running_.load())
                {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (!input)
            {
                continue;
            }

            if (workerIndex >= workerInstances_.size())
            {
                LOG.error("[Pipeline] Worker index %zu out of range", workerIndex);
                break;
            }

            auto instance = workerInstances_[workerIndex];
            if (!instance)
            {
                LOG.error("[Pipeline] Worker instance %zu is null", workerIndex);
                break;
            }

            std::chrono::time_point<std::chrono::high_resolution_clock> frameStart;
            if (profilingEnabled_)
            {
                frameStart = std::chrono::high_resolution_clock::now();
            }

            std::unordered_map<std::string, double> frameTaskTimes;
            try
            {
                instance->prepare(processor_, input, outputNodeId_, profilingEnabled_);

                auto result = instance->execute(outputNodeId_);
                if (result)
                {
                    outputQueue_->push(std::move(result));
                    processedItems_.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    LOG.error("[Pipeline] Pipeline execution returned null output");
                    errorCount_.fetch_add(1, std::memory_order_relaxed);
                }

                if (profilingEnabled_)
                {
                    auto scheduler = instance->getBuilder()->getScheduler();
                    if (scheduler)
                    {
                        frameTaskTimes = scheduler->getTaskExecutionTimes();
                    }
                }
            }
            catch (const std::exception &ex)
            {
                LOG.error("[Pipeline] Execution error: %s", ex.what());
                errorCount_.fetch_add(1, std::memory_order_relaxed);
            }
            catch (...)
            {
                LOG.error("[Pipeline] Execution error: unknown exception");
                errorCount_.fetch_add(1, std::memory_order_relaxed);
            }

            if (profilingEnabled_)
            {
                const auto frameEnd = std::chrono::high_resolution_clock::now();
                const double frameDurationMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

                {
                    std::lock_guard<std::mutex> lock(statsMutex_);
                    totalProcessingTime_ += frameDurationMs;
                    workerTotalProcessingTime_[workerIndex] += frameDurationMs;
                    workerProcessedItems_[workerIndex] += 1;
                    for (const auto &entry : frameTaskTimes)
                    {
                        auto &stat = taskStats_[entry.first];
                        stat.first += entry.second;
                        stat.second += 1;

                        auto &workerStat = workerTaskStats_[workerIndex][entry.first];
                        workerStat.first += entry.second;
                        workerStat.second += 1;
                    }
                    if (frameTaskTimes.empty())
                    {
                        // 确保即使 frameTaskTimes 为空也记录处理次数
                        workerTaskStats_[workerIndex];
                    }
                }

                LOG.debug("[Pipeline] Processed item %zu in %.3f ms",
                          processedItems_.load(std::memory_order_relaxed), frameDurationMs);
            }

            instance->reset();
        }

        LOG.debug("[Pipeline] Processing loop completed");
    }

} // namespace GryFlux
