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

namespace GryFlux
{

    /**
     * @brief 硬件资源上下文基类
     *
     * 用户应继承此类实现特定的硬件资源上下文（如 NPUContext、DPUContext、TrackerContext 等）
     */
    class Context
    {
    public:
        virtual ~Context() = default;
    };

    /**
     * @brief 空上下文（None 模式）
     *
     * 用于不需要硬件资源的 CPU 任务。
     * 这样所有任务函数都可以使用 Context& 引用，避免 nullptr。
     */
    class None : public Context
    {
    public:
        // 单例模式
        static None &instance()
        {
            static None inst;
            return inst;
        }

    private:
        None() = default;
        ~None() = default;
        None(const None &) = delete;
        None &operator=(const None &) = delete;
    };

} // namespace GryFlux
