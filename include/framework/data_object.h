/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <typeinfo>
#include <typeindex>

namespace GryFlux
{

    // 基础数据对象类，所有处理数据都应继承自此类
    class DataObject
    {
    public:
        virtual ~DataObject() = default;

        // 获取数据对象的类型信息
        virtual std::type_index getType() const
        {
            return std::type_index(typeid(*this));
        }

        // 获取数据对象的类型名称
        virtual std::string getTypeName() const
        {
            return typeid(*this).name();
        }

        // 安全类型转换模板方法
        template <typename T>
        T *as()
        {
            return dynamic_cast<T *>(this);
        }

        // 安全类型转换模板方法（常量版本）
        template <typename T>
        const T *as() const
        {
            return dynamic_cast<const T *>(this);
        }

        // 检查是否为特定类型
        template <typename T>
        bool is() const
        {
            return dynamic_cast<const T *>(this) != nullptr;
        }
    };

} // namespace GryFlux
