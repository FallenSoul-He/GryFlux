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
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

namespace GryFlux
{

    std::shared_ptr<GraphTemplate> GraphTemplate::buildOnce(
        const std::function<void(TemplateBuilder *)> &buildFunc)
    {
        auto tmpl = std::make_shared<GraphTemplate>();
        auto builder = std::make_unique<TemplateBuilder>(tmpl);

        // 用户定义图结构
        buildFunc(builder.get());

        // 完成构建（建立后继链接）
        builder->finalizeBuild();

        LOG.info("GraphTemplate built with %zu nodes", tmpl->getNodeCount());

        return tmpl;
    }

} // namespace GryFlux
