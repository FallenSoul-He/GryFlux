#ifndef GRYFLUX_NODE_BASE_H
#define GRYFLUX_NODE_BASE_H

#include "data_packet.h"
#include "context.h"
#include "utils/noncopyable.h"

namespace GryFlux
{

/**
 * @brief 节点基类（所有节点必须继承）
 *
 * 框架的图构建 API 只接受继承 NodeBase 的节点实现对象，
 * 每个节点通过 execute() 读写 DataPacket 内的数据字段。
 *
 * @par 示例
 * @code
 * class YoloInferenceNode : public NodeBase {
 * private:
 *     float nmsThreshold_;
 *     float confThreshold_;
 *
 *     void decodeOutput(std::vector<float>& rawOutput);
 *     void applyNMS(std::vector<Detection>& detections);
 *
 * public:
 *     YoloInferenceNode(float nmsThresh, float confThresh)
 *         : nmsThreshold_(nmsThresh), confThreshold_(confThresh) {}
 *
 *     void execute(DataPacket& packet, Context& ctx) override {
 *         auto& p = static_cast<MyDataPacket&>(packet);
 *         auto& npu = static_cast<NPUContext&>(ctx);
 *
 *         // 1. 推理
 *         npu.runInference();
 *
 *         // 2. 解码输出
 *         decodeOutput(p.rawOutput);
 *
 *         // 3. NMS 后处理
 *         applyNMS(p.detections);
 *     }
 * };
 * @endcode
 */
class NodeBase
    : private NonCopyableNonMovable
{
public:
    virtual ~NodeBase() = default;

    /**
     * @brief 节点执行逻辑（子类必须实现）
     *
     * @param packet 数据包引用（借用语义，不可删除/重新赋值）
     * @param ctx    上下文引用（CPU 任务由框架传入 None::instance()）
     */
    virtual void execute(DataPacket &packet, Context &ctx) = 0;

    /**
     * @brief 函数调用运算符（方便调用）
     */
    void operator()(DataPacket &packet, Context &ctx)
    {
        execute(packet, ctx);
    }

protected:
    // 允许子类构造
    NodeBase() = default;
};

} // namespace GryFlux

#endif // GRYFLUX_NODE_BASE_H
