#ifndef GRYFLUX_NODE_BASE_H
#define GRYFLUX_NODE_BASE_H

#include "data_packet.h"
#include "context.h"

namespace GryFlux
{

/**
 * @brief 节点基类（可选，用于复杂节点）
 *
 * 对于需要封装复杂逻辑的节点（如 YOLO 的解码/NMS、多阶段处理等），
 * 可以继承此基类并实现 execute() 方法。
 *
 * operator() 使其可以直接与 std::function<void(DataPacket&, Context&)> 兼容。
 *
 * @note 这是可选的！简单节点仍可使用自由函数或 lambda。
 *
 * @par 生命周期管理
 * 类节点应该用 std::shared_ptr 管理，并通过 lambda 传递：
 * @code
 * auto yoloNode = std::make_shared<YoloInferenceNode>(0.45f, 0.5f);
 * builder.addTaskNode("yolo",
 *     [yoloNode](DataPacket& packet, Context& ctx) {
 *         (*yoloNode)(packet, ctx);
 *     });
 * @endcode
 *
 * @note 为什么用 lambda？因为 std::function 要求可拷贝，但 NodeBase 禁止拷贝。
 *       Lambda 捕获 shared_ptr（可拷贝），shared_ptr 保持对象存活。
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
{
public:
    virtual ~NodeBase() = default;

    /**
     * @brief 节点执行逻辑（子类必须实现）
     *
     * @param packet 数据包引用（借用语义，不可删除/重新赋值）
     * @param ctx    上下文引用（CPU 任务传 NullContext::instance()）
     */
    virtual void execute(DataPacket &packet, Context &ctx) = 0;

    /**
     * @brief 函数调用运算符（适配 std::function）
     *
     * 此方法使 NodeBase 子类可以直接传递给 TaskFunc。
     * std::function<void(DataPacket&, Context&)> 可以接受任何实现了此运算符的对象。
     */
    void operator()(DataPacket &packet, Context &ctx)
    {
        execute(packet, ctx);
    }

protected:
    // 允许子类构造
    NodeBase() = default;

    // 禁止拷贝（节点通常包含状态，应该用 shared_ptr 显式管理）
    NodeBase(const NodeBase &) = delete;
    NodeBase &operator=(const NodeBase &) = delete;

    // 禁止移动（保持简单，需要转移所有权时用 shared_ptr）
    NodeBase(NodeBase &&) = delete;
    NodeBase &operator=(NodeBase &&) = delete;
};

} // namespace GryFlux

#endif // GRYFLUX_NODE_BASE_H
