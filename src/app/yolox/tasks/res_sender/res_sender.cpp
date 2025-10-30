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

#include "res_sender.h"
#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    static std::vector<std::string> g_classes = {
        "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
        "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
        "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
        "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "};

    std::shared_ptr<DataObject> ResSender::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // input picture and object info
        if (inputs.size() != 2)
            return nullptr;

        auto image_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        int img_id = image_data->get_id();
        auto img = image_data->get_data();

        auto object_data = std::dynamic_pointer_cast<ObjectPackage>(inputs[1]);
        auto objects = object_data->get_data();
        int object_count = objects.size();

        // 画框
        for (int i = 0; i < object_count; ++i)
        {
            auto obj = objects[i];
            cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0, 255, 0), 2);
            std::string label = g_classes[obj.class_id] + ": " + std::to_string(obj.prob);
            cv::putText(img, label, cv::Point(obj.left, obj.top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        return std::make_shared<ImagePackage>(img, img_id);
    }
}
