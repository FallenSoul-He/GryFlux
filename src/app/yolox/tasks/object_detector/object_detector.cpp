#include "object_detector.h"
#include "package.h"
#include "utils/logger.h"
namespace GryFlux
{
    const int OBJ_CLASS_NUM = 80;
    inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

    static int process_fp32(float *input, int grid_h, int grid_w, int height, int width, int stride,
                            std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold)
    {
        int validCount = 0;
        int grid_len = grid_h * grid_w;

        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                float box_confidence = input[4 * grid_len + i * grid_w + j];
                if (box_confidence >= threshold)
                {
                    int offset = i * grid_w + j;
                    float *in_ptr = input + offset;
                    float box_x = *in_ptr;
                    float box_y = in_ptr[grid_len];
                    float box_w = in_ptr[2 * grid_len];
                    float box_h = in_ptr[3 * grid_len];
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    float maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > threshold)
                    {
                        objProbs.push_back(maxClassProbs * box_confidence);
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
        return validCount;
    }
    static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
    {
        float key;
        int key_index;
        int low = left;
        int high = right;
        if (left < right)
        {
            key_index = indices[left];
            key = input[left];
            while (low < high)
            {
                while (low < high && input[high] <= key)
                {
                    high--;
                }
                input[low] = input[high];
                indices[low] = indices[high];
                while (low < high && input[low] >= key)
                {
                    low++;
                }
                input[high] = input[low];
                indices[high] = indices[low];
            }
            input[low] = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }
    static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                                float ymax1)
    {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
                int filterId, float threshold)
    {
        for (int i = 0; i < validCount; ++i)
        {
            int n = order[i];
            if (n == -1 || classIds[n] != filterId)
            {
                continue;
            }
            for (int j = i + 1; j < validCount; ++j)
            {
                int m = order[j];
                if (m == -1 || classIds[m] != filterId)
                {
                    continue;
                }
                float xmin0 = outputLocations[n * 4 + 0];
                float ymin0 = outputLocations[n * 4 + 1];
                float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
                float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

                float xmin1 = outputLocations[m * 4 + 0];
                float ymin1 = outputLocations[m * 4 + 1];
                float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
                float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

                float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold)
                {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }
    std::shared_ptr<DataObject> ObjectDetector::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // runner and preprocess
        if (inputs.size() != 2) {
            LOG.error("ObjectDetector: inputs size is not 2");
            return nullptr;
        }

        auto preprocess_img = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        int img_id = preprocess_img->get_id();
        float scale = preprocess_img->get_scale();
        int x_pad = preprocess_img->get_x_pad();
        int y_pad = preprocess_img->get_y_pad();
        int model_in_w = preprocess_img->get_width();
        int model_in_h = preprocess_img->get_height();
        LOG.info("Image preprocess scale: %f, x_pad: %d, y_pad: %d", scale, x_pad, y_pad);

        auto input_data = std::dynamic_pointer_cast<RunnerPackage>(inputs[1]);
        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int> classId;
        int valid = 0;
        for (std::size_t i = 0; i < input_data->size(); i++) {
            auto [output_data, output_cnt] = input_data->get_output()[i];
            auto [grid_h, grid_w] = input_data->get_grid()[i];
            valid += process_fp32(output_data.get(), grid_h, grid_w, input_data->get_model_height(), input_data->get_model_width(),
                            input_data->get_model_width() / grid_w, filterBoxes, objProbs, classId, threshold_);
            LOG.info("Output cnt: %d, grid size: { %d }x{ %d }", output_cnt, grid_w, grid_h);
        }
        LOG.info("valid count: %d", valid);
            // no object detect
        if (valid <= 0)
        {
            return nullptr;
        }
        std::vector<int> indexArray;
        for (int i = 0; i < valid; ++i)
        {
            indexArray.push_back(i);
        }
        quick_sort_indice_inverse(objProbs, 0,  valid- 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));

        for (auto c : class_set)
        {
            nms(valid, filterBoxes, classId, indexArray, c,threshold_);
        }

        int last_count = 0;
        std::shared_ptr<ObjectPackage> object_data = std::make_shared<ObjectPackage>(img_id);
        /* box valid detect target */
        for (int i = 0; i < valid; ++i)
        {
            if (indexArray[i] == -1 || last_count >= 80)
            {
                continue;
            }
            int n = indexArray[i];

            float x1 = filterBoxes[n * 4 + 0] - x_pad;
            float y1 = filterBoxes[n * 4 + 1] - y_pad;
            float x2 = x1 + filterBoxes[n * 4 + 2];
            float y2 = y1 + filterBoxes[n * 4 + 3];
            int id = classId[n];
            float obj_conf = objProbs[i];

            ObjectInfo obj_info;
            obj_info.left = (int)(clamp(x1, 0, model_in_w) / scale);
            obj_info.top = (int)(clamp(y1, 0, model_in_h) / scale);
            obj_info.right = (int)(clamp(x2, 0, model_in_w) / scale);
            obj_info.bottom = (int)(clamp(y2, 0, model_in_h) / scale);
            obj_info.class_id = id;
            obj_info.prob = obj_conf;
            object_data->push_data(obj_info);
            last_count++;
            LOG.info("x1: %d, y1: %d, x2: %d, y2: %d, id: %d, obj_conf: %f", obj_info.left, obj_info.top, obj_info.right, obj_info.bottom, obj_info.class_id, obj_info.prob);
        }

        return object_data;
    }

}
