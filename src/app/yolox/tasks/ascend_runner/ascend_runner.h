#pragma once

#include <string_view>
#include <optional>
#include "framework/processing_task.h"
#include "fstream"
#include "utils/logger.h"
#include "acl/acl.h"
#include "opencv2/opencv.hpp"

namespace GryFlux
{
    using ModelData = std::pair<std::unique_ptr<unsigned char[]>, std::size_t>; //buffer, size

    class AscendRunner: public ProcessingTask
    {
    public:
    	explicit AscendRunner(std::string_view model_path, const int device_id = 0,
                          const std::size_t model_width = 640, const std::size_t model_height = 640);
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

        // Function to read binary file into a buffer allocated in memory
        std::optional<ModelData> load_model(std::string_view filename);

        void dump_tensor_attr(aclmdlIODims* dims, const char* name, aclDataType dataType, aclFormat format);
        ~AscendRunner();

    private:
        aclrtContext context_;
        aclrtStream stream_;
        uint32_t model_id_;
        aclmdlDesc* model_desc_;
        aclmdlDataset* input_dataset_;
        aclmdlDataset* output_dataset_;
        
        std::size_t input_num_;
        std::size_t output_num_;
        
        std::vector<void*> input_device_buffers_;
        std::vector<void*> output_device_buffers_;
        std::vector<size_t> input_buffer_sizes_;
        std::vector<size_t> output_buffer_sizes_;
        
        std::size_t model_width_;
        std::size_t model_height_;
        int device_id_;
        
        // Performance optimization: pre-allocated buffer for reuse
        std::vector<float> host_input_buffer_;  // Reusable input buffer to avoid repeated allocations
        
        // Helper method for optimized preprocessing
        void preprocess_image(const cv::Mat& frame, float* output_buffer);
    };
}