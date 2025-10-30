#include "ascend_runner.h"
#include "package.h"
#include "utils/logger.h"

#define ACL_CHECK(op, error_msg) \
    do { \
        aclError ret = (op); \
        if (ret != ACL_SUCCESS) { \
            LOG.error("%s failed! ret=%d", error_msg, ret); \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

namespace GryFlux {
	AscendRunner::AscendRunner(std::string_view model_path, int device_id, std::size_t model_width, std::size_t model_height) {
		this->model_width_ = model_width;
		this->model_height_ = model_height;
		this->device_id_ = device_id;
		LOG.info("model path: %s", model_path.data());
		
		// Initialize ACL
		ACL_CHECK(aclInit(nullptr), "aclInit");
		
		// Set device
		ACL_CHECK(aclrtSetDevice(device_id_), "aclrtSetDevice");
		
		// Create context
		ACL_CHECK(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
		
		// Create stream
		ACL_CHECK(aclrtCreateStream(&stream_), "aclrtCreateStream");
		
		// Load model
		auto model_meta = load_model(model_path);
		if (!model_meta) {
			LOG.error("Fail to read model");
			throw std::runtime_error("fail to read model");
		}
		auto &[model_data, model_data_size] = *model_meta;
		
		// Load model from memory
		ACL_CHECK(aclmdlLoadFromMem(model_data.get(), model_data_size, &model_id_), "aclmdlLoadFromMem");
		
		// Create model description
		model_desc_ = aclmdlCreateDesc();
		ACL_CHECK(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");

		// Get input and output information
		input_num_ = aclmdlGetNumInputs(model_desc_);
		output_num_ = aclmdlGetNumOutputs(model_desc_);

		// Initialize input tensors
		input_dataset_ = aclmdlCreateDataset();
		input_device_buffers_.resize(input_num_);
		input_buffer_sizes_.resize(input_num_);
		
		for (size_t i = 0; i < input_num_; i++) {
			aclmdlIODims dims;
			ACL_CHECK(aclmdlGetInputDims(model_desc_, i, &dims), "aclmdlGetInputDims");
			
			aclDataType dataType = aclmdlGetInputDataType(model_desc_, i);
			aclFormat format = aclmdlGetInputFormat(model_desc_, i);
			size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
			
			input_buffer_sizes_[i] = buffer_size;
			ACL_CHECK(aclrtMalloc(&input_device_buffers_[i], buffer_size, ACL_MEM_MALLOC_HUGE_FIRST), 
					 "aclrtMalloc for input");
			
			aclDataBuffer* dataBuffer = aclCreateDataBuffer(input_device_buffers_[i], buffer_size);
			ACL_CHECK(aclmdlAddDatasetBuffer(input_dataset_, dataBuffer), "aclmdlAddDatasetBuffer for input");
			
			this->dump_tensor_attr(&dims, "input", dataType, format);
		}

		// Initialize output tensors  
		output_dataset_ = aclmdlCreateDataset();
		output_device_buffers_.resize(output_num_);
		output_buffer_sizes_.resize(output_num_);
		
		for (size_t i = 0; i < output_num_; i++) {
			aclmdlIODims dims;
			ACL_CHECK(aclmdlGetOutputDims(model_desc_, i, &dims), "aclmdlGetOutputDims");
			
			aclDataType dataType = aclmdlGetOutputDataType(model_desc_, i);
			aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
			size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
			
			output_buffer_sizes_[i] = buffer_size;
			ACL_CHECK(aclrtMalloc(&output_device_buffers_[i], buffer_size, ACL_MEM_MALLOC_HUGE_FIRST), 
					 "aclrtMalloc for output");
			
			aclDataBuffer* dataBuffer = aclCreateDataBuffer(output_device_buffers_[i], buffer_size);
			ACL_CHECK(aclmdlAddDatasetBuffer(output_dataset_, dataBuffer), "aclmdlAddDatasetBuffer for output");
			
			this->dump_tensor_attr(&dims, "output", dataType, format);
		}

		// Pre-allocate host input buffer to avoid repeated allocations
		size_t input_elem_count = model_width_ * model_height_ * 3; // RGB/BGR 3 channels
		host_input_buffer_.resize(input_elem_count);
		
		LOG.info("Pre-allocated input buffer: %zu elements (%zu bytes)", 
				 input_elem_count, input_elem_count * sizeof(float));
	}

	void AscendRunner::dump_tensor_attr(aclmdlIODims* dims, const char* name, aclDataType dataType, aclFormat format) {
		const char* typeStr = [dataType]() {
			switch (dataType) {
				case ACL_FLOAT: return "FLOAT";
				case ACL_FLOAT16: return "FLOAT16"; 
				case ACL_INT8: return "INT8";
				case ACL_INT32: return "INT32";
				case ACL_UINT8: return "UINT8";
				default: return "UNKNOWN";
			}
		}();
		
		const char* fmtStr = [format]() {
			switch (format) {
				case ACL_FORMAT_NCHW: return "NCHW";
				case ACL_FORMAT_NHWC: return "NHWC";
				case ACL_FORMAT_ND: return "ND";
				case ACL_FORMAT_NC1HWC0: return "NC1HWC0";
				default: return "UNKNOWN";
			}
		}();

		LOG.info("MODEL TENSOR: %s | dims=%zu | shape=[%zu,%zu,%zu,%zu] | type=%s | format=%s",
				name, dims->dimCount, 
				dims->dimCount > 0 ? dims->dims[0] : 0,
				dims->dimCount > 1 ? dims->dims[1] : 0, 
				dims->dimCount > 2 ? dims->dims[2] : 0,
				dims->dimCount > 3 ? dims->dims[3] : 0,
				typeStr, fmtStr);
	}

	void AscendRunner::preprocess_image(const cv::Mat& frame, float* output_buffer) {
		
		const int channels = 3;
		const size_t channel_size = model_width_ * model_height_;
		
		// Convert to float and rearrange from HWC to CHW in a single loop
		if (frame.type() == CV_8UC3) {
			// Most common case: 8-bit BGR image
			const uint8_t* src = frame.data;
			for (size_t h = 0; h < model_height_; ++h) {
				for (size_t w = 0; w < model_width_; ++w) {
					size_t src_idx = (h * model_width_ + w) * channels;
					size_t dst_idx = h * model_width_ + w;
					// BGR to CHW (keeping BGR order as original)
					output_buffer[dst_idx] = static_cast<float>(src[src_idx]); // B
					output_buffer[channel_size + dst_idx] = static_cast<float>(src[src_idx + 1]); // G
					output_buffer[2 * channel_size + dst_idx] = static_cast<float>(src[src_idx + 2]); // R
				}
			}
		} else if (frame.type() == CV_32FC3) {
			// Already float type
			const float* src = reinterpret_cast<const float*>(frame.data);
			for (size_t h = 0; h < model_height_; ++h) {
				for (size_t w = 0; w < model_width_; ++w) {
					size_t src_idx = (h * model_width_ + w) * channels;
					size_t dst_idx = h * model_width_ + w;
					output_buffer[dst_idx] = src[src_idx]; // B
					output_buffer[channel_size + dst_idx] = src[src_idx + 1]; // G
					output_buffer[2 * channel_size + dst_idx] = src[src_idx + 2]; // R
				}
			}
		} else {
			// Fallback to original method for other types
			cv::Mat float_frame;
			frame.convertTo(float_frame, CV_32F);
			
			std::vector<cv::Mat> channels_vec(channels);
			cv::split(float_frame, channels_vec);
			
			for (int i = 0; i < channels; i++) {
				memcpy(output_buffer + i * channel_size, channels_vec[i].data, channel_size * sizeof(float));
			}
		}
	}

	std::optional<ModelData> AscendRunner::load_model(std::string_view filename) {
		std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			LOG.error("Failed to open file: %s", filename.data());
			return std::nullopt;
		}

		size_t fileSize = static_cast<size_t>(file.tellg());
		file.seekg(0);
		
		auto buffer = std::make_unique<unsigned char[]>(fileSize);
		if (!buffer || !file.read(reinterpret_cast<char*>(buffer.get()), fileSize)) {
			LOG.error("Failed to read model file");
			return std::nullopt;
		}
		
		return std::make_pair(std::move(buffer), fileSize);
	}

    std::shared_ptr<DataObject> AscendRunner::process(const std::vector<std::shared_ptr<DataObject>> &inputs) {

		auto input_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
		auto frame = input_data->get_data();
		
		if (frame.rows != static_cast<int>(model_height_) || frame.cols != static_cast<int>(model_width_)) {
			LOG.error("Input image size (%dx%d) doesn't match model input size (%zux%zu)", 
					  frame.cols, frame.rows, model_width_, model_height_);
			return nullptr;
		}
		
		// Optimized preprocessing: use pre-allocated buffer and single-pass conversion
		preprocess_image(frame, host_input_buffer_.data());

		// Set context before operations
		ACL_CHECK(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
		
		// Copy input data to device
		size_t input_size = host_input_buffer_.size() * sizeof(float);
		ACL_CHECK(aclrtMemcpy(input_device_buffers_[0], input_buffer_sizes_[0], 
							 host_input_buffer_.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 
							 "aclrtMemcpy input to device");
		// Execute model
		ACL_CHECK(aclmdlExecute(model_id_, input_dataset_, output_dataset_), "aclmdlExecute");

		// Process outputs
		auto output_data = std::make_shared<RunnerPackage>(model_width_, model_height_);
		
		for (size_t i = 0; i < output_num_; i++) {
			aclmdlIODims dims;
			ACL_CHECK(aclmdlGetOutputDims(model_desc_, i, &dims), "aclmdlGetOutputDims");

			size_t n_elems = 1;
			for (size_t j = 0; j < dims.dimCount; j++) {
				n_elems *= dims.dims[j];
			}
			
			std::shared_ptr<float[]> output_host(new float[n_elems]);

			ACL_CHECK(aclrtMemcpy(output_host.get(), n_elems * sizeof(float),
									output_device_buffers_[i], output_buffer_sizes_[i],
									ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy output from device");
			size_t height = dims.dimCount > 2 ? dims.dims[2] : 1;
			size_t width = dims.dimCount > 3 ? dims.dims[3] : 1;
			output_data->push_data({output_host, n_elems}, {height, width});
		}
		
		return output_data;
	}

	AscendRunner::~AscendRunner() {
		// Cleanup datasets and device buffers
		auto cleanup_dataset = [](aclmdlDataset* dataset, std::vector<void*>& buffers) {
			if (dataset) {
				for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); i++) {
					aclDestroyDataBuffer(aclmdlGetDatasetBuffer(dataset, i));
				}
				aclmdlDestroyDataset(dataset);
			}
			for (auto& buffer : buffers) {
				if (buffer) aclrtFree(buffer);
			}
		};
		
		cleanup_dataset(input_dataset_, input_device_buffers_);
		cleanup_dataset(output_dataset_, output_device_buffers_);

		// Cleanup model and runtime
		if (model_desc_) aclmdlDestroyDesc(model_desc_);
		aclmdlUnload(model_id_);
		if (stream_) aclrtDestroyStream(stream_);
		if (context_) aclrtDestroyContext(context_);
		aclrtResetDevice(device_id_);
		aclFinalize();
	}
}