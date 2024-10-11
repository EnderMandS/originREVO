#include "pidinet.h"

inline int64_t volume(const nvinfer1::Dims &d) {
  int64_t v = 1;
  for (int64_t i = 0; i < d.nbDims; i++)
    v *= d.d[i];
  return v;
}
inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kINT32:
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kBF16:
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kUINT8:
  case nvinfer1::DataType::kINT8:
  case nvinfer1::DataType::kFP8:
    return 1;
  case nvinfer1::DataType::kINT4:
    return 0;
  }
  return 0;
}

namespace PiDi {

bool PiDiNet::init(void) {
  std::ifstream engineFile("../eigen/pidinet.engine", std::ios::binary);
  std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)),
                               std::istreambuf_iterator<char>());
  auto runtime =
      std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engineData.data(), engineData.size()));

  std::unordered_map<std::string, int32_t> mNames;
  for (int32_t i = 0, e = engine->getNbIOTensors(); i < e; i++) {
    auto const name = engine->getIOTensorName(i);
    mNames[name] = i;
    std::cout << "Binding Index: " << i << ", Binding Name: " << name
              << std::endl;
  }

  nvinfer1::Dims input_dims = engine->getTensorShape("input");
  nvinfer1::Dims output_dims = engine->getTensorShape("output");
  nvinfer1::DataType input_type = engine->getTensorDataType("input");
  nvinfer1::DataType output_type = engine->getTensorDataType("output");
  int32_t input_vecDim = engine->getTensorVectorizedDim("input");
  int32_t output_vecDim = engine->getTensorVectorizedDim("output");

  size_t input_vol = 1, output_vol = 1;
  if (-1 != input_vecDim) {
    int32_t scalarsPerVec = engine->getTensorComponentsPerElement("input");
    input_dims.d[input_vecDim] =
        (input_dims.d[input_vecDim] + scalarsPerVec - 1) / scalarsPerVec;
    input_vol *= scalarsPerVec;
  }
  input_vol *= volume(input_dims);

  if (-1 != output_vecDim) {
    int32_t scalarsPerVec = engine->getTensorComponentsPerElement("output");
    output_dims.d[output_vecDim] =
        (output_dims.d[output_vecDim] + scalarsPerVec - 1) / scalarsPerVec;
    output_vol *= scalarsPerVec;
  }
  output_vol *= volume(output_dims);

  input_buffer_size = input_vol * getElementSize(input_type);
  output_buffer_size = output_vol * getElementSize(output_type);
  if (640 * 480 * 3 * sizeof(float) != input_buffer_size) {
    std::cout << "input_buffer_size=" << input_buffer_size
              << ".Not equal to 640*480*3*sizeof(float) btyes." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (640 * 480 * sizeof(float) != output_buffer_size) {
    std::cout << "output_buffer_size=" << output_buffer_size
              << ".Not equal to 640*480*sizeof(float) btyes." << std::endl;
    exit(EXIT_FAILURE);
  }

  if (cudaSuccess != cudaMalloc(&input_device_buffer, input_buffer_size) ||
      cudaSuccess != cudaMalloc(&output_device_buffer, output_buffer_size)) {
    std::cout << "cudaMalloc error." << std::endl;
    exit(EXIT_FAILURE);
  }
  context = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());
  context->setTensorAddress("input", input_device_buffer);
  context->setTensorAddress("output", output_device_buffer);
  return true;
}

cv::Mat PiDiNet::inference(cv::Mat &img_in) {
  if (nullptr == context) {
    std::cout << "Call init before inference." << std::endl;
    exit(EXIT_FAILURE);
  }

  cv::Mat image_in = img_in.clone();
  // Convert BGR to RGB
  cvtColor(image_in, image_in, cv::COLOR_BGR2RGB);
  // Convert image to float and normalize to [0, 1]
  image_in.convertTo(image_in, CV_32FC1, 1.0 / 255);
  // Normalize the image using mean and std
  cv::Mat mean = (cv::Mat_<float>(1, 3) << 0.485, 0.456, 0.406);
  cv::Mat std = (cv::Mat_<float>(1, 3) << 0.229, 0.224, 0.225);

  std::vector<cv::Mat> channels(3);
  split(image_in, channels);
  for (int i = 0; i < 3; i++) {
    channels[i] = (channels[i] - mean.at<float>(i)) / std.at<float>(i);
  }
  cv::merge(channels, image_in);

  cudaMemcpy(input_device_buffer, image_in.data, input_buffer_size,
             cudaMemcpyHostToDevice);

  if (false == context->executeV2((void *const *)input_device_buffer)) {
    std::cout << "Error while inferring network." << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(image_out_f32.data, output_device_buffer, output_buffer_size,
             cudaMemcpyDeviceToHost);
  image_out_f32.convertTo(image_out_u8, CV_8UC1, 255.0);

  return image_out_u8.clone();
}
} // namespace PiDi
