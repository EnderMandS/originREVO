#pragma once
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <math.h>
#include <memory>
#include <opencv2/opencv.hpp>

namespace PiDi {

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      std::cerr << "\033[1;31m[INTERNAL_ERROR]\033[0m " << msg << std::endl;
      break;
    case Severity::kERROR:
      std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl;
      break;
    case Severity::kWARNING:
      std::cerr << "\033[1;33m[WARNING]\033[0m " << msg << std::endl;
      break;
    case Severity::kINFO:
      std::cout << "\033[1;37m[INFO]\033[0m " << msg << std::endl;
      break;
    case Severity::kVERBOSE:
      std::cout << "\033[1;37m[VERBOSE]\033[0m " << msg << std::endl;
      break;
    default:
      std::cout << "[UNKNOWN] " << msg << std::endl;
      break;
    }
  }
};

class PiDiNet {
public:
  bool init(void);
  cv::Mat inference(cv::Mat &img_in);

private:
  std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
  Logger logger;
  cv::Mat image_out_f32 = cv::Mat(640, 480, CV_32FC1),
          image_out_u8 = cv::Mat(640, 480, CV_8UC1);

  size_t input_buffer_size = 0, output_buffer_size = 0;
  void *input_device_buffer = nullptr;
  void *output_device_buffer = nullptr;
};
} // namespace PiDi
