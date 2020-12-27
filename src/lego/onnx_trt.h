/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: parseOnnx.h
 * \brief: parse onxx model
 * Created Date: Sunday, November 1st 2020, 3:17:14 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, November 16th 2020, 12:00:44 am
 * Modified By: raphael hao
 */

#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <memory>
#include <string>

#include "src/lego/buffer.h"
#include "src/lego/onnx_trt_config.h"
#include "src/lego/utils.h"


class Instance {
  template <typename T>
  using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

  using Severity = nvinfer1::ILogger::Severity;

 public:
  Instance(
      const ModelConfig& model_config, const size_t& min_batch_size = 1,
      const size_t& opt_batch_size = 1, const size_t& max_batch_size = 1,
      const Severity severity = Severity::kWARNING);

  bool build();

  bool infer(const size_t& num_test, const size_t& batch_size = 1);

  bool infer(const int& left, const int& right, const size_t& num_test, const size_t& batch_size = 1);

  bool multistream_infer(
      const size_t& num_test, const size_t& batch_size = 1,
      const size_t& stream_cnt = 1);

  bool subInfer(const size_t& sub_index, cudaStream_t stream);

  bool exportTrtModel(std::string save_path);

  int getSegNum();
  
  bool run(const int& left, const int& right);

  bool setBindingDimentions(const size_t& batch_size);

  std::string getModelName();

 private:
  size_t min_batch_size_;
  size_t opt_batch_size_;
  size_t max_batch_size_;
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
  // std::vector<cudaEvent_t> events_;
  nvinfer1::Dims input_dim_;
  std::string input_tensor_name_;
  nvinfer1::Dims output_dim_;
  std::string output_tensor_name_;
  std::vector<nvinfer1::Dims> sub_input_dims_;
  std::vector<nvinfer1::Dims> sub_output_dims_;
  std::vector<std::string> sub_input_tensor_names_;
  std::vector<std::string> sub_output_tensor_names_;
  std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> sub_engines_;
  std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> sub_contexts_;
  std::vector<std::shared_ptr<ManagedBuffer>> sub_buffers_;
  std::shared_ptr<ManagedBuffer> input_buffer_;
  std::shared_ptr<ManagedBuffer> output_buffer_;

  ModelConfig model_config_;
  Logger logger_;

  bool constructSubNet(
      TRTUniquePtr<nvinfer1::IBuilder>& builder,
      TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
      TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
      TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index);

  std::vector<void*> getDeviceBindings(const size_t& sub_index);

  bool exportSubTrtModel(
      std::shared_ptr<nvinfer1::ICudaEngine> engine, const std::string &sub_model_fname);

  bool run();

};
