/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: onnx_trt.cc
 * \brief: implementation of onnx_trt
 * Created Date: Sunday, November 1st 2020, 9:28:40 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Wednesday, November 18th 2020, 10:17:20 am
 * Modified By: raphael hao
 */


#include "src/lego/onnx_trt.h"

#include <cuda_runtime.h>

#include "src/lego/buffer.h"


Instance::Instance(
    const ModelConfig& model_config, const size_t& min_batch_size,
    const size_t& opt_batch_size, const size_t& max_batch_size,
    const Severity severity)
    : min_batch_size_(min_batch_size), opt_batch_size_(opt_batch_size),
      max_batch_size_(max_batch_size)
{
  CHECK(cudaStreamCreate(&stream_));
  CHECK(cudaEventCreate(&start_));
  CHECK(cudaEventCreate(&stop_));
  model_config_ = model_config;
  logger_.setSeverity(severity);
}

bool
Instance::build()
{
  size_t sub_model_cnt = model_config_.getSegNum();
  sub_engines_.clear();
  sub_contexts_.clear();
  sub_buffers_.clear();
  sub_input_dims_.resize(sub_model_cnt);
  sub_input_tensor_names_.resize(sub_model_cnt);
  sub_output_dims_.resize(sub_model_cnt);
  sub_output_tensor_names_.resize(sub_model_cnt);
  for (size_t i = 0; i < sub_model_cnt; i++) {
    auto builder =
        TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
      std::cout << "Failed to create " << i << "-th sub builder";
      return false;
    }
    const auto explicit_batch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TRTUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch));
    if (!network) {
      std::cout << "Failed to create " << i << "-th sub network";
      return false;
    }
    auto config =
        TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
      return false;
    }

    auto paser = TRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!paser) {
      std::cout << "Failed to create " << i << "-th sub parser";
      return false;
    }
    auto constructed = constructSubNet(builder, network, config, paser, i);
    if (!constructed) {
      std::cout << "Failed to construct " << i << "-th sub network";
      return false;
    }
    auto tmp_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), TRTDeleter());
    if (!tmp_engine) {
      std::cout << "Failed to create " << i << "-th sub engine";
      return false;
    }

    sub_engines_.emplace_back(tmp_engine);
    sub_contexts_.emplace_back(std::shared_ptr<nvinfer1::IExecutionContext>(
        tmp_engine->createExecutionContext(), TRTDeleter()));
  }

  input_dim_ = sub_input_dims_[0];
  input_tensor_name_ = sub_input_tensor_names_[0];
  output_dim_ = sub_output_dims_[sub_model_cnt - 1];
  output_tensor_name_ = sub_output_tensor_names_[sub_model_cnt - 1];

  input_buffer_ = sub_buffers_[0];

  return true;
}

bool
Instance::constructSubNet(
    TRTUniquePtr<nvinfer1::IBuilder>& builder,
    TRTUniquePtr<nvinfer1::INetworkDefinition>& network,
    TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
    TRTUniquePtr<nvonnxparser::IParser>& parser, size_t model_index)
{
  auto parsed = parser->parseFromFile(
      locateFile(
          model_config_.getSegFileName(model_index) + ".onnx",
          model_config_.getDataDir())
          .c_str(),
      static_cast<int>(logger_.getSeverity()));
  if (!parsed) {
    return false;
  }

  auto profile = builder->createOptimizationProfile();

  assert(network->getNbInputs() == 1);
  sub_input_dims_[model_index] = network->getInput(0)->getDimensions();
  sub_input_tensor_names_[model_index] = network->getInput(0)->getName();
  assert(sub_input_dims_[model_index].nbDims == 4);
  // set input buffer
  sub_buffers_.emplace_back(std::make_shared<ManagedBuffer>());
  sub_buffers_.back()->deviceBuffer.resize(nvinfer1::Dims4{
      static_cast<int>(max_batch_size_), sub_input_dims_[model_index].d[1],
      sub_input_dims_[model_index].d[2], sub_input_dims_[model_index].d[3]});
  sub_buffers_.back()->hostBuffer.resize(nvinfer1::Dims4{
      static_cast<int>(max_batch_size_), sub_input_dims_[model_index].d[1],
      sub_input_dims_[model_index].d[2], sub_input_dims_[model_index].d[3]});
  // set input and output dimensions
  bool if_set = false;
  if_set = profile->setDimensions(
      sub_input_tensor_names_[model_index].c_str(),
      nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims4{
          static_cast<int>(min_batch_size_), sub_input_dims_[model_index].d[1],
          sub_input_dims_[model_index].d[2],
          sub_input_dims_[model_index].d[3]});
  assert(if_set);
  if_set = profile->setDimensions(
      sub_input_tensor_names_[model_index].c_str(),
      nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims4{
          static_cast<int>(opt_batch_size_), sub_input_dims_[model_index].d[1],
          sub_input_dims_[model_index].d[2],
          sub_input_dims_[model_index].d[3]});
  assert(if_set);
  if_set = profile->setDimensions(
      sub_input_tensor_names_[model_index].c_str(),
      nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims4{
          static_cast<int>(max_batch_size_), sub_input_dims_[model_index].d[1],
          sub_input_dims_[model_index].d[2],
          sub_input_dims_[model_index].d[3]});
  assert(if_set);

  assert(network->getNbOutputs() == 1);
  sub_output_dims_[model_index] = network->getOutput(0)->getDimensions();
  sub_output_tensor_names_[model_index] = network->getOutput(0)->getName();

  if (model_index + 1 == model_config_.getSegNum()) {
    assert(sub_output_dims_[model_index].nbDims == 2);
    if_set = profile->setDimensions(
        sub_output_tensor_names_[model_index].c_str(),
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims2{
            static_cast<int>(min_batch_size_),
            sub_output_dims_[model_index].d[1]});
    assert(if_set);
    if_set = profile->setDimensions(
        sub_output_tensor_names_[model_index].c_str(),
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims2{
            static_cast<int>(opt_batch_size_),
            sub_output_dims_[model_index].d[1]});
    assert(if_set);
    if_set = profile->setDimensions(
        sub_output_tensor_names_[model_index].c_str(),
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims2{
            static_cast<int>(max_batch_size_),
            sub_output_dims_[model_index].d[1]});
    assert(if_set);
    sub_buffers_.emplace_back(std::make_shared<ManagedBuffer>());
    sub_buffers_.back()->deviceBuffer.resize(nvinfer1::Dims2{
        static_cast<int>(max_batch_size_), sub_output_dims_[model_index].d[1]});
    sub_buffers_.back()->hostBuffer.resize(nvinfer1::Dims2{
        static_cast<int>(max_batch_size_), sub_output_dims_[model_index].d[1]});
  } else {
    assert(sub_output_dims_[model_index].nbDims == 4);
    if_set = profile->setDimensions(
        sub_output_tensor_names_[model_index].c_str(),
        nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{
            static_cast<int>(min_batch_size_),
            sub_output_dims_[model_index].d[1],
            sub_output_dims_[model_index].d[2],
            sub_output_dims_[model_index].d[3]});
    assert(if_set);
    if_set = profile->setDimensions(
        sub_output_tensor_names_[model_index].c_str(),
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{
            static_cast<int>(opt_batch_size_),
            sub_output_dims_[model_index].d[1],
            sub_output_dims_[model_index].d[2],
            sub_output_dims_[model_index].d[3]});
    assert(if_set);
    if_set = profile->setDimensions(
        sub_output_tensor_names_[model_index].c_str(),
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{
            static_cast<int>(max_batch_size_),
            sub_output_dims_[model_index].d[1],
            sub_output_dims_[model_index].d[2],
            sub_output_dims_[model_index].d[3]});
    assert(if_set);
  }

  config->addOptimizationProfile(profile);

  config->setMaxWorkspaceSize(3_GiB);
  if (model_config_.fp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  return true;
}

bool
Instance::subInfer(const size_t& model_index, cudaStream_t stream)
{
  return sub_contexts_[model_index]->enqueueV2(
      getDeviceBindings(model_index).data(), stream, nullptr);
}

std::vector<void*>
Instance::getDeviceBindings(const size_t& model_index)
{
  return std::vector<void*>{
      sub_buffers_[model_index]->deviceBuffer.data(),
      sub_buffers_[model_index + 1]->deviceBuffer.data()};
}

bool
Instance::setBindingDimentions(const size_t& batch_size)
{
  for (size_t i = 0; i < model_config_.getSegNum(); i++) {
    sub_input_dims_[i].d[0] = batch_size;
    sub_output_dims_[i].d[0] = batch_size;
    sub_contexts_[i]->setBindingDimensions(0, sub_input_dims_[i]);
    if (!sub_contexts_[i]->allInputDimensionsSpecified()) {
      std::cout << "Failed to set the input dimesion for " << i
                << "-th subcontext" << std::endl;
      return false;
    }
  }
  return true;
}

bool
Instance::run()
{
  for (size_t i = 0; i < model_config_.getSegNum(); i++) {
    bool status = subInfer(i, stream_);
    if (!status) {
      std::cout << "Error when inference " << i << "-th sub model" << std::endl;
      return false;
    }
  }
  return true;
}

bool
Instance::run(const int &left, const int &right)
{
  for (size_t i = left; i <= (unsigned int) right; i++) {
    bool status = subInfer(i, stream_);
    if (!status) {
      std::cout << "Error when inference " << i << "-th sub model" << std::endl;
      return false;
    }
  }
  return true;
}

int
Instance::getSegNum() {
  return model_config_.getSegNum();
}

bool
Instance::infer(const size_t& num_test, const size_t& batch_size)
{
  std::cout << "Setting up input dimension!!!" << std::endl;
  setBindingDimentions(batch_size);
  std::cout << "Warming up!!!" << std::endl;
  bool status = run();
  if (!status) {
    return false;
  }
  CHECK(cudaDeviceSynchronize());
  std::cout << "Testing !!!" << std::endl;
  CHECK(cudaEventRecord(start_, stream_));
  for (size_t i = 0; i < num_test; i++) {
    // NVTX_RANGE(nvtx_, "inference");
    run();
  }
  CHECK(cudaEventRecord(stop_, stream_));
  CHECK(cudaEventSynchronize(stop_));
  float milli_sec = 0;
  CHECK(cudaEventElapsedTime(&milli_sec, start_, stop_));
  std::cout << "Elapsed time: " << milli_sec / num_test << std::endl;
  return true;
}

bool
Instance::infer(const int &left, const int &right, const size_t& num_test, const size_t& batch_size)
{
  for (size_t i = 0; i < num_test; i++) {
    // NVTX_RANGE(nvtx_, "inference");
    run(left, right);
  }
  return true;
}

bool
Instance::multistream_infer(
    const size_t& num_test, const size_t& batch_size, const size_t& stream_cnt)
{
  std::cout << "Not Implemented Yet" << std::endl;
  return false;
}

std::string
Instance::getModelName() {
  return model_config_.getModelName();
}

bool
Instance::exportTrtModel(std::string save_path)
{
  size_t sub_model_cnt = model_config_.getSegNum();
  if (!save_path.empty() && save_path.back() != '/') {
    save_path = save_path + "/";
  }
  for (size_t sub_index = 0; sub_index < sub_model_cnt; sub_index++) {
    std::string sub_model_fname =
        save_path + model_config_.getSegFileName(sub_index) + ".plan";
    auto if_export =
        exportSubTrtModel(sub_engines_[sub_index], sub_model_fname);
    if (!if_export) {
      logger_.log(
          Severity::kINFO,
          ("Failed to export" + std::to_string(sub_index) + "-th submodel")
              .c_str());
    }
  }
  return true;
}

bool
Instance::exportSubTrtModel(
    std::shared_ptr<nvinfer1::ICudaEngine> engine,
    const std::string& sub_model_fname)
{
  std::ofstream ofs(sub_model_fname, std::ios::binary);
  if (!ofs) {
    return false;
  }
  auto host_mem = TRTUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
  ofs.write(reinterpret_cast<const char*>(host_mem->data()), host_mem->size());
  ofs.close();
  logger_.log(Severity::kINFO, (sub_model_fname + " is saved to disk").c_str());

  return true;
}