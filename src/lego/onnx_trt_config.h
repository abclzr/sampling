/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: onnx_trt_config.h
 * \brief: configuration for onnx-trt model instance
 * Created Date: Monday, November 2nd 2020, 11:01:27 am
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Thursday, November 12th 2020, 8:40:05 pm
 * Modified By: raphael hao
 */

#pragma once
#include <NvInfer.h>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>


class ModelConfig {
 protected:
  size_t seg_num_{0};
  std::vector<std::string> segment_fnames_;
  std::string data_dir_{};

  nvinfer1::DataType model_dtype_;
  bool fp16_{false};
  bool int8_{false};

  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;

  std::string model_name_;

 public:
  ModelConfig() {}
  /*!
   * \brief Construct a new Model Config object
   *
   * \param seg_num
   * \param data_dir
   * \param severity
   * \param model_dtype
   * \param fp16
   * \param int8
   */
  ModelConfig(
      const size_t& seg_num, const std::string& data_dir,
      nvinfer1::DataType model_dtype = nvinfer1::DataType::kFLOAT,
      bool fp16 = true, bool int8 = false)
      : seg_num_(seg_num), data_dir_(data_dir), model_dtype_(model_dtype),
        fp16_(fp16), int8_(int8)
  {
    segment_fnames_.resize(seg_num_);
  }

  ~ModelConfig() {}

 public:
  void setModelDtype(const nvinfer1::DataType modelDtype)
  {
    model_dtype_ = modelDtype;
  }
  
  void setModelName(std::string str) {
    model_name_ = str;
  }

  std::string getModelName() {
    return model_name_;
  }

  nvinfer1::DataType getModelDtype() const { return model_dtype_; }

  void setFp16(bool fp16) { fp16_ = fp16; }

  void setInt8(bool int8) { int8_ = int8; }

  bool fp16() { return fp16_; }

  bool int8() { return int8_; }

  void setSegNum(const size_t& seg_num)
  {
    seg_num_ = seg_num;
    segment_fnames_.resize(seg_num_);
  }

  size_t getSegNum() { return seg_num_; }

  void setDataDir(const std::string& data_dir) { data_dir_ = data_dir; }

  const std::string getDataDir() const { return data_dir_; }

  const std::string getSegFileName(const size_t& index) const
  {
    assert(index < segment_fnames_.size());
    return segment_fnames_[index];
  }
  void setSegFileName(const size_t& index, const std::string& onnxFilename)
  {
    assert(index < segment_fnames_.size());
    segment_fnames_[index] = onnxFilename;
  }
  void setSegFileName(const size_t& index, const char* onnxFilename)
  {
    assert(index < segment_fnames_.size());
    segment_fnames_[index] = std::string(onnxFilename);
  }

  void destroy() { delete this; }
};