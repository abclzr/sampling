/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: utils.h
 * \brief: utils for lego
 * Created Date: Sunday, November 1st 2020, 4:35:59 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Wednesday, November 18th 2020, 11:00:54 am
 * Modified By: raphael hao
 */

#pragma once

#include <nvtx3/nvToolsExt.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

struct TRTDeleter {
  template <typename T>
  void operator()(T* obj) const
  {
    if (obj) {
      obj->destroy();
    }
  }
};

inline int64_t
volume(const nvinfer1::Dims& d)
{
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int
getElementSize(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

template <typename A, typename B>
inline A
divUp(A x, B n)
{
  return (x + n - 1) / n;
}

constexpr long double operator"" _GiB(long double val)
{
  return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val)
{
  return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val)
{
  return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
  return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
  return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
  return val * (1 << 10);
}

inline std::string
locateFile(const std::string& filepathSuffix, const std::string& directory)
{
  const int MAX_DEPTH{10};
  bool found{false};
  std::string filepath;

  if (!directory.empty() && directory.back() != '/') {
    filepath = directory + "/" + filepathSuffix;
  } else
    filepath = directory + filepathSuffix;

  for (int i = 0; i < MAX_DEPTH && !found; i++) {
    std::ifstream checkFile(filepath);
    found = checkFile.is_open();
    if (found)
      break;
    filepath = "../" + filepath;  // Try again in parent dir
  }

  if (filepath.empty()) {
    std::cout << "Could not find " << filepathSuffix
              << " in data directory:\n\t" << directory << std::endl;
    std::cout << "&&&& FAILED" << std::endl;
    exit(EXIT_FAILURE);
  }
  return filepath;
}

class Logger : public nvinfer1::ILogger {
  using Severity = nvinfer1::ILogger::Severity;

 public:
  Logger(Severity severity = Severity::kWARNING) : severity_(severity) {}

  void log(Severity severity, const char* msg) override
  {
    if (severity <= severity_) {
      std::cout << msg << std::endl;
    }
  }

  void setSeverity(Severity severity) { severity_ = severity; }

  Severity getSeverity() { return severity_; }

 private:
  Severity severity_;
};

#define CHECK(status)                                    \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cout << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

#define CHECK_RETURN_W_MSG(status, val, errMsg)                        \
  do {                                                                 \
    if (!(status)) {                                                   \
      std::cout << errMsg << " Error in " << __FILE__ << ", function " \
                << FN_NAME << "(), line " << __LINE__ << std::endl;    \
      return val;                                                      \
    }                                                                  \
  } while (0)

#define ASSERT(condition)                                            \
  do {                                                               \
    if (!(condition)) {                                              \
      std::cout << "Assertion failure: " << #condition << std::endl; \
      abort();                                                       \
    }                                                                \
  } while (0)

#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

class NvtxRange {
 public:
  explicit NvtxRange(const std::string& label) : label_(label)
  {
    depth_ = nvtxRangePushA(label_.c_str());
    std::cout << "Starting the NVTX range: '" << label_ << "' @ depth: '"
              << depth_ << "'" << std::endl;
    if (depth_ < 0) {
      std::cout << "Unable to start NVTX range '" << label_ << "'" << std::endl;
    }
  }

  ~NvtxRange()
  {
    if (depth_ >= 0) {
      nvtxRangePop();
      std::cout << "Ending the NVTX range: '" << label_ << "'" << std::endl;
    }
  }

 private:
  int depth_;
  std::string label_;
};

#define NVTX_INITIALIZE nvtxInitialize(nullptr)
#define NVTX_RANGE(V, L) NvtxRange V(L)
#define NVTX_MARKER(L) nvtxMarkA(L)