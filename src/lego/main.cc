/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: main.cc
 * \brief: test entry
 * Created Date: Sunday, November 1st 2020, 12:07:59 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Wednesday, November 18th 2020, 11:01:26 am
 * Modified By: raphael hao
 */

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include "src/lego/onnx_trt.h"
#include "src/lego/record.h"

#include <iostream>
#include <fstream>

ModelConfig
getInstConfig(const rapidjson::Document& config_doc)
{
  assert(config_doc["seg_files"].IsArray());
  const rapidjson::Value& model_files = config_doc["seg_files"];
  assert(model_files.Size() == config_doc["seg_num"].GetUint());
  ModelConfig model_config{
      config_doc["seg_num"].GetUint(), config_doc["dir"].GetString(),
      nvinfer1::DataType::kFLOAT};
  model_config.setModelName(config_doc["name"].GetString());
  for (rapidjson::SizeType i = 0; i < model_files.Size(); i++) {
    model_config.setSegFileName(i, model_files[i].GetString());
  }
  return model_config;
}

Instance* getInstance(char const* argvi) {
  std::string config_path = argvi;
  std::cout << config_path << std::endl;
  FILE* config_fp = fopen(config_path.c_str(), "r");
  char read_buffer[65536];
  rapidjson::FileReadStream config_fs(
      config_fp, read_buffer, sizeof(read_buffer));
  rapidjson::Document config_doc;
  config_doc.ParseStream(config_fs);
  ModelConfig model_config = getInstConfig(config_doc);
  Instance* inst = new Instance(
      model_config, config_doc["min_bs"].GetUint(),
      config_doc["opt_bs"].GetUint(), config_doc["max_bs"].GetUint(),
      nvinfer1::ILogger::Severity::kINFO);
  inst->build();
  return inst;
}

std::vector<Instance *> instList;
int numTest = 100;
int numSample = 100;
int batchSize = 1;
cudaEvent_t start_;
cudaEvent_t stop_;

float runTwoInstance(Instance *inst1, const int &l1, const int &r1, Instance *inst2, const int &l2, const int &r2) {
  std::cout<<inst1->getModelName() << " "<<l1<<" "<<r1<<" "<<inst2->getModelName() <<" "<<l2<<" "<<r2<<std::endl;
  CHECK(cudaEventRecord(start_));
  for (int i = 0; i < numTest; ++i) {
    inst1->run(l1, r1);
    inst2->run(l2, r2);
    cudaStreamSynchronize(nullptr);
  }
  CHECK(cudaEventRecord(stop_));
  CHECK(cudaEventSynchronize(stop_));
  float milli_sec = 0;
  CHECK(cudaEventElapsedTime(&milli_sec, start_, stop_));
  return milli_sec / numTest;
}

float runOneInstance(Instance *inst1, const int &l1, const int &r1) {
  std::cout<<inst1->getModelName() << " "<<l1<<" "<<r1<<std::endl;
  CHECK(cudaEventRecord(start_));
  for (int i = 0; i < numTest; ++i) {
    inst1->run(l1, r1);
    cudaStreamSynchronize(nullptr);
  }
  CHECK(cudaEventRecord(stop_));
  CHECK(cudaEventSynchronize(stop_));
  float milli_sec = 0;
  CHECK(cudaEventElapsedTime(&milli_sec, start_, stop_));
  return milli_sec / numTest;
}

Record makeRecord(Instance *inst1, std::pair<int, int> interval1, int bs1, Instance *inst2, std::pair<int, int> interval2, int bs2) {
  inst1->setBindingDimentions(bs1);
  inst2->setBindingDimentions(bs2);
  float result1 = runOneInstance(inst1, interval1.first, interval1.second);
  float result2 = runOneInstance(inst2, interval2.first, interval2.second);
  float result = runTwoInstance(inst1, interval1.first, interval1.second, inst2, interval2.first, interval2.second);
  Record record(inst1->getModelName(), interval1.first, interval1.second, bs1, inst2->getModelName(), interval2.first, interval2.second, bs2, result1, result2, result);
  return record;
}

std::pair<int, int> pickTwo(int n) {
  int n1 = rand() % (n-4);
  int n2 = n1 + 4 + rand() % (n - n1 - 4);
  if (n1 > n2) std::swap(n1, n2);
  return std::make_pair(n1, n2);
}

int
main(int argc, char const* argv[])
{
  srand(time(0));
  NVTX_INITIALIZE;
  instList.clear();
  numTest = std::atoi(argv[1]);
  numSample = std::atoi(argv[2]);
  std::ofstream outputFile;
  outputFile.open(argv[3], std::ios::out | std::ios::app);
  for (int i = 4; i < argc; ++i) {
    Instance *inst = getInstance(argv[i]);
    instList.push_back(inst);
  }
  CHECK(cudaEventCreate(&start_));
  CHECK(cudaEventCreate(&stop_));  

  for (unsigned int i = 0; i < instList.size(); ++i) {
    Instance *inst1 = instList[i];
    for (unsigned int j = i + 1; j < instList.size(); ++j) {
      Instance *inst2 = instList[j];
      for (int k = 0; k < numSample; ++k) {
        std::pair<int, int> interval1 = pickTwo(inst1->getSegNum());
        std::pair<int, int> interval2 = pickTwo(inst2->getSegNum());
        for (int bs1 = 4; bs1 <= 16; bs1 = bs1 * 2)
          for (int bs2 = 4; bs2 <= 16; bs2 = bs2 * 2)
            outputFile<<makeRecord(inst1, interval1, bs1, inst2, interval2, bs2).toString()<<std::endl;
      }
    }
  }

  outputFile.close();
  for (auto inst : instList) {
    delete inst;
  }
  return 0;
}