#pragma once

#include <memory>

#include "NvInfer.h"

using CudaEngineTRT = nvinfer1::ICudaEngine;
using RuntimeTRT = nvinfer1::IRuntime;
using HostMemoryTRT = nvinfer1::IHostMemory;

std::unique_ptr<CudaEngineTRT> deserialize_cuda_engine(const std::unique_ptr<RuntimeTRT> &runtime, const std::unique_ptr<HostMemoryTRT> &host_memory);