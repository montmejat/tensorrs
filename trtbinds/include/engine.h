#pragma once

#include <memory>

#include "../include/logger.h"
#include "NvInfer.h"

using CudaEngineTRT = nvinfer1::ICudaEngine;
using ExecutionContextTRT = nvinfer1::IExecutionContext;

std::unique_ptr<ExecutionContextTRT> create_execution_context(const std::unique_ptr<CudaEngineTRT> &engine);