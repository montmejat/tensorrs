#pragma once

#include <memory>

#include "../include/logger.h"
#include "NvInfer.h"

struct BuilderTRT
{
    nvinfer1::IBuilder* builder;
};

std::unique_ptr<BuilderTRT> create_builder(std::unique_ptr<LoggerTRT> logger);