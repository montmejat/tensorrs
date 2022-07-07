#pragma once

#include <memory>

#include "../include/logger.h"
#include "NvInfer.h"

using BuilderTRT = nvinfer1::IBuilder;
using NetworkDefinitionTRT = nvinfer1::INetworkDefinition;

std::unique_ptr<BuilderTRT> create_builder(const std::unique_ptr<LoggerTRT> &logger);
std::unique_ptr<NetworkDefinitionTRT> create_network(const std::unique_ptr<BuilderTRT> &builder, bool explicit_batch);