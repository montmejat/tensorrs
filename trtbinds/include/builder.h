#pragma once

#include <memory>

#include "../include/logger.h"
#include "NvInfer.h"

using BuilderTRT = nvinfer1::IBuilder;
using NetworkDefinitionTRT = nvinfer1::INetworkDefinition;
using BuilderConfigTRT = nvinfer1::IBuilderConfig;
using HostMemoryTRT = nvinfer1::IHostMemory;
using MemoryPoolTypeTRT = nvinfer1::MemoryPoolType;

std::unique_ptr<BuilderTRT> create_builder(const std::unique_ptr<LoggerTRT> &logger);
std::unique_ptr<NetworkDefinitionTRT> create_network(const std::unique_ptr<BuilderTRT> &builder, bool explicit_batch);
std::unique_ptr<BuilderConfigTRT> create_builder_config(const std::unique_ptr<BuilderTRT> &builder);
std::unique_ptr<HostMemoryTRT> build_serialized_network(
    const std::unique_ptr<BuilderTRT> &builder,
    const std::unique_ptr<NetworkDefinitionTRT> &network,
    const std::unique_ptr<BuilderConfigTRT> &config);
void set_memory_pool_limit(const std::unique_ptr<BuilderConfigTRT> &config, int32_t memory_pool_type, uint32_t size);