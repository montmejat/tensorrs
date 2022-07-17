#include "../include/builder.h"
#include <iostream>

std::unique_ptr<BuilderTRT> create_builder(const std::unique_ptr<LoggerTRT> &logger)
{
    auto builder = nvinfer1::createInferBuilder(*logger);
    return std::unique_ptr<BuilderTRT>(builder);
}

// Since explicit batch is the only option available in NetworkDefinitionCreationFlag
// we simplify the network creation with just an argument.
std::unique_ptr<NetworkDefinitionTRT> create_network(const std::unique_ptr<BuilderTRT> &builder, bool explicit_batch)
{
    uint32_t flag = 0U;
    if (explicit_batch)
        flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    nvinfer1::INetworkDefinition *network = builder.get()->createNetworkV2(flag);
    return std::unique_ptr<NetworkDefinitionTRT>(network);
}

std::unique_ptr<BuilderConfigTRT> create_builder_config(const std::unique_ptr<BuilderTRT> &builder)
{
    auto config = builder.get()->createBuilderConfig();
    return std::unique_ptr<BuilderConfigTRT>(config);
}

std::unique_ptr<HostMemoryTRT> build_serialized_network(
    const std::unique_ptr<BuilderTRT> &builder,
    const std::unique_ptr<NetworkDefinitionTRT> &network,
    const std::unique_ptr<BuilderConfigTRT> &config)
{
    auto serialized_network = builder.get()->buildSerializedNetwork(*network, *config);
    return std::unique_ptr<HostMemoryTRT>(serialized_network);
}

void set_memory_pool_limit(const std::unique_ptr<BuilderConfigTRT> &config, int32_t memory_pool_type, uint32_t size)
{
    config.get()->setMemoryPoolLimit((nvinfer1::MemoryPoolType)memory_pool_type, size);
}