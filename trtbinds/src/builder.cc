#include "../include/builder.h"
#include <iostream>

std::unique_ptr<BuilderTRT> create_builder(std::unique_ptr<LoggerTRT> logger)
{
    auto builder = nvinfer1::createInferBuilder(*logger);
    return std::unique_ptr<BuilderTRT>(builder);
}

// Since explicit batch is the only option available in NetworkDefinitionCreationFlag
// we simplify the network creation with just an argument.
std::unique_ptr<NetworkDefinitionTRT> create_network(const std::unique_ptr<BuilderTRT> &builder, bool explicit_batch)
{
    uint32_t flag;
    if (explicit_batch)
        flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    else
        flag = 0U;
    
    nvinfer1::INetworkDefinition *network = builder.get()->createNetworkV2(flag);
    return std::unique_ptr<NetworkDefinitionTRT>(network);
}