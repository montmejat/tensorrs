#include "../include/builder.h"
#include <iostream>

std::unique_ptr<BuilderTRT> create_builder(std::unique_ptr<LoggerTRT> logger)
{
    auto builder = nvinfer1::createInferBuilder(*logger);
    return std::unique_ptr<BuilderTRT>(builder);
}

std::unique_ptr<NetworkDefinitionTRT> create_network(const std::unique_ptr<BuilderTRT> &builder)
{
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder.get()->createNetworkV2(flag);
    return std::unique_ptr<NetworkDefinitionTRT>(network);
}