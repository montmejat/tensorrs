#include "../include/builder.h"
#include <iostream>

std::unique_ptr<BuilderTRT> create_builder(std::unique_ptr<LoggerTRT> logger)
{
    std::cout << "Builder initialized" << std::endl;
    auto builder = nvinfer1::createInferBuilder(*logger);

    return std::unique_ptr<BuilderTRT>(new BuilderTRT{builder});
}