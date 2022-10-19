#include "../include/logger.h"

std::unique_ptr<LoggerTRT> create_logger(int min_severity)
{
    return std::make_unique<LoggerTRT>(min_severity);
}

std::unique_ptr<RuntimeTRT> create_infer_runtime(const std::unique_ptr<LoggerTRT> &logger)
{
    auto runtime = nvinfer1::createInferRuntime(*logger);
    return std::unique_ptr<RuntimeTRT>(runtime);
}