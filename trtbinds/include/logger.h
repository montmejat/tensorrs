#pragma once

#include <iostream>
#include <memory>

#include "NvInfer.h"

class LoggerTRT : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

std::unique_ptr<LoggerTRT> create_logger();