#include "../include/logger.h"

#include <iostream>

#include "logger.h"
#include "NvInfer.h"

using namespace nvinfer1;

void init_log()
{
    class Logger : public ILogger
    {
        void log(Severity severity, const char *msg) noexcept override
        {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    } logger;

    std::cout << "Logger initialized" << std::endl;
}