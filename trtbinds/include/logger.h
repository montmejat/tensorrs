#pragma once

#include <iostream>
#include <memory>
#include <ctime>
#include <chrono>
#include "NvInfer.h"

using RuntimeTRT = nvinfer1::IRuntime;

class LoggerTRT : public nvinfer1::ILogger
{
private:
    const char *RED = "\033[1;31m";
    const char *GREEN = "\033[1;32m";
    const char *YELLOW = "\033[1;33m";
    const char *BLUE = "\033[1;34m";
    const char *MAGENTA = "\033[1;35m";
    const char *CYAN = "\033[1;36m";
    const char *END_COLOR = "\033[1;0m";

public:
    int m_verbosity;

    LoggerTRT(int verbosity) : m_verbosity(verbosity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if ((int)severity > m_verbosity)
            return;

        std::time_t now = time(0);
        std::tm *local_time = localtime(&now);

        std::cout << CYAN << "[" << local_time->tm_hour << ":"
                  << local_time->tm_min << ":"
                  << local_time->tm_sec << "] " << END_COLOR
                  << GREEN << "[TRT] " << END_COLOR;

        if (severity == Severity::kINTERNAL_ERROR)
            std::cout << RED << "[Internal Error] " << END_COLOR;
        else if (severity == Severity::kERROR)
            std::cout << RED << "[E] " << END_COLOR;
        else if (severity == Severity::kWARNING)
            std::cout << YELLOW << "[W] " << END_COLOR;
        else if (severity == Severity::kINFO)
            std::cout << BLUE << "[I] " << END_COLOR;
        else if (severity == Severity::kVERBOSE)
            std::cout << MAGENTA << "[V] " << END_COLOR;

        std::cout << msg << std::endl;
    }
};

std::unique_ptr<LoggerTRT> create_logger(int min_severity);
std::unique_ptr<RuntimeTRT> create_infer_runtime(const std::unique_ptr<LoggerTRT> &logger);