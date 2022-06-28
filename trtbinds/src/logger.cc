#include "../include/logger.h"

std::unique_ptr<LoggerTRT> create_logger()
{
    std::cout << "Logger initialized" << std::endl;
    return std::unique_ptr<LoggerTRT>(new LoggerTRT());
}