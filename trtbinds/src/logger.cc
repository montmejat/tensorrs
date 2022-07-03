#include "../include/logger.h"

std::unique_ptr<LoggerTRT> create_logger()
{
    return std::unique_ptr<LoggerTRT>(new LoggerTRT());
}