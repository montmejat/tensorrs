#include "../include/logger.h"

std::unique_ptr<LoggerTRT> create_logger(int min_severity)
{
    return std::make_unique<LoggerTRT>(min_severity);
}