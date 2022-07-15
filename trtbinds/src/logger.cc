#include "../include/logger.h"

std::unique_ptr<LoggerTRT> create_logger()
{
    return std::make_unique<LoggerTRT>();
}