#include "../include/logger.h"
#include "NvInfer.h"

#include <iostream>
#include <memory>

std::unique_ptr<Logger> create_logger()
{
    std::cout << "Logger initialized" << std::endl;
    return std::unique_ptr<Logger>(new Logger());
}