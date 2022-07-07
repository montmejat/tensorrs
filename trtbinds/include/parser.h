#pragma once

#include <memory>

#include "../include/builder.h"
#include "../include/logger.h"
#include "NvOnnxParser.h"

using ONNXParserTRT = nvonnxparser::IParser;

std::unique_ptr<ONNXParserTRT> create_parser(std::unique_ptr<NetworkDefinitionTRT> network, const std::unique_ptr<LoggerTRT> &logger);