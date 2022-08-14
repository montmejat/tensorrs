#pragma once

#include <memory>
#include <string>

#include "../include/builder.h"
#include "../include/logger.h"
#include "rust/cxx.h"
#include "NvOnnxParser.h"

using ONNXParserTRT = nvonnxparser::IParser;

std::unique_ptr<ONNXParserTRT> create_parser(const std::unique_ptr<NetworkDefinitionTRT> &network, const std::unique_ptr<LoggerTRT> &logger);
bool parse(const std::unique_ptr<ONNXParserTRT> &parser, rust::Str onnx_model_file, int verbosity);