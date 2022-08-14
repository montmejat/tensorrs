#include "../include/parser.h"
#include <cstring>

std::unique_ptr<ONNXParserTRT> create_parser(
    const std::unique_ptr<NetworkDefinitionTRT> &network,
    const std::unique_ptr<LoggerTRT> &logger)
{
    auto parser = nvonnxparser::createParser(*network, *logger);
    return std::unique_ptr<ONNXParserTRT>(parser);
}

bool parse(
    const std::unique_ptr<ONNXParserTRT> &parser,
    rust::Str onnx_model,
    int verbosity)
{
    // Rust::str are not null-terminated, a trailing \0 needs to be added
    char onnx_model_nterm[onnx_model.length() + 1];
    std::memcpy(onnx_model_nterm, onnx_model.data(), onnx_model.length());
    onnx_model_nterm[onnx_model.length()] = '\0';

    auto parsed_successfully = parser->parseFromFile(onnx_model_nterm, verbosity);
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    return parsed_successfully;
}