#include "../include/parser.h"

std::unique_ptr<ONNXParserTRT> create_parser(std::unique_ptr<NetworkDefinitionTRT> network, const std::unique_ptr<LoggerTRT> &logger)
{
    auto parser = nvonnxparser::createParser(*network, *logger);
    return std::unique_ptr<ONNXParserTRT>(parser);
}