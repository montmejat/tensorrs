#include "../include/parser.h"

std::unique_ptr<ONNXParserTRT> create_parser(const std::unique_ptr<NetworkDefinitionTRT> &network, const std::unique_ptr<LoggerTRT> &logger)
{
    auto parser = nvonnxparser::createParser(*network, *logger);
    return std::unique_ptr<ONNXParserTRT>(parser);
}

bool parse(const std::unique_ptr<ONNXParserTRT> &parser, rust::Str onnxModelFile, int verbosity)
{
    auto parsedSuccessfully = parser->parseFromFile(onnxModelFile.data(), verbosity);

    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    return parsedSuccessfully;
}