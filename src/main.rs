fn main() {
    let logger = tensorrs::logging::Logger::new();
    let builder = tensorrs::Builder::new(&logger);
    let network = builder.create_network(Some(true));

    let parser = tensorrs::OnnxParser::new(&network, &logger);
    parser.parse(
        "/home/aurelien/Projects/Personal/Models/resnet50-v1-12.onnx",
        tensorrs::logging::Sererity::InternalError,
    );
}
