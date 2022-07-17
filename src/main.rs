fn main() {
    // Setup logger
    let logger = tensorrs::logging::Logger::new();

    // Create builder
    let builder = tensorrs::Builder::new(&logger);
    let network = builder.create_network(Some(true));

    // Parse ONNX model
    let parser = tensorrs::OnnxParser::new(&network, &logger);
    parser.parse(
        "/home/aurelien/Projects/Personal/Models/bvlcalexnet-12.onnx",
        tensorrs::logging::Sererity::Info,
    );

    // Build engine
    let builder_config = builder.create_config();
    builder_config.set_memory_pool_limit(tensorrs::MemoryPoolType::Workspace, 5_000_000);
    let engine = builder.build_serialized_network(&network, &builder_config);
}
