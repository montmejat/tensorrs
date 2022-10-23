use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!(
            "You need to specify the path to the onnx model.\n\
             You can do it like this: cargo run /path/to/model.onnx"
        );
        process::exit(1)
    }

    let onnx_model_path = &args[1];

    // Setup logger
    let logger = tensorrs::logging::Logger::new(None);

    // Create builder
    let builder = tensorrs::Builder::new(&logger);
    let network = builder.create_network(Some(true));

    // Parse ONNX model
    let parser = tensorrs::OnnxParser::new(&network, &logger);
    parser.parse(&onnx_model_path, tensorrs::logging::Sererity::Info);

    // Build serialized model for this platform
    let builder_config = builder.create_config();
    builder_config.set_memory_pool_limit(tensorrs::MemoryPoolType::Workspace, 5_000_000);
    
    // TODO: This model can be saved to disk and re-used later
    let serialized_model = builder.build_serialized_network(&network, &builder_config);

    // Deserialize the plan
    let runtime = logger.create_infer_runtime();
    let _engine = runtime.deserialize_cuda_engine(serialized_model);
}
