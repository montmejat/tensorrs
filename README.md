# TensorRT bindings for Rust

## Example

Use TensorRT like you already do. Here's what works for now:

```rust
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
    let serialized_model = builder.build_serialized_network(&network, &builder_config);

    // Deserialize the plan
    let runtime = logger.create_infer_runtime();
    let _engine = runtime.deserialize_cuda_engine(serialized_model);
}
```

## Requirements

Make sure you have `/path/to/tensorrt/lib` and `/path/to/cuda/lib64` in your `LD_LIBRARY_PATH` environment variable, for example:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-8.4.2.4/lib:/usr/local/cuda-11.6/lib64
```
