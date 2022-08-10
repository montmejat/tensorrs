# Requirements

Make sure you have `/path/to/tensorrt/lib` and `/path/to/cuda/lib64` in your `LD_LIBRARY_PATH` environment variable.

# TensorRT bindings for Rust

Use TensorRT like you already do. Here's what works for now:

```rust
fn main() {
    // Setup logger
    let logger = tensorrs::logging::Logger::new();

    // Create builder
    let builder = tensorrs::Builder::new(&logger);
    let network = builder.create_network(Some(true));

    // Parse ONNX model
    let parser = tensorrs::OnnxParser::new(&network, &logger);
    parser.parse(
        "path/to/my/model.onnx",
        tensorrs::logging::Sererity::Info,
    );

    // Build engine
    let builder_config = builder.create_config();
    builder_config.set_memory_pool_limit(tensorrs::MemoryPoolType::Workspace, 5_000_000);
    let engine = builder.build_serialized_network(&network, &builder_config);
}
```