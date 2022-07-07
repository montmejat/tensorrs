fn main() {
    let logger = tensorrs::Logger::new();
    let builder = tensorrs::Builder::new(&logger);
    let network = builder.create_network(None);

    let parser = tensorrs::ONNXParser::new(network, &logger);

    println!("Hello, TensorRT!");
}
