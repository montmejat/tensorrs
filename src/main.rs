fn main() {
    let logger = tensorrs::Logger::new();
    let builder = tensorrs::Builder::new(logger);
    let _network = builder.create_network(None);

    println!("Hello, TensorRT!");
}
