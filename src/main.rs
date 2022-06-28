fn main() {
    println!("Hello, TensorRT!");

    let logger = tensorrs::Logger::new();
    let _builder = tensorrs::Builder::new(logger);
}