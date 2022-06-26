fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("trtbinds/src/logger.cc")
        .flag_if_supported("-I/opt/TensorRT-8.4.1.5/include")
        .flag_if_supported("-I/opt/cuda/targets/x86_64-linux/include")
        .compile("trt");
}