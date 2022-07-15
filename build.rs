fn main() {
    println!("cargo:rustc-link-search=native=/opt/TensorRT-8.4.1.5/lib");
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
    println!("cargo:rustc-link-lib=nvparsers");

    cxx_build::bridge("src/lib.rs")
        .file("trtbinds/src/logger.cc")
        .file("trtbinds/src/builder.cc")
        .file("trtbinds/src/parser.cc")
        .flag_if_supported("-I/opt/TensorRT-8.4.1.5/include")
        .flag_if_supported("-I/opt/cuda/targets/x86_64-linux/include")
        .compile("trt");
}
