fn main() {
    println!("cargo:rustc-link-search=native=/opt/TensorRT-8.4.1.5/lib");
    println!("cargo:rustc-link-lib=nvinfer");

    // Extra libs that might come in handy later
    // println!("cargo:rustc-link-lib=nvonnxparser");
    // println!("cargo:rustc-link-lib=nvparsers");

    cxx_build::bridge("src/lib.rs")
        .file("trtbinds/src/logger.cc")
        .file("trtbinds/src/builder.cc")
        .flag_if_supported("-I/opt/TensorRT-8.4.1.5/include")
        .flag_if_supported("-I/opt/cuda/targets/x86_64-linux/include")
        .compile("trt");
}
