use std::env;
use std::path::Path;

fn main() {
    // Link paths in LD_LIBRARY_PATH and search for TensorRT and Cuda folders
    let ld_library_path = match env::var("LD_LIBRARY_PATH") {
        Ok(path) => path,
        Err(_) => panic!("LD_LIBRARY_PATH must be set with /path/to/tensorrt/lib!"),
    };

    let mut tensorrt_include_dir = None;
    let mut cuda_include_dir = None;

    let ld_library_paths = ld_library_path.split(":");
    for path in ld_library_paths {
        if path.is_empty() {
            continue;
        }

        // Search for the include directory of TensorRT
        if path.contains("lib") {
            let include_dir = format!("{}{}", path.replace("lib", "include"), "/NvInfer.h");
            if Path::new(include_dir.as_str()).exists() {
                tensorrt_include_dir = Some(path.replace("lib", "include"));
            }
        }

        // Search for the include directory of CUDA
        if path.contains("lib64") {
            let include_dir = format!(
                "{}{}",
                path.replace("lib64", "targets/x86_64-linux/include"),
                "/cuda.h"
            );
            if Path::new(include_dir.as_str()).exists() {
                cuda_include_dir = Some(path.replace("lib64", "targets/x86_64-linux/include"));
            }
        }

        println!("cargo:rustc-link-search=native={}", path);
    }

    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
    println!("cargo:rustc-link-lib=nvparsers");

    cxx_build::bridge("src/lib.rs")
        .file("trtbinds/src/logger.cc")
        .file("trtbinds/src/builder.cc")
        .file("trtbinds/src/parser.cc")
        .file("trtbinds/src/runtime.cc")
        .file("trtbinds/src/engine.cc")
        .flag_if_supported(format!(
            "-I{}", tensorrt_include_dir.expect("TensorRT hasn't been found. 
                    Make sure '/path/to/tensorrt/lib' is included in LD_LIBRARY_PATH.")).as_str())
        .flag_if_supported(format!(
            "-I{}", cuda_include_dir.expect("Cuda hasn't been found. 
                    Make sure '/path/to/cuda/lib64' is included in LD_LIBRARY_PATH.")).as_str())
        .compile("trt");
}
