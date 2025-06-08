use std::env;
use std::path::PathBuf;

fn find_dir(
    env_key: &'static str,
    candidates: Vec<&'static str>,
    file_to_find: &'static str,
) -> Option<PathBuf> {
    match env::var_os(env_key) {
        Some(val) => Some(PathBuf::from(&val)),
        _ => {
            for candidate in candidates {
                let path = PathBuf::from(candidate);
                let file_path = path.join(file_to_find);
                if file_path.exists() {
                    return Some(path);
                }
            }
            None
        }
    }
}

fn main() {
    let nrt_include_dir = find_dir(
        "LIBNRT_INCLUDE_PATH",
        vec!["/opt/neuron/include", "/usr/local/neuron/include"],
        "nrt/nrt.h",
    ).expect("Could not find Neuron runtime include path");

    let nrt_library_dir = find_dir(
        "LIBNRT_LIB_PATH",
        vec!["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"],
        "libnrt.so",
    ).expect("Could not find Neuron runtime library path");

    let include_files = vec![
        "cxx/include/model.h",
        "cxx/include/runtime.h",
        "cxx/include/tensor.h"
    ];
    let cpp_files = vec![
        "cxx/src/model.cc",
        "cxx/src/tensor.cc"
    ];
    let rust_files = vec![
        "src/lib.rs",
    ];

    cxx_build::bridges(&rust_files)
        .include(nrt_include_dir)
        .include("cxx/include")
        .files(&cpp_files)
        .define("FMT_HEADER_ONLY", None)
        .flag_if_supported("-std=c++20")
        .compile("neuronx-rust-cxxbridge");

    println!("cargo:rustc-link-search={}", nrt_library_dir.to_string_lossy());

    // Mutable in case I want to add more optional libraries later (IE libtorch, etc.)
    let mut libraries = vec![
        "nrt"
    ];

    for library in libraries {
        println!("cargo:rustc-link-lib={}", library);
    }

    for file in include_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    for file in cpp_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    for file in rust_files {
        println!("cargo:rerun-if-changed={}", file);
    }
}