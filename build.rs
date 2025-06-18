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

    // Find the library directory
    let nrt_lib_dir = find_dir(
        "LIBNRT_LIB_PATH",
        vec!["/opt/aws/neuron/lib", "/usr/local/neuron/lib"],
        "libnrt.so",
    ).expect("Could not find Neuron runtime library path");

    // Tell cargo to link against nrt
    println!("cargo:rustc-link-search={}", nrt_lib_dir.display());
    println!("cargo:rustc-link-lib=nrt");
    #[cfg(feature = "ndl")]
    println!("cargo:rustc-link-lib=nds");

    // In build.rs
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", nrt_lib_dir.display());

    let nrt_include_dir = find_dir(
        "LIBNRT_INCLUDE_PATH",
        vec!["/opt/aws/neuron/include", "/usr/local/neuron/include"],
        "cxx/include",
    ).expect("Could not find Neuron runtime include path");

    let mut builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", nrt_include_dir.display()))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++20")
        .header(nrt_include_dir.join("nrt/nrt_experimental.h").to_string_lossy())
        .header(nrt_include_dir.join("nrt/nrt.h").to_string_lossy())
        .header(nrt_include_dir.join("nrt/nrt_status.h").to_string_lossy())
        .header(nrt_include_dir.join("nrt/nrt_profile.h").to_string_lossy())
        .header(nrt_include_dir.join("nrt/nrt_version.h").to_string_lossy())
        .header(nrt_include_dir.join("nrt/nec.h").to_string_lossy());

    if cfg!(feature = "ndl") {
        builder = builder.header(nrt_include_dir.join("nrt/nds/neuron_ds.h").to_string_lossy());
    }

    let bindings = builder
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from("./src");
    bindings
        .write_to_file(out_path.join("ffi.rs"))
        .expect("Couldn't write bindings!");

    // Rebuild if the libs change
    println!("cargo:rerun-if-changed={}", nrt_lib_dir.join("libnrt.so").display());
    //rebuild if the headers change
    println!("cargo:rerun-if-changed={}", nrt_include_dir.join("nrt/nrt_experimental.h").display());
    println!("cargo:rerun-if-changed={}", nrt_include_dir.join("nrt/nrt.h").display());
    println!("cargo:rerun-if-changed={}", nrt_include_dir.join("nrt/nrt_status.h").display());
    println!("cargo:rerun-if-changed={}", nrt_include_dir.join("nrt/nrt_profile.h").display());
    println!("cargo:rerun-if-changed={}", nrt_include_dir.join("nrt/nec.h").display());
    println!("cargo:rerun-if-changed={}", nrt_include_dir.join("nrt/nds/neuron_ds.h").display());
    // Rebuild if the build script itself changes
    println!("cargo:rerun-if-changed=build.rs");
}