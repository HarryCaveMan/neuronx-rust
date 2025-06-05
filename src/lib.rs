#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "neuronx_rs::model"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/model.h");
        //ModelResult
        type ModelResult;
        fn success(self: &ModelResult) -> bool;
        //Model
        type Model;
        fn load(path: &str, start_nc: i32, nc_count: i32) -> ModelResult;
        fn bind(self: Pin<&mut Model>, name: &str, usage: u32, buffer: *mut c_void) -> u32;
        fn bind_slice(self: Pin<&mut Model>, name: &str, usage: u32, buffer: &mut [u8]) -> u32;
        fn execute(self: Pin<&mut Model>) -> u32;
    }
    #[namespace = "neuronx_rs::runtime"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/runtime.h");
        //NrtVersionResult
        type NrtVersionResult;
        fn success(self: &NrtVersionResult) -> bool;
        //Uint32Result
        type Uint32Result;
        fn success(self: &Uint32Result) -> bool;
        //NrtVersion
        type NrtVersion;
        fn major(self: &NrtVersion) -> u64;
        fn minor(self: &NrtVersion) -> u64;
        fn patch(self: &NrtVersion) -> u64;
        fn maintenance(self: &NrtVersion) -> u64;
        fn detail(self: &NrtVersion) -> String;
        fn git_hash(self: &NrtVersion) -> String;

        // module methods
        fn neuronx_init() -> u32;
        fn neuronx_close() -> u32;
        fn neuronx_version() -> NrtVersionResult;
        fn neuronx_get_nc_count() -> Uint32Result;
        fn neuronx_get_visible_nc_count() -> Uint32Result;
    }
}

pub mod model;
pub mod runtime;
pub mod error;
pub mod macros;

/// Re-export the error module for easier access
pub use error::{NrtError, NrtResult};
pub use model::Model;
pub use runtime::*;