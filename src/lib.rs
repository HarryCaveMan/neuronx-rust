#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "neuronx_rs::data"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/tensor.h");
        //TensorInfo
        type TensorInfo;
        fn name(self: &TensorInfo) -> &str;
        fn size(self: &TensorInfo) -> usize;
        fn usage(self: &TensorInfo) -> u32;
        fn data_type(self: &TensorInfo) -> u32;
        fn ndim(self: &TensorInfo) -> u32;
        fn shape_slice(self: &TensorInfo) -> &[u32];
        // IoTensorsResult
        type IoTensorsResult;
        fn success(self: &IoTensorsResult) -> bool;
        // IoTensors
        type IoTensors;
        fn get_input_tensor_info_slice(self: &IoTensors) -> &[TensorInfo];
        fn get_output_tensor_info_slice(self: &IoTensors) -> &[TensorInfo];
        fn bind(
            self: &mut IoTensors,
            name: &str,
            usage: u32,
            buffer: &mut [u8],
        ) -> u32;
    }
    #[namespace = "neuronx_rs::model"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/model.h");
        //ModelResult
        type ModelResult;
        fn success(self: &ModelResult) -> bool;
        //Model
        type Model;
        fn load(path: &str, start_nc: i32, nc_count: i32) -> ModelResult;
        fn get_new_io_tensors(self: &Model) -> IoTensorsResult;
        fn execute(self: Pin<&mut Model>, input: &mut IoTensors) -> u32;
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

/// Re-export the error module for easier access
pub use error::{NrtError, NrtResult};
pub use model::Model;
pub use runtime::*;


#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_init() {
        let init_result = neuronx_init();
        assert_eq!(init_result, 0, "Failed to initialize NeuronX runtime");
    }

    #[test]
    fn test_version() {
        let version_result = neuronx_version();
        assert!(version_result.success(), "Failed to get NeuronX version");
        let version = version_result.value;
        println!("NeuronX Version: {}.{}.{}.{} ({})",
                 version.major(), version.minor(), version.patch(), version.maintenance(), version.detail());
    }
}