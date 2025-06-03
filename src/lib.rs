#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "neuron_rs::model"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/model.h");
        type Model;
        fn load(path: &str, start_nc: i32. nc_count: i32) -> UniquePtr<Model>;
        fn bind_slice(self: Pin<&mut Model>, usage: u32, buffer: &mut [u8]) -> u32;
        fn execute(self: self: Pin<&mut Model>);
    }
    #[namespace = "neuron_rs::runtime"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/runtime.h");
        type Runtime;
        fn neuronx_init() -> u32;
        fn neuronx_close() -> u32;
    }
}