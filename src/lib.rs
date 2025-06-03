#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "neuron_rs::model"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/model.h");
        //Model
        type Model;
        fn load(path: &str, start_nc: i32. nc_count: i32) -> Result<UniquePtr<Model>>;
        fn bind_slice(self: Pin<&mut Model>, usage: u32, buffer: &mut [u8]) -> u32;
        fn execute(self: self: Pin<&mut Model>);
    }
    #[namespace = "neuron_rs::runtime"]
    unsafe extern "C++" {
        include!("neuronx-rust/cxx/include/runtime.h");
        //Nrtversion
        type Nrtversion;
        fn major(self: &Version) -> u64;
        fn minor(self: &Version) -> u64;
        fn patch(self: &Version) -> u64;
        fn maintenance(self: &Version) -> u64;
        fn detail(self: &Version) -> String;
        fn git_hash(self: &Version) -> String;

        // module methods
        fn neuronx_init() -> u32;
        fn neuronx_close() -> u32;
        fn neuronx_version() -> Result<UniquePtr<Nrtversion>>;
        fn get_nc_count() -> Result<u32>;
        fn get_visible_nc_count() -> Result<u32>;
    }
}