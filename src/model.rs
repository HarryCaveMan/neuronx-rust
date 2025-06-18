use crate::ffi;
use crate::data::MemMappedROBuffer;
use crate::error::{
    NrtError,
    NrtResult
};
use std::os::raw::{
    c_void,
    c_char,
    c_uint
};
use std::sync::Arc;
use std::path::PathBuf;

pub struct Model {
    neff_buffer: MemMappedROBuffer,
    handle: *mut ffi::nrt_model_t
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {

    pub fn from_neff_file(path: PathBuf,start_vnc: i32,vnc_count: i32,model: *mut *mut ffi::nrt_model_t) -> NrtResult<Self> {
        let neff_buffer = MemMappedROBuffer::from_file(path)?;
        let mut handle: *mut ffi::nrt_model_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::nrt_load(
                neff_buffer.as_const_ptr(),
                neff_buffer.size(),
                start_vnc,
                vnc_count,
                &mut handle
            )
        } as u32;
        nrt_wrap_status!(Model{handle,neff_buffer},status)
    }

    pub unsafe fn handle(&self) -> *mut ffi::nrt_model_t {self.handle}
}

    impl Drop for Model {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                let status = unsafe { 
                    ffi::nrt_unload(self.handle)
                } as u32;
                self.handle = std::ptr::null_mut();
                // The neff_buffer has its own RAII, only handle cleanup is needed by libnrt
                nrt_wrap_status!((),status).expect("Failed to unload model");
            }
        }
    }