use crate::ffi;
use crate::error::{
    NrtError,
    NrtResult
};
use std::path::PathBuf;
use std::os::raw::{
    c_void,
    c_int,
    c_char
};
use std::ptr::null_mut;
use std::fs::File;
use std::sync::Arc;
use memmap2::{Mmap, MmapOptions};
use ndarray::ArrayD;
use log::{
    info,
    error,
    debug,
    warn
};

pub struct MemMappedROBuffer {
    buffer: Mmap
}

impl MemMappedROBuffer {
    pub fn from_file(path: PathBuf) -> NrtResult<Self> {
        info!("Creating a ReadOnly Memory-Mapped buffer from: {}", path.display());
        match File::open(&path) {
            Ok(file) => {
                match unsafe { MmapOptions::new().map(&file) } {
                    Ok(buffer) => Ok(MemMappedROBuffer { buffer }),
                    Err(e) => Err({
                        error!("Failed to memmap file {}: {}", path.display(), e);
                        NrtError::NRT_FAILURE
                    })
                }
            },
            Err(e) => Err({
                error!("Failed to open file {}: {}", path.display(), e);
                NrtError::NRT_FAILURE
            })
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
    pub fn as_const_ptr(&self) -> *const c_void {
        self.buffer.as_ptr() as *const c_void
    }
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

enum_from_primitive! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NrtTensorPlacement {
        NRT_TENSOR_PLACEMENT_DEVICE = 0,
        NRT_TENSOR_PLACEMENT_HOST = 1,
        NRT_TENSOR_PLACEMENT_VIRTUAL = 2
    }
}
impl From<u32> for NrtTensorPlacement {
    fn from(t: u32) -> NrtResult<Self> {
        NrtTensorPlacement::from_u32(t).ok_or(NrtError::NRT_INVALID)
    }
}
impl From<ffi::nrt_tensor_placement_t> for NrtTensorPlacement {
    fn from(t: ffi::nrt_tensor_placement_t) -> NrtResult<Self> {
        NrtTensorPlacement::from(t as u32)?
    }
}

enum_from_primitive! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NrtTensorUsage {
        NRT_TENSOR_USAGE_INPUT = 0,
        NRT_TENSOR_USAGE_OUTPUT = 1
    }
}
impl From<u32> for NrtTensorUsage {
    fn from(t: u32) -> NrtResult<Self> {
        NrtTensorUsage::from_u32(t).ok_or(NrtError::NRT_INVALID)
    }
}
impl From<ffi::nrt_tensor_usage_t> for NrtTensorUsage {
    fn from(t: ffi::nrt_tensor_usage_t) -> NrtResult<Self> {
        NrtTensorUsage::from(t as u32)?
    }
}



enum_from_primitive! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum NrtTensorType {
        NNRT_DTYPE_UNKNOWN = 0,
        NRT_DTYPE_FLOAT32 = 1,
        NRT_DTYPE_FLOAT64 = 2,
        NRT_DTYPE_FLOAT16 = 3,
        NRT_DTYPE_BFLOAT16 = 4,
        NRT_DTYPE_INT8 = 5,
        NRT_DTYPE_UINT8 = 6,
        NRT_DTYPE_INT16 = 7,
        NRT_DTYPE_UINT16 = 8,
        NRT_DTYPE_INT32 = 9,
        NRT_DTYPE_UINT32 = 10,
        NRT_DTYPE_INT64 = 11,
        NRT_DTYPE_UINT64 = 12
    }
}
impl From<u32> for NrtTensorType {
    fn from(t: u32) -> NrtResult<Self> {
        NrtTensorType::from_u32(t).ok_or(NrtError::NRT_INVALID)
    }
}
impl From<ffi::nrt_tensor_type_t> for NrtTensorType {
    fn from(t: ffi::nrt_tensor_type_t) -> NrtResult<Self> {
        NrtTensorType::from(t as u32)?
    }
}

pub struct NrtTensorInfo {
    pub name: String,
    pub size: usize,
    pub usage: NrtTensorUsage,
    pub dtype: NrtTensorType
}

pub struct NrtTensor<'model> {
    handle: *mut ffi::nrt_tensor_t,
    placement: NrtTensorPlacement,
    info: &'model ffi::nrt_tensor_info_t
    has_owned_storage: bool
}

impl<'model> NrtTensor<'model> {
    pub (crate) fn from_tensor_info_t(
        tensor_placement: NrtTensorPlacement,
        nc_id: u32,
        info: &'model ffi::nrt_tensor_info_t,
    ) -> NrtResult<Self> {
        let mut handle = null_mut();
        let status = unsafe {
            nrt_tensor_allocate(
                tensor_placement as c_int
                nc_id as c_int,
                info.size,
                info.name,
                &mut handle
            )
        } as u32;
        has_owned_storage = true;
        if status == 0 {t = NrtTensor{handle,tensor_placement,info,has_owned_storage};}
        nrt_result!(t,status)
    }

    pub unsafe fn bind<T>(&mut self, data: &mut ArrayD<T>) -> NrtResult<()> {
        if self.has_owned_storage {
            println!("Warning: Binding to a tensor with owned removes owned storage cleanup");
        }
        let status = unsafe {
            ffi::nrt_tensor_bind(
                self.handle as *mut c_void,
                data.as_mut_ptr() as *mut c_void,
                data.len() * std::mem::size_of::<T>()
            )
        };
        nrt_result!((), status)
    }
}

impl<'model> Drop for NrtTensor<'model> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let status = unsafe { ffi::nrt_tensor_free(self.handle) } as u32;
            nrt_result!((), status).expect("Failed to free tensor");
            self.handle = null_mut();
        }
    }
}