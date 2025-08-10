use crate::ffi;
use crate::error::{
    NrtError,
    NrtResult
};
use std::os::raw::{
    c_void,
    c_int,
    c_char
};
use std::ptr::null_mut;
use std::ffi::{
    CString,
    CStr
};
use ndarray::ArrayD;
use log::{
    info,
    error,
    debug,
    warn
};

enum_from_primitive! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum TensorPlacement {
        NRT_TENSOR_PLACEMENT_DEVICE = 0,
        NRT_TENSOR_PLACEMENT_HOST = 1,
        NRT_TENSOR_PLACEMENT_VIRTUAL = 2
    }
}
impl From<u32> for TensorPlacement {
    fn from(t: u32) -> NrtResult<Self> {
        TensorPlacement::from_u32(t).ok_or(NrtError::NRT_INVALID)
    }
}
impl From<ffi::nrt_tensor_placement_t> for TensorPlacement {
    fn from(t: ffi::nrt_tensor_placement_t) -> NrtResult<Self> {
        TensorPlacement::from(t as u32)?
    }
}

enum_from_primitive! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum TensorUsage {
        NRT_TENSOR_USAGE_INPUT = 0,
        NRT_TENSOR_USAGE_OUTPUT = 1
    }
}
impl From<u32> for TensorUsage {
    fn from(t: u32) -> NrtResult<Self> {
        TensorUsage::from_u32(t).ok_or(NrtError::NRT_INVALID)
    }
}
impl From<ffi::nrt_tensor_usage_t> for TensorUsage {
    fn from(t: ffi::nrt_tensor_usage_t) -> NrtResult<Self> {
        TensorUsage::from(t as u32)?
    }
}

enum_from_primitive! {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum TensorType {
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
impl From<u32> for TensorType {
    fn from(t: u32) -> NrtResult<Self> {
        TensorType::from_u32(t).ok_or(NrtError::NRT_INVALID)
    }
}
impl From<ffi::nrt_tensor_type_t> for TensorType {
    fn from(t: ffi::nrt_tensor_type_t) -> NrtResult<Self> {
        TensorType::from(t as u32)?
    }
}

pub struct TensorInfo<'model> {
    pub name: String,
    pub size: usize,
    pub usage: TensorUsage,
    pub dtype: TensorType
}

impl From<&'model ffi::nrt_tensor_info_t> for TensorInfo<'model> {
    fn from(info: &'model ffi::nrt_tensor_info_t) -> Self {
        TensorInfo {
            name: unsafe { CStr::from_ptr(info.name.as_ptr()).to_str().unwrap().to_string() },
            size: info.size,
            usage: TensorUsage::from(info.usage),
            dtype: TensorType::from(info.dtype)
        }
    }
}

pub struct Tensor<'model> {
    handle: *mut ffi::nrt_tensor_t,
    placement: TensorPlacement,
    nc_id: u32,
    info: TensorInfo<'model>,
    has_owned_storage: bool
}

impl<'model>Tensor<'model> {
    pub (crate) fn from_tensor_info_t(
        placement: TensorPlacement,
        nc_id: u32,
        model_tensor_info: &'model ffi::nrt_tensor_info_t,
    ) -> NrtResult<Self> {
        let mut handle = null_mut();
        let status = unsafe {
            nrt_tensor_allocate(
                tensor_placement as c_int,
                nc_id as c_int,
                info.size,
                info.name,
                &mut handle
            )
        } as u32;
        let has_owned_storage = true;
        let info = TensorInfo::from(model_tensor_info);
        nrt_result!(Tensor {handle,placement,nc_id,info,has_owned_storage}, status)
    }

    pub fn bind<T>(&mut self, data: &mut ArrayD<T>) -> NrtResult<()> {
        if self.has_owned_storage {
            warn!("Warning: Binding a user-managed buffer to a tensor frees existing storage and may lead to undefined behavior if buffer lifetime is incorrect.");
        }
        self.has_owned_storage = false;
        size = data.len() * std::mem::size_of::<T>();
        let status = unsafe {
            ffi::nrt_tensor_bind(
                self.handle as *mut c_void,
                data.as_mut_ptr() as *mut c_void,
                size
            )
        } as u32;
        if status == 0 {
            self.info.size = size;

        } else {
            error!("Failed to bind tensor: {}",status);
        }
        nrt_result!((),status)
    }
}
impl<'model> Drop for Tensor<'model> {
    fn drop(&mut self) -> {
        if !self.handle.is_null() {
            let status = unsafe { ffi::nrt_tensor_free(self.handle) };
            self.handle = null_mut();  
        }
    }
}


struct TensorSet<'model> {
    handle: *mut ffi::nrt_tensor_set_t,
    tensors: Vec<Tensor<'model>>
}
impl From<&'model ffi::nrt_tensor_set_t> for TensorSet<'model> {
    pub fn empty() -> NrtResult<Self> {
        let mut handle = null_mut();
        let status = unsafe {
            ffi::nrt_tensor_set_create(&mut handle)
        } as u32;
        let tensors = Vec::new();
        nrt_result!(TensorSet{handle,tensors},status)
        
    }
}