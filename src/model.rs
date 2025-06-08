use crate::{
    ffi,
    error::{NrtError, NrtResult}
};
use cxx::UniquePtr;
use std::ffi::c_void;
use std::pin::Pin;
use std::sync::Arc;

pub enum TensorUsage {
    NRT_TENSOR_USAGE_INPUT = 0,
    NRT_TENSOR_USAGE_OUTPUT = 1
}

pub struct Model {
    inner: UniquePtr<ffi::Model>,
}

impl Model {
    pub fn load(path: &str, start_nc: i32, nc_count: i32) -> NrtResult<Self> {
        let result: ffi::ModelResult = ffi::Model::load(path, start_nc, nc_count);
        if result.success() {
            Ok(Model { inner: result.value })
        } else {
            Err(NrtError::from(result.status))
        }
    }

    pub fn bind(
        &mut self,
        name: &str,
        usage: u32,
        buffer: *mut c_void,
    ) -> NrtResult<()> {
        let status: u32 = self.inner.pin_mut().bind(name, usage, buffer);
        if status == 0 {
            Ok(())
        } else {
            Err(NrtError::from(status))
        }
    }

    pub fn bind_slice(
        &mut self,
        name: &str,
        usage: u32,
        buffer: &mut [u8]
    ) -> NrtResult<()> {    
        let status: u32 = self.inner.pin_mut().bind_slice(name, usage, buffer);
        if status == 0 {
            Ok(())
        } else {
            Err(NrtError::from(status))
        }
    }

    pub fn execute(&mut self,inputs: &[(String,&[u8])],outputs: &[(String,&mut [u8])]) -> NrtResult<()> {
        for (name, buffer) in inputs {
            self.bind_slice(name, TensorUsage::NRT_TENSOR_USAGE_INPUT as u32, buffer)?;
        }
        for (name, buffer) in outputs {
            self.bind_slice(name, TensorUsage::NRT_TENSOR_USAGE_OUTPUT as u32, buffer)?;
        }
        let status: u32 = self.inner.pin_mut().execute();
        if status == 0 {
            Ok(())
        } else {
            Err(NrtError::from(status))
        }
    }
}
