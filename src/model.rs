use crate::{
    ffi,
    error::{NrtError, NrtResult}
};
use cxx::UniquePtr;

pub enum TensorUsage {
    NRT_TENSOR_USAGE_INPUT = 0,
    NRT_TENSOR_USAGE_OUTPUT = 1
}

pub struct Model<'nrt> {
    inner: UniquePtr<'nrt,ffi::Model>,
}

unsafe impl<'nrt> Send for Model<'nrt> {}
unsafe impl<'nrt> Sync for Model<'nrt> {}

impl<'nrt> Model<'nrt> {
    pub fn load(path: &str, start_nc: i32, nc_count: i32) -> NrtResult<Self> {
        let result: ffi::ModelResult = ffi::Model::load(path, start_nc, nc_count);
        if result.success() {
            Ok(Model { inner: result.value })
        } else {
            Err(NrtError::from(result.status))
        }
    }

    pub fn get_new_io_tensors(&self) -> NrtResult<IoTensors<'nrt>> {
        let result: ffi::IoTensorsResult = self.inner.get_new_io_tensors();
        if result.success() {
            Ok(IoTensors { inner: result.value })
        } else {
            Err(NrtError::from(result.status))
        }
    }

    pub fn execute(&mut self, io_tensors: &mut IoTensors<'nrt>) -> NrtResult<()> {
        let status: u32 = self.inner.pin_mut().execute(io_tensors.inner.pin_mut());
        if status == 0 {
            Ok(())
        } else {
            Err(NrtError::from(status))
        }
    }
}
