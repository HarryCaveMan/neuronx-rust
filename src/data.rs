use crate::{
    ffi,
    error::{NrtError, NrtResult}
};

use num_traits::FromPrimitive;

use cxx::UniquePtr;

enum_from_primitive! {
    #[derive(Debug, PartialEq)]
    pub enum TensorUsage {
        NRT_TENSOR_USAGE_INPUT = 0,
        NRT_TENSOR_USAGE_OUTPUT = 1
    }
}
impl From<u32> for TensorUsage {
    fn from(value: u32) -> NrtResult<Self> {
        from_u32(value).unwrap_or(NrtError::NRT_INVALID)
    }
}
impl From<TensorUsage> for u32 {
    fn from(usage: TensorUsage) -> Self {
        usage as u32
    }
}

pub struct IoTensors<'nrt> {
    inner: UniquePtr<'nrt, ffi::IoTensors>
}

pub struct TensorInfo<'_> {
    pub name: String,
    pub usage: TensorUsage,
    pub size: usize,
    pub data_type: u32,
    pub ndim: u32,
    pub shape: &'_ [u32]
}

impl From<&'_ ffi::TensorInfo> for TensorInfo<'_> {
    fn from(tensor_info: &ffi::TensorInfo) -> NrtResult<Self> {
        TensorInfo {
            name: tensor_info.name().to_string(),
            usage: TensorUsage::from(tensor_info.usage())?,
            size: tensor_info.size(),
            data_type: tensor_info.data_type(),
            ndim: tensor_info.ndim(),
            shape: tensor_info.shape_slice(),
        }
    }
}

unsafe impl<'nrt> Send for IoTensors<'nrt> {}
unsafe impl<'nrt> Sync for IoTensors<'nrt> {}

impl<'nrt> IoTensors<'nrt> {
    pub fn get_input_tensor_info(&self) -> NrtResult<Vec<TensorInfo<'_>>> {
        let info_result = self.inner.get_input_tensor_info_slice();
        if info_result.success() {
            Ok(info_result.iter().map(|info| TensorInfo::from(info))?.collect())
        } else {
            Err(NrtError::from(info_result.status))
        }    
    }

    pub fn get_output_tensor_info(&self) -> NrtResult<Vec<TensorInfo<'_>>> {
        let info_result = self.inner.get_output_tensor_info_slice();
        if info_result.success() {
            Ok(info_result.iter().map(|info| TensorInfo::from(info))?.collect())
        } else {
            Err(NrtError::from(info_result.status))
        }
    }

    pub fn bind(
        &mut self,
        name: &str,
        usage: TensorUsage,
        buffer: &mut [u8],
    ) -> NrtResult<()> {
        let status = self.inner.bind(name, usage as u32, buffer);
        if status == 0 {
            Ok(())
        } else {
            Err(NrtError::from(status))
        }
    }
}