use crate::ffi;
use crate::error::{NrtError, NrtResult};

pub struct NrtVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub maintenance: u64,
    pub detail: String,
    pub git_hash: String,
}
impl From<&ffi::NrtVersion> for NrtVersion {
    fn from(version: &ffi::NrtVersion) -> Self {
        NrtVersion {
            major: version.major(),
            minor: version.minor(),
            patch: version.patch(),
            maintenance: version.maintenance(),
            detail: version.detail(),
            git_hash: version.git_hash(),
        }
    }
}

pub fn init() -> NrtResult<()> {
    let status: u32 = ffi::neuronx_init();
    if status == 0 {
        Ok(())
    } else {
        Err(NrtError::from(status))
    }
}
pub fn close() -> NrtResult<()> {
    let status: u32 = ffi::neuronx_close();
    if status == 0 {
        Ok(())
    } else {
        Err(NrtError::from(status))
    }
}
pub fn version() -> NrtResult<NrtVersion> {
    let result: ffi::NrtVersionResult = ffi::neuronx_version();
    if result.success() {
        Ok(NrtVersion::from(&result.value))
    } else {
        Err(NrtError::from(result.status))
    }
}
pub fn get_nc_count() -> NrtResult<u32> {
    let result: ffi::Uint32Result = ffi::neuronx_get_nc_count();
    if result.success() {
        Ok(result.value)
    } else {
        Err(NrtError::from(result.status))
    }
}
pub fn get_visible_nc_count() -> NrtResult<u32> {
    let result: ffi::Uint32Result = ffi::neuronx_get_visible_nc_count();
    if result.success() {
        Ok(result.value)
    } else {
        Err(NrtError::from(result.status))
    }
}