use crate::ffi;
use crate::error::{
    NrtError,
    NrtResult
};
use std::ffi::{
    CString,
    CStr
};
use std::os::raw::c_char;

#[derive(Debug, Clone, PartialEq)]
pub struct NrtVersionInfo {
    pub rt_major: u64,
    pub rt_minor: u64,
    pub rt_patch: u64,
    pub rt_maintenance: u64,
    pub rt_detail: String,
    pub git_hash: String,
}

impl From<ffi::nrt_version_t> for NrtVersionInfo {
    fn from(version: ffi::nrt_version_t) -> Self {
        NrtVersionInfo {
            rt_major: version.rt_major,
            rt_minor: version.rt_minor,
            rt_patch: version.rt_patch,
            rt_maintenance: version.rt_maintenance,
            rt_detail: unsafe { CStr::from_ptr(version.rt_detail.as_ptr()).to_str().unwrap().to_string() },
            git_hash: unsafe { CStr::from_ptr(version.git_hash.as_ptr()).to_str().unwrap().to_string() }
        }
    }
}

pub fn init() -> NrtResult<()> {
    let status = unsafe {
        ffi::nrt_init(
            ffi::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_NO_FW,
            CString::new("").unwrap().as_ptr(),
            CString::new("").unwrap().as_ptr()
        )
    } as u32;
    nrt_wrap_status!((),status)
}

pub fn close() {
    unsafe { ffi::nrt_close() };
}

pub fn get_version() -> NrtResult<NrtVersionInfo> {
    let mut version: ffi::nrt_version_t = ffi::nrt_version_t {
        rt_major: 0,
        rt_minor: 0,
        rt_patch: 0,
        rt_maintenance: 0,
        rt_detail: [0 as c_char; ffi::RT_VERSION_DETAIL_LEN as usize],
        git_hash: [0 as c_char; ffi::GIT_HASH_LEN as usize],
    };
    let status = unsafe {
        ffi::nrt_get_version(&mut version,size_of::<ffi::nrt_version_t>())
    };
    nrt_wrap_status!(NrtVersionInfo::from(version),status)
}

pub fn get_nc_count() -> NrtResult<u32> {
    let mut count: u32 = 0;
    let status = unsafe {
        ffi::nrt_get_total_nc_count(&mut count)
    };
    nrt_wrap_status!(count,status)
}

pub fn get_total_vnc_count() -> NrtResult<u32> {
    let mut count: u32 = 0;
    let status = unsafe { ffi::nrt_get_total_vnc_count(&mut count) };
    nrt_wrap_status!(count,status)
}

pub fn get_visible_nc_count() -> NrtResult<u32> {
    let mut count: u32 = 0;
    let status = unsafe { ffi::nrt_get_visible_nc_count(&mut count) };
    nrt_wrap_status!(count,status)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{
        NrtError,
        NrtResult
    };

    #[cfg(hardware_tests)]
    #[test]
    fn test_init() {
        match init() {
            Ok(_) => close(),
            Err(err) => {
                close();
                panic!("Failed to initialize runtime: {:?}", err);
            }
        }
    }

    #[test]
    fn test_get_version() {
        match get_version() {
            Ok(version) => println!("Neuron runtime version: {:?}", version),
            Err(err) => panic!("Failed to get version: {:?}", err)
        }
    }

    #[cfg(hardware_tests)]
    #[test]
    fn test_get_nc_count() {
        match get_nc_count() {
            Ok(count) => println!("Neuron core count: {:?}", count),
            Err(err) => panic!("Failed to get core count: {:?}", err)
        }
    }

    #[cfg(hardware_tests)]
    #[test]
    fn test_get_total_vnc_count() {
        match get_total_vnc_count() {
            Ok(count) => println!("Total virtual core count: {:?}", count),
            Err(err) => panic!("Failed to get total virtual core count: {:?}", err)
        }
    }

    #[test]
    fn test_get_visible_nc_count() {
        match get_visible_nc_count() {
            Ok(count) => println!("Visible core count: {:?}", count),
            Err(err) => panic!("Failed to get visible core count: {:?}", err)
        }
    }
}