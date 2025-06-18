use std::fs::File;
use crate::error::{
    NrtError,
    NrtResult
};
use std::path::PathBuf;
use std::os::raw::c_void;
use memmap2::{Mmap, MmapOptions};


pub struct MemMappedROBuffer {
    buffer: Mmap
}

impl MemMappedROBuffer {
    pub fn from_file(path: PathBuf) -> NrtResult<Self> {
        match File::open(&path) {
            Ok(file) => {
                match unsafe { MmapOptions::new().map(&file) } {
                    Ok(buffer) => Ok(MemMappedROBuffer { buffer }),
                    Err(e) => Err(NrtError::NRT_FAILURE)
                }
            },
            Err(e) => Err(NrtError::NRT_FAILURE)
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