use thiserror::Error;
use num_traits::FromPrimitive;

enum_from_primitive! {
    #[derive(Error, Debug, PartialEq)]
    pub enum NrtError {
        #[error("Runtime Error: {0}")]
        NRT_FAILURE = 1,
        #[error("Invalid NEFF, bad instruction, bad DMA descriptor, input tensor name/size does not match the model, etc: {0}")]
        NRT_INVALID = 2,
        #[error("Invalid handle (e.g. an invalid model handle): {0}")]
        NRT_INVALID_HANDLE = 3,
        #[error("Failed to allocate a resource for the requested operation: {0}")]
        NRT_RESOURCE = 4,
        #[error("Operation timed out: {0}")]
        NRT_TIMEOUT = 5,
        #[error("Hardware failure: {0}")]
        NRT_HW_ERROR = 6,
        #[error("Too many pending nrt_execute() requests. The runtime request queue is full. Cannot enqueue more nrt_execute() requests: {0}")]
        NRT_QUEUE_FULL = 7,
        #[error("The number of available NeuronCores is insufficient for the requested operation: {0}")]
        NRT_LOAD_NOT_ENOUGH_NC = 9,
        #[error("NEFF version unsupported: {0}")]
        NRT_UNSUPPORTED_NEFF_VERSION = 10,
        #[error("Returned when attempting an API call when the library is not initialized: {0}")]
        NRT_UNINITIALIZED = 13,
        #[error("Returned when attempting an API call after nrt_close() was called: {0}")]
        NRT_CLOSED = 14,
        #[error("Invalid input has been submitted to nrt_execute(): {0}")]
        NRT_EXEC_BAD_INPUT = 1002,
        #[error("Execution completed with numerical errors (produced NaN): {0}")]
        NRT_EXEC_COMPLETED_WITH_NUM_ERROR = 1003,
        #[error("Execution was completed with other errors, either logical (event double clear), or hardware (parity error): {0}")]
        NRT_EXEC_COMPLETED_WITH_ERROR = 1004,
        #[error("The neuron core is locked (in use) by another model/thread: {0}")]
        NRT_EXEC_NC_BUSY = 1005,
        #[error("One or more indirect memcopies and/or embedding updates are out of bound due to input corruptions: {0}")]
        NRT_OOB = 1006,
        #[error("Suspected hang in collectives operation due to hardware errors on this or other workers: {0}")]
        NRT_EXEC_HW_ERR_COLLECTIVES = 1200,
        #[error("An HBM suffered from an uncorrectable error and produced incorrect results: {0}")]
        NRT_EXEC_HW_ERR_HBM_UE = 1201
    }
}

impl From<NrtError> for u16 {
    fn from(error: NrtError) -> Self {
        error as u16
    }
}

impl From<u16> for NrtError {
    fn from(code: u16) -> Self {
        NrtError::from_u16(code as u16).unwrap_or(NrtError::NRT_FAILURE)
    }
}

pub type NrtResult<T> = Result<T, NrtError>;

impl From<u16,T> for NrtResult<T> {
    fn from(status: u16,result:T) -> Self {
        if status == 0 {
            return Ok(result);
        }
        Err(NrtError::from(code))
    }
}