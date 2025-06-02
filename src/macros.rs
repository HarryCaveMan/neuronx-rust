use super::error::{
    NrtError,
    NrtResult
};

#[macro_export]
macro_rules! nrt_try {
    ($status:expr) => {
        if $status == 0 {
            Ok(())
        } else {
            Err(NrtError::from($status))
        }
    };
    ($status:expr, $value:expr) => {
        if $status == 0 {
            Ok($value)
        } else {
            Err(NrtError::from($status))
        }
    };
}