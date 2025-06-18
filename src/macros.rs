macro_rules! nrt_wrap_status {
    ($val:expr, $status:ident) => (
        if $status == 0 {
            Ok($val)
        } else {
            use crate::error::NrtError;
            Err(NrtError::from($status))
        }
    )
}