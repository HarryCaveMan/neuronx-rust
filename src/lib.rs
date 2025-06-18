#[macro_use]
extern crate enum_primitive;

#[macro_use]
mod macros;

pub (crate) mod ffi;
pub (crate) mod data;
pub mod error;
pub mod model;
pub mod runtime;

#[cfg(feature = "tokenizer")]
pub mod tokenizer;
#[cfg(feature = "service")]
pub mod service;