//! Fast batch prediction for LightGBM models, bit-identical to LightGBM.

mod avx512;
pub mod forest;
pub mod model;
#[cfg(feature = "python")]
mod python;

pub use forest::{Forest, Missing, Node};
pub use model::{Model, Output};

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A valid model, input or option that fastleaves does not support.
    Unsupported(String),
    /// A malformed model.
    Invalid(String),
}

impl Error {
    pub(crate) fn unsupported(what: String) -> Self {
        Error::Unsupported(format!("not supported: {what}"))
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Unsupported(m) | Error::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

/// The SIMD path predictions use: "avx512", or "scalar" (no AVX-512, or FASTLEAVES_ISA=scalar).
pub fn isa() -> &'static str {
    if avx512::Avx512::detect().is_some() { "avx512" } else { "scalar" }
}
