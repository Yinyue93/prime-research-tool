//! Error types for the prime research tool

use thiserror::Error;

/// Result type alias for prime tool operations
pub type Result<T> = std::result::Result<T, PrimeError>;

/// Errors that can occur during prime research operations
#[derive(Error, Debug)]
pub enum PrimeError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Range error: {0}")]
    RangeError(String),
    
    #[error("Factorization failed for {number}: {reason}")]
    FactorizationError { number: u64, reason: String },
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Computation timeout after {seconds} seconds")]
    Timeout { seconds: u64 },
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Overflow error: number too large for operation")]
    Overflow,
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl PrimeError {
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Self::InvalidInput(msg.into())
    }
    
    pub fn range_error<S: Into<String>>(msg: S) -> Self {
        Self::RangeError(msg.into())
    }
    
    pub fn factorization_error(number: u64, reason: &str) -> Self {
        Self::FactorizationError {
            number,
            reason: reason.to_string(),
        }
    }
    
    pub fn network_error<S: Into<String>>(msg: S) -> Self {
        Self::NetworkError(msg.into())
    }
    
    pub fn timeout(seconds: u64) -> Self {
        Self::Timeout { seconds }
    }
    
    pub fn parse_error<S: Into<String>>(msg: S) -> Self {
        Self::ParseError(msg.into())
    }
    
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        Self::Internal(msg.into())
    }
}