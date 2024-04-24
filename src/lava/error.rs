use std::fmt::Display;

#[derive(Debug, thiserror::Error)]
pub enum LavaError {
    Io(#[from] std::io::Error),
    Bincode(#[from] bincode::Error),
    Compression(String),
    Arrow(#[from] arrow::error::ArrowError),
    #[cfg(feature = "opendal")]
    OpenDAL(#[from] opendal::Error),
    #[cfg(feature = "aws_sdk")]
    AwsSdk(String),
    Parse(String),
    Parquet(#[from] parquet::errors::ParquetError),
    Thrift(#[from] thrift::Error),
    Tokenizers(#[from] tokenizers::Error),
    Unsupported(String),
    Unknown,
    #[cfg(feature = "py")]
    Pyo3(#[from] pyo3::PyErr),
}

impl Display for LavaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LavaError::Io(err) => write!(f, "IO error: {}", err),
            LavaError::Bincode(err) => write!(f, "Bincode error: {}", err),
            LavaError::Compression(err) => write!(f, "Compression error: {}", err),
            LavaError::Arrow(err) => write!(f, "Arrow error: {}", err),
            #[cfg(feature = "opendal")]
            LavaError::OpenDAL(err) => write!(f, "OpenDAL error: {}", err),
            #[cfg(feature = "aws_sdk")]
            LavaError::AwsSdk(err) => write!(f, "AWS SDK error: {}", err),
            LavaError::Parse(err) => write!(f, "Parse error: {}", err),
            LavaError::Unknown => write!(f, "Unkown error"),
            LavaError::Unsupported(err) => write!(f, "Unsupported error: {}", err),
            LavaError::Parquet(err) => write!(f, "Parquet error: {}", err),
            LavaError::Thrift(err) => write!(f, "Thrift error: {}", err),
            LavaError::Tokenizers(err) => write!(f, "Tokenizers error: {}", err),
            #[cfg(feature = "py")]
            LavaError::Pyo3(err) => write!(f, "Pyo3 error: {}", err),
        }
    }
}

#[cfg(feature = "py")]
impl From<LavaError> for pyo3::PyErr {
    fn from(e: LavaError) -> pyo3::PyErr {
        pyo3::exceptions::PyOSError::new_err(e.to_string())
    }
}
