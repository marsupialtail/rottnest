use std::fmt::Display;

#[derive(Debug, thiserror::Error)]
pub enum LavaError {
    Io(#[from] std::io::Error),
    Bincode(#[from] bincode::Error),
    Compression(String),
    Arrow(#[from] arrow::error::ArrowError),
    OpenDAL(#[from] opendal::Error),
    Parse(String),
    Parquet(#[from] parquet::errors::ParquetError),
    Thrift(#[from] thrift::Error),
    Tokenizers(#[from] tokenizers::Error),
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
            LavaError::OpenDAL(err) => write!(f, "OpenDAL error: {}", err),
            LavaError::Parse(err) => write!(f, "Parse error: {}", err),
            LavaError::Unknown => write!(f, "Unkown error"),
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
