use std::fmt::Display;

#[derive(Debug, thiserror::Error)]
pub enum LavaError {
    Io(#[from] std::io::Error),
    Bincode(#[from] bincode::Error),
    Compression(String),
    Tantivy(#[from] tantivy::TantivyError),
    Arrow(#[from] arrow::error::ArrowError),
    OpenDAL(#[from] opendal::Error),
    Parse(String),
    Unknown,
}

impl Display for LavaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LavaError::Io(err) => write!(f, "IO error: {}", err),
            LavaError::Bincode(err) => write!(f, "Bincode error: {}", err),
            LavaError::Compression(err) => write!(f, "Compression error: {}", err),
            LavaError::Tantivy(err) => write!(f, "Tantivy error: {}", err),
            LavaError::Arrow(err) => write!(f, "Arrow error: {}", err),
            LavaError::OpenDAL(err) => write!(f, "OpenDAL error: {}", err),
            LavaError::Parse(err) => write!(f, "Parse error: {}", err),
            LavaError::Unknown => write!(f, "Unkown error"),
        }
    }
}
